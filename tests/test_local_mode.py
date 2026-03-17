import os
import tempfile
import unittest

import torch
import torch.nn as nn

from hqde import create_hqde_system
from hqde.core.hqde_system import DistributedEnsembleManager, validate_training_config


class TinyModel(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LocalModeTests(unittest.TestCase):
    def _build_loader(self):
        return [
            (torch.randn(6, 3, 16, 16), torch.randint(0, 3, (6,))),
            (torch.randn(6, 3, 16, 16), torch.randint(0, 3, (6,))),
        ]

    @staticmethod
    def _state_equal(left, right):
        if left.keys() != right.keys():
            return False
        return all(torch.equal(left[name], right[name]) for name in left)

    def test_training_config_validation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_training_config({"ensemble_mode": "unknown"})
        with self.assertRaises(ValueError):
            validate_training_config({"learning_rate": 0.0})
        with self.assertRaises(ValueError):
            validate_training_config({"prediction_aggregation": "invalid"})
        with self.assertRaises(ValueError):
            validate_training_config({"warmup_epochs": -1})
        with self.assertRaises(ValueError):
            validate_training_config({"compile_mode": "fastest"})

    def test_quantizer_is_only_enabled_for_fedavg_mode(self):
        independent_system = create_hqde_system(
            TinyModel,
            {"num_classes": 3},
            num_workers=2,
            quantization_config={"base_bits": 8, "min_bits": 4, "max_bits": 16},
            training_config={"use_amp": False, "ensemble_mode": "independent"},
        )
        fedavg_system = create_hqde_system(
            TinyModel,
            {"num_classes": 3},
            num_workers=2,
            quantization_config={"base_bits": 8, "min_bits": 4, "max_bits": 16},
            training_config={"use_amp": False, "ensemble_mode": "fedavg", "batch_assignment": "split"},
        )
        try:
            self.assertIsNone(independent_system.quantizer)
            self.assertIsNotNone(fedavg_system.quantizer)
        finally:
            independent_system.cleanup()
            fedavg_system.cleanup()

    def test_batch_assignment_contract_matches_documentation(self):
        data = torch.randn(6, 3, 16, 16)
        targets = torch.randint(0, 3, (6,))
        replicate_manager = DistributedEnsembleManager(
            num_workers=3,
            training_config={"use_amp": False, "batch_assignment": "replicate"},
        )
        split_manager = DistributedEnsembleManager(
            num_workers=3,
            training_config={
                "use_amp": False,
                "ensemble_mode": "fedavg",
                "batch_assignment": "split",
            },
        )
        try:
            replicate_batches = replicate_manager._build_worker_batches(data, targets)
            self.assertEqual(len(replicate_batches), 3)
            for worker_data, worker_targets in replicate_batches:
                self.assertTrue(torch.equal(worker_data, data))
                self.assertTrue(torch.equal(worker_targets, targets))

            split_batches = split_manager._build_worker_batches(data, targets)
            self.assertEqual(len(split_batches), 3)
            self.assertEqual(sum(batch_data.size(0) for batch_data, _ in split_batches), data.size(0))
            self.assertEqual(sum(batch_targets.size(0) for _, batch_targets in split_batches), targets.size(0))
            self.assertTrue(torch.equal(torch.cat([batch_data for batch_data, _ in split_batches], dim=0), data))
            self.assertTrue(
                torch.equal(torch.cat([batch_targets for _, batch_targets in split_batches], dim=0), targets)
            )
        finally:
            replicate_manager.shutdown()
            split_manager.shutdown()

    def test_local_train_predict_save_and_load_preserves_worker_state(self):
        loader = self._build_loader()
        system = create_hqde_system(
            TinyModel,
            {"num_classes": 3},
            num_workers=2,
            training_config={
                "use_amp": False,
                "batch_assignment": "replicate",
                "ensemble_mode": "independent",
            },
        )
        try:
            metrics = system.train(loader, num_epochs=1)
            self.assertEqual(len(metrics["epoch_history"]), 1)

            predictions = system.predict([torch.randn(4, 3, 16, 16)])
            self.assertEqual(tuple(predictions.shape), (4, 3))

            checkpoints_before = system.ensemble_manager.get_worker_checkpoints()
            self.assertEqual(len(checkpoints_before), 2)

            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
                checkpoint_path = handle.name

            try:
                system.save_model(checkpoint_path)
                system.load_model(checkpoint_path)
                checkpoints_after = system.ensemble_manager.get_worker_checkpoints()
            finally:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)

            self.assertEqual(len(checkpoints_after), len(checkpoints_before))
            for before, after in zip(checkpoints_before, checkpoints_after):
                self.assertTrue(
                    self._state_equal(before["model_state_dict"], after["model_state_dict"])
                )
                self.assertAlmostEqual(before["efficiency_score"], after["efficiency_score"], places=6)
        finally:
            system.cleanup()

    def test_fedavg_mode_reports_quantization_metrics(self):
        loader = self._build_loader()
        system = create_hqde_system(
            TinyModel,
            {"num_classes": 3},
            num_workers=2,
            quantization_config={"base_bits": 8, "min_bits": 4, "max_bits": 16},
            training_config={
                "use_amp": False,
                "ensemble_mode": "fedavg",
                "batch_assignment": "split",
            },
        )
        try:
            metrics = system.train(loader, num_epochs=1)
            self.assertIn("compression_ratio", metrics)
            self.assertGreaterEqual(metrics["compression_ratio"], 1.0)
        finally:
            system.cleanup()

    def test_use_amp_flag_is_safe_without_cuda(self):
        loader = self._build_loader()
        system = create_hqde_system(
            TinyModel,
            {"num_classes": 3},
            num_workers=2,
            training_config={
                "use_amp": True,
                "ensemble_mode": "independent",
                "batch_assignment": "replicate",
            },
        )
        try:
            metrics = system.train(loader, num_epochs=1)
            self.assertEqual(len(metrics["epoch_history"]), 1)
        finally:
            system.cleanup()


if __name__ == "__main__":
    unittest.main()
