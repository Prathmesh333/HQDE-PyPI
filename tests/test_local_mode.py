import os
import tempfile
import unittest

import torch
import torch.nn as nn

from hqde import create_hqde_system
from hqde.core.hqde_system import (
    AdaptiveQuantizer,
    DistributedEnsembleManager,
    validate_training_config,
)


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


class TinyBNModel(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)


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
            validate_training_config({"training_aggregation": "unknown"})
        with self.assertRaises(ValueError):
            validate_training_config({"warmup_epochs": -1})
        with self.assertRaises(ValueError):
            validate_training_config({"server_optimizer": "adaptive"})
        with self.assertRaises(ValueError):
            validate_training_config({"federated_normalization": "batchnorm_only"})
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

    def test_fedavg_workers_start_from_synchronized_weights(self):
        system = create_hqde_system(
            TinyModel,
            {"num_classes": 3},
            num_workers=2,
            training_config={
                "use_amp": False,
                "ensemble_mode": "fedavg",
                "batch_assignment": "split",
            },
        )
        try:
            worker_states = system.ensemble_manager._resolve(
                [
                    system.ensemble_manager._worker_call(worker, "get_weights")
                    for worker in system.ensemble_manager.workers
                ]
            )
            self.assertEqual(len(worker_states), 2)
            self.assertTrue(self._state_equal(worker_states[0], worker_states[1]))
        finally:
            system.cleanup()

    def test_fedbn_broadcast_preserves_worker_local_batch_norm_state(self):
        system = create_hqde_system(
            TinyBNModel,
            {"num_classes": 3},
            num_workers=2,
            training_config={
                "use_amp": False,
                "ensemble_mode": "fedavg",
                "batch_assignment": "split",
                "federated_normalization": "local_bn",
                "server_optimizer": "fedadam",
            },
        )
        try:
            local_bn_keys = sorted(system.ensemble_manager.local_normalization_keys)
            self.assertIn("features.1.running_mean", local_bn_keys)
            self.assertIn("features.1.weight", local_bn_keys)

            worker_states = system.ensemble_manager._resolve(
                [
                    system.ensemble_manager._worker_call(worker, "get_weights")
                    for worker in system.ensemble_manager.workers
                ]
            )
            updated_worker_states = []
            for worker_index, state in enumerate(worker_states):
                mutated = {name: tensor.clone() for name, tensor in state.items()}
                for name in local_bn_keys:
                    if torch.is_floating_point(mutated[name]):
                        mutated[name].fill_(float(worker_index + 1))
                    else:
                        mutated[name].fill_(worker_index + 1)
                updated_worker_states.append(mutated)

            for worker, state in zip(system.ensemble_manager.workers, updated_worker_states):
                system.ensemble_manager._resolve(
                    [system.ensemble_manager._worker_call(worker, "set_weights", state)]
                )

            new_global_state = {name: tensor.clone() for name, tensor in worker_states[0].items()}
            new_global_state["features.0.weight"].zero_()
            new_global_state["head.weight"].zero_()

            system.ensemble_manager.broadcast_weights(new_global_state, preserve_local_norm=True)

            after_states = system.ensemble_manager._resolve(
                [
                    system.ensemble_manager._worker_call(worker, "get_weights")
                    for worker in system.ensemble_manager.workers
                ]
            )
            self.assertTrue(torch.equal(after_states[0]["features.0.weight"], new_global_state["features.0.weight"]))
            self.assertTrue(torch.equal(after_states[1]["features.0.weight"], new_global_state["features.0.weight"]))
            self.assertTrue(torch.equal(after_states[0]["head.weight"], new_global_state["head.weight"]))
            self.assertTrue(torch.equal(after_states[1]["head.weight"], new_global_state["head.weight"]))

            for worker_index, after_state in enumerate(after_states):
                for name in local_bn_keys:
                    self.assertTrue(torch.equal(after_state[name], updated_worker_states[worker_index][name]))
        finally:
            system.cleanup()

    def test_delta_quantizer_uses_blockwise_compression_and_skip_rules(self):
        quantizer = AdaptiveQuantizer(
            base_bits=12,
            min_bits=8,
            max_bits=16,
            block_size=16,
            warmup_rounds=0,
            min_tensor_elements=1,
        )

        quantized_delta, quantized_metadata = quantizer.quantize_delta(
            "layer.weight",
            torch.randn(64),
            worker_id=0,
            round_index=1,
            total_rounds=10,
        )
        skipped_delta, skipped_metadata = quantizer.quantize_delta(
            "layer.bias",
            torch.randn(16),
            worker_id=0,
            round_index=1,
            total_rounds=10,
        )

        self.assertEqual(tuple(quantized_delta.shape), (64,))
        self.assertTrue(quantized_metadata["quantized"])
        self.assertGreater(quantized_metadata["compression_ratio"], 1.0)
        self.assertGreater(quantized_metadata["original_bytes"], quantized_metadata["transmitted_bytes"])
        self.assertIn("0::layer.weight", quantizer.residual_buffers)

        self.assertEqual(tuple(skipped_delta.shape), (16,))
        self.assertFalse(skipped_metadata["quantized"])
        self.assertEqual(skipped_metadata["compression_ratio"], 1.0)
        self.assertEqual(skipped_metadata["original_bytes"], skipped_metadata["transmitted_bytes"])

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
            quantization_config={
                "base_bits": 12,
                "min_bits": 8,
                "max_bits": 16,
                "warmup_rounds": 0,
                "min_tensor_elements": 1,
                "block_size": 16,
            },
            training_config={
                "use_amp": False,
                "ensemble_mode": "fedavg",
                "batch_assignment": "split",
            },
        )
        try:
            metrics = system.train(loader, num_epochs=1)
            self.assertIn("compression_ratio", metrics)
            self.assertGreater(metrics["compression_ratio"], 1.0)
            quantization_metrics = system.ensemble_manager.get_quantization_metrics()
            self.assertGreater(quantization_metrics["original_bytes"], 0.0)
            self.assertGreater(quantization_metrics["transmitted_bytes"], 0.0)
            self.assertGreater(
                quantization_metrics["original_bytes"],
                quantization_metrics["transmitted_bytes"],
            )
            self.assertEqual(system.ensemble_manager.server_step, 1)
            self.assertTrue(system.ensemble_manager.server_first_moment)
            self.assertTrue(system.ensemble_manager.server_second_moment)
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
