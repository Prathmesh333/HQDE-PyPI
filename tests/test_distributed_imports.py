import importlib
import unittest


class DistributedImportTests(unittest.TestCase):
    def test_direct_distributed_submodule_imports_work_without_ray(self):
        modules = [
            "hqde.distributed",
            "hqde.distributed.mapreduce_ensemble",
            "hqde.distributed.hierarchical_aggregator",
            "hqde.distributed.load_balancer",
            "hqde.distributed.fault_tolerance",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)

    def test_ray_backed_manager_raises_clean_error_when_ray_missing(self):
        mapreduce_module = importlib.import_module("hqde.distributed.mapreduce_ensemble")
        with self.assertRaises(ImportError):
            mapreduce_module.MapReduceEnsembleManager()


if __name__ == "__main__":
    unittest.main()
