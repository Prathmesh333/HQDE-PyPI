#!/usr/bin/env python3
"""
Simple import test for HQDE package.
Tests basic functionality without external dependencies.
"""

def test_imports():
    """Test that all main components can be imported."""
    try:
        import hqde
        print("[OK] HQDE package imported successfully")

        from hqde import create_hqde_system
        print("[OK] create_hqde_system imported successfully")

        # Test core components
        from hqde import HQDESystem, AdaptiveQuantizer, QuantumInspiredAggregator
        print("[OK] Core classes imported successfully")

        # Test quantum components
        from hqde.quantum import QuantumEnsembleAggregator, QuantumNoiseGenerator
        print("[OK] Quantum components imported successfully")

        # Test distributed components
        from hqde.distributed import MapReduceEnsembleManager, ByzantineFaultTolerantAggregator
        print("[OK] Distributed components imported successfully")

        print("[OK] All import tests passed!")
        return True

    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)