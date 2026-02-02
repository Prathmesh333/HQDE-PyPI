#!/usr/bin/env python3
"""
Simple import test for HQDE package.
Tests that all main modules can be imported successfully.
"""

import sys

def test_imports():
    """Test that all main HQDE modules can be imported."""
    print("Testing HQDE imports...")
    
    try:
        # Test main package import
        import hqde
        print("  [PASS] import hqde")
        
        # Test core module
        from hqde.core import hqde_system
        print("  [PASS] from hqde.core import hqde_system")
        
        # Test quantum modules
        from hqde.quantum import quantum_aggregator
        print("  [PASS] from hqde.quantum import quantum_aggregator")
        
        from hqde.quantum import quantum_noise
        print("  [PASS] from hqde.quantum import quantum_noise")
        
        from hqde.quantum import quantum_optimization
        print("  [PASS] from hqde.quantum import quantum_optimization")
        
        # Test utils modules
        from hqde.utils import config_manager
        print("  [PASS] from hqde.utils import config_manager")
        
        from hqde.utils import data_utils
        print("  [PASS] from hqde.utils import data_utils")
        
        from hqde.utils import performance_monitor
        print("  [PASS] from hqde.utils import performance_monitor")
        
        from hqde.utils import visualization
        print("  [PASS] from hqde.utils import visualization")
        
        # Test main function
        from hqde import create_hqde_system
        print("  [PASS] from hqde import create_hqde_system")
        
        # Check version
        if hasattr(hqde, '__version__'):
            print(f"\n  HQDE Version: {hqde.__version__}")
        
        print("\nAll imports successful!")
        return 0
        
    except ImportError as e:
        print(f"\n  [FAIL] Import error: {e}")
        return 1
    except Exception as e:
        print(f"\n  [FAIL] Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(test_imports())
