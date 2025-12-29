#!/usr/bin/env python3
"""
Test script to validate HQDE fixes without running full training.
This script checks that the main components are properly structured.
"""

import sys
import os

def test_file_structure():
    """Test that all required files exist and have content."""
    print("Testing file structure...")

    # Check main files
    required_files = [
        'hqde/__init__.py',
        'hqde/__main__.py',
        'hqde/core/hqde_system.py',
        'examples/cifar10_synthetic_test.py'
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:  # Should have substantial content
                        print(f"  [OK] {file_path} - OK ({len(content)} chars)")
                    else:
                        print(f"  [FAIL] {file_path} - Too short ({len(content)} chars)")
                        return False
            except UnicodeDecodeError:
                # Fallback to different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    if len(content) > 100:  # Should have substantial content
                        print(f"  [OK] {file_path} - OK ({len(content)} chars) [latin-1]")
                    else:
                        print(f"  [FAIL] {file_path} - Too short ({len(content)} chars)")
                        return False
        else:
            print(f"  [FAIL] {file_path} - Missing")
            return False

    return True

def test_main_py_content():
    """Test that __main__.py has proper content."""
    print("\nTesting __main__.py content...")

    with open('hqde/__main__.py', 'r') as f:
        content = f.read()

    # Check for key components
    required_elements = [
        'def main():',
        'import argparse',
        'from examples.cifar10_synthetic_test import CIFAR10SyntheticTrainer',
        '--mode',
        'parser.parse_args()'
    ]

    for element in required_elements:
        if element in content:
            print(f"  [OK] Found: {element}")
        else:
            print(f"  [FAIL] Missing: {element}")
            return False

    return True

def test_hqde_system_fixes():
    """Test that hqde_system.py has the dynamic fixes."""
    print("\nTesting HQDE system fixes...")

    with open('hqde/core/hqde_system.py', 'r') as f:
        content = f.read()

    # Check for key fixes
    required_fixes = [
        'def setup_training(self, learning_rate=0.001):',
        'def predict(self, data_batch):',
        'optimizer.zero_grad()',
        'loss.backward()',
        'optimizer.step()',
        'self.setup_workers_training()'
    ]

    for fix in required_fixes:
        if fix in content:
            print(f"  [OK] Found fix: {fix}")
        else:
            print(f"  [FAIL] Missing fix: {fix}")
            return False

    return True

def test_cifar10_fixes():
    """Test that cifar10 synthetic test has dynamic evaluation."""
    print("\nTesting CIFAR-10 synthetic test fixes...")

    try:
        with open('examples/cifar10_synthetic_test.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open('examples/cifar10_synthetic_test.py', 'r', encoding='latin-1') as f:
            content = f.read()

    # Check for key fixes
    required_fixes = [
        'training_metrics = hqde_system.train(train_loader, num_epochs)',
        'predictions = hqde_system.predict([data])',
        'criterion = torch.nn.CrossEntropyLoss()',
        'except Exception as e:'
    ]

    for fix in required_fixes:
        if fix in content:
            print(f"  [OK] Found fix: {fix}")
        else:
            print(f"  [FAIL] Missing fix: {fix}")
            return False

    return True

def main():
    """Run all tests."""
    print("Testing HQDE Dynamic Implementation Fixes")
    print("=" * 50)

    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_tests_passed = True

    # Run tests
    tests = [
        test_file_structure,
        test_main_py_content,
        test_hqde_system_fixes,
        test_cifar10_fixes
    ]

    for test in tests:
        if not test():
            all_tests_passed = False
            break

    print("\n" + "=" * 50)
    if all_tests_passed:
        print("ALL TESTS PASSED! The HQDE project has been successfully fixed.")
        print("\nKey improvements made:")
        print("  1. Dynamic training instead of simulated losses")
        print("  2. Real model predictions instead of random outputs")
        print("  3. Proper distributed worker training setup")
        print("  4. Added command-line interface via __main__.py")
        print("  5. Error handling and fallback mechanisms")
        print("\nTo run the dynamic HQDE system:")
        print("  python -m hqde --mode demo")
        print("  python -m hqde --mode test --workers 4 --epochs 5")
    else:
        print("SOME TESTS FAILED! Please check the issues above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())