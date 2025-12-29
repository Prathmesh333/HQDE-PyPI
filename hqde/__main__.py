"""
HQDE Package Main Entry Point

This module allows running the HQDE package directly using:
python -m hqde
"""

import sys
import argparse
import logging
from examples.cifar10_synthetic_test import CIFAR10SyntheticTrainer

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hqde_runtime.log')
        ]
    )

def main():
    """Main entry point for HQDE package."""
    parser = argparse.ArgumentParser(description='HQDE: Hierarchical Quantum-Distributed Ensemble Learning')
    parser.add_argument('--mode', choices=['test', 'demo'], default='test',
                       help='Run mode: test (comprehensive) or demo (quick)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of distributed workers')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of training samples')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting HQDE Framework")
    logger.info(f"Configuration: mode={args.mode}, workers={args.workers}, epochs={args.epochs}")

    try:
        if args.mode == 'test':
            # Run comprehensive test
            trainer = CIFAR10SyntheticTrainer(num_workers=args.workers)
            results = trainer.run_comprehensive_test(
                train_samples=args.samples,
                test_samples=args.samples // 5,
                batch_size=64,
                num_epochs=args.epochs
            )

            logger.info("HQDE Test completed successfully!")
            logger.info(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
            logger.info(f"Training Time: {results['training_time']:.2f} seconds")

        elif args.mode == 'demo':
            # Run quick demo
            trainer = CIFAR10SyntheticTrainer(num_workers=min(args.workers, 2))
            results = trainer.run_comprehensive_test(
                train_samples=1000,
                test_samples=200,
                batch_size=32,
                num_epochs=2
            )

            logger.info("HQDE Demo completed successfully!")
            logger.info(f"Demo Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")

    except KeyboardInterrupt:
        logger.info("HQDE execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"HQDE execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()