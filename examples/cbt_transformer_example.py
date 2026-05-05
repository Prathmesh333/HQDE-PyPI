"""
HQDE Framework - CBT Transformer Classification Example

This example demonstrates how to use HQDE with transformer models for
Cognitive Behavioral Therapy (CBT) text classification tasks.

Task: Classify text into different types of cognitive distortions
"""

import torch
import torch.nn as nn
import logging
import time
from typing import List, Tuple

# Import HQDE components
from hqde import create_hqde_system, PerformanceMonitor
from hqde.models.transformers import (
    CBTTransformerClassifier,
    LightweightTransformerClassifier,
    TransformerTextClassifier
)
from hqde.utils.text_data_utils import (
    SimpleTokenizer,
    CBTDataset,
    TextDataLoader,
    create_cbt_sample_data,
    preprocess_cbt_text
)
from hqde.utils.transformer_presets import (
    make_cbt_training_config,
    make_transformer_training_config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_cbt_cognitive_distortion_data(
    num_train: int = 1000,
    num_test: int = 200
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Create sample CBT cognitive distortion dataset.
    
    In a real scenario, you would load actual therapy session transcripts
    or patient journal entries labeled by mental health professionals.
    
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    # 10 types of cognitive distortions
    distortion_examples = {
        0: [  # All-or-Nothing Thinking
            "I always fail at everything I try",
            "Nobody ever likes me",
            "I'm a complete failure",
            "Everything I do is wrong",
        ],
        1: [  # Overgeneralization
            "This always happens to me",
            "I never get anything right",
            "Everyone thinks I'm stupid",
            "Nothing ever works out for me",
        ],
        2: [  # Mental Filter
            "That one mistake ruined everything",
            "I only remember the bad parts",
            "The negative comment is all that matters",
            "I can't stop thinking about that failure",
        ],
        3: [  # Disqualifying the Positive
            "That success doesn't count",
            "They're just being nice",
            "It was just luck, not skill",
            "Anyone could have done that",
        ],
        4: [  # Jumping to Conclusions
            "They must think I'm incompetent",
            "This will definitely go wrong",
            "I know they don't like me",
            "It's obvious this won't work",
        ],
        5: [  # Magnification/Minimization
            "This is a complete disaster",
            "My achievement is nothing special",
            "This mistake is catastrophic",
            "That success was trivial",
        ],
        6: [  # Emotional Reasoning
            "I feel stupid, so I must be stupid",
            "I feel anxious, so something bad will happen",
            "I feel like a failure, so I am one",
            "I feel guilty, so I must have done something wrong",
        ],
        7: [  # Should Statements
            "I should be perfect at this",
            "I must never make mistakes",
            "I ought to be better than this",
            "I have to succeed at everything",
        ],
        8: [  # Labeling
            "I'm a loser",
            "I'm worthless",
            "I'm incompetent",
            "I'm a bad person",
        ],
        9: [  # Personalization
            "It's all my fault",
            "I'm responsible for their unhappiness",
            "I caused this problem",
            "Everything bad is because of me",
        ]
    }
    
    import random
    random.seed(42)
    
    # Generate training data
    train_texts = []
    train_labels = []
    
    for _ in range(num_train):
        label = random.randint(0, 9)
        text = random.choice(distortion_examples[label])
        # Add some variation
        text = preprocess_cbt_text(text)
        train_texts.append(text)
        train_labels.append(label)
    
    # Generate test data
    test_texts = []
    test_labels = []
    
    for _ in range(num_test):
        label = random.randint(0, 9)
        text = random.choice(distortion_examples[label])
        text = preprocess_cbt_text(text)
        test_texts.append(text)
        test_labels.append(label)
    
    return train_texts, train_labels, test_texts, test_labels


def demonstrate_cbt_transformer_hqde():
    """Demonstrate HQDE with transformer for CBT classification."""
    logger.info("=" * 70)
    logger.info("HQDE Framework - CBT Transformer Classification Demo")
    logger.info("=" * 70)
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(monitoring_interval=1.0)
    monitor.start_monitoring()
    
    try:
        # Step 1: Create CBT dataset
        logger.info("\n[Step 1] Creating CBT cognitive distortion dataset...")
        train_texts, train_labels, test_texts, test_labels = \
            create_cbt_cognitive_distortion_data(num_train=1000, num_test=200)
        
        logger.info(f"  Training samples: {len(train_texts)}")
        logger.info(f"  Test samples: {len(test_texts)}")
        logger.info(f"  Number of classes: 10 (cognitive distortions)")
        
        # Step 2: Build tokenizer and vocabulary
        logger.info("\n[Step 2] Building tokenizer and vocabulary...")
        tokenizer = SimpleTokenizer(
            vocab_size=5000,
            max_seq_length=128,
            lowercase=True
        )
        
        # Build vocabulary from training texts
        tokenizer.build_vocab(train_texts, min_freq=1)
        
        # Step 3: Create datasets
        logger.info("\n[Step 3] Creating PyTorch datasets...")
        train_dataset = CBTDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=tokenizer,
            max_seq_length=128
        )
        
        test_dataset = CBTDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=tokenizer,
            max_seq_length=128
        )
        
        # Step 4: Create data loaders
        logger.info("\n[Step 4] Creating data loaders...")
        train_loader = TextDataLoader.create_text_loader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        test_loader = TextDataLoader.create_text_loader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        
        # Step 5: Configure HQDE system
        logger.info("\n[Step 5] Configuring HQDE system...")
        
        # Model configuration
        model_kwargs = {
            'vocab_size': len(tokenizer.word2idx),
            'num_classes': 10,
            'd_model': 128,
            'nhead': 4,
            'num_encoder_layers': 3,
            'dim_feedforward': 512,
            'dropout_rate': 0.15,
            'max_seq_length': 128,
            'use_domain_adaptation': True
        }
        
        # Training configuration (CBT-optimized)
        training_config = make_cbt_training_config(
            ensemble_mode='independent',
            learning_rate=3e-4,
            warmup_epochs=2,
            label_smoothing=0.05,
            dropout_rate=0.15
        )
        
        # Quantization configuration (for FedAvg mode)
        quantization_config = {
            'base_bits': 12,
            'min_bits': 8,
            'max_bits': 16,
            'block_size': 1024
        }
        
        # Aggregation configuration
        aggregation_config = {
            'noise_scale': 0.005,
            'exploration_factor': 0.1
        }
        
        logger.info(f"  Model: CBTTransformerClassifier")
        logger.info(f"  Ensemble mode: {training_config['ensemble_mode']}")
        logger.info(f"  Number of workers: 4")
        logger.info(f"  Learning rate: {training_config['learning_rate']}")
        
        # Step 6: Create HQDE system
        logger.info("\n[Step 6] Creating HQDE ensemble system...")
        hqde_system = create_hqde_system(
            model_class=CBTTransformerClassifier,
            model_kwargs=model_kwargs,
            num_workers=4,
            training_config=training_config,
            quantization_config=quantization_config,
            aggregation_config=aggregation_config
        )
        
        logger.info("  ✓ HQDE system created successfully")
        
        # Step 7: Train the ensemble
        logger.info("\n[Step 7] Training HQDE ensemble...")
        logger.info("  This may take a few minutes...")
        
        start_time = time.time()
        
        training_metrics = hqde_system.train(
            data_loader=train_loader,
            num_epochs=10,
            validation_loader=test_loader
        )
        
        training_time = time.time() - start_time
        
        logger.info(f"\n  ✓ Training completed in {training_time:.2f} seconds")
        
        # Step 8: Evaluate the ensemble
        logger.info("\n[Step 8] Evaluating ensemble on test set...")
        
        eval_metrics = hqde_system.evaluate(test_loader)
        
        logger.info(f"  Test Loss: {eval_metrics.get('loss', 0):.4f}")
        logger.info(f"  Test Accuracy: {eval_metrics.get('accuracy', 0):.2%}")
        
        # Step 9: Make predictions
        logger.info("\n[Step 9] Making predictions...")
        
        predictions = hqde_system.predict(test_loader)
        logger.info(f"  Predictions shape: {predictions.shape}")
        logger.info(f"  Predicted classes: {predictions.argmax(dim=1)[:10].tolist()}")
        
        # Step 10: Get performance metrics
        logger.info("\n[Step 10] Performance metrics...")
        
        performance_metrics = hqde_system.get_performance_metrics()
        
        logger.info("  System Performance:")
        for metric, value in performance_metrics.items():
            if isinstance(value, float):
                logger.info(f"    {metric}: {value:.4f}")
            else:
                logger.info(f"    {metric}: {value}")
        
        # Step 11: Save model
        logger.info("\n[Step 11] Saving model...")
        model_path = "cbt_transformer_hqde_model.pth"
        hqde_system.save_model(model_path)
        logger.info(f"  ✓ Model saved to {model_path}")
        
        # Step 12: Demonstrate inference on new text
        logger.info("\n[Step 12] Testing inference on new examples...")
        
        test_examples = [
            "I always mess everything up",
            "This one mistake ruined my entire day",
            "I feel like a failure so I must be one"
        ]
        
        for i, text in enumerate(test_examples, 1):
            # Encode text
            encoded = tokenizer.encode(text, padding=True, truncation=True)
            input_ids = encoded['input_ids'].unsqueeze(0)  # Add batch dimension
            attention_mask = encoded['attention_mask'].unsqueeze(0)
            
            # Create a mini dataset and loader
            mini_dataset = CBTDataset(
                texts=[text],
                labels=[0],  # Dummy label
                tokenizer=tokenizer
            )
            mini_loader = TextDataLoader.create_text_loader(
                mini_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            
            # Predict
            pred = hqde_system.predict(mini_loader)
            predicted_class = pred.argmax(dim=1).item()
            
            distortion_names = [
                "All-or-Nothing", "Overgeneralization", "Mental Filter",
                "Disqualifying Positive", "Jumping to Conclusions",
                "Magnification", "Emotional Reasoning", "Should Statements",
                "Labeling", "Personalization"
            ]
            
            logger.info(f"\n  Example {i}: '{text}'")
            logger.info(f"    Predicted: {distortion_names[predicted_class]}")
        
        # Record completion
        monitor.record_event(
            'training_complete',
            'CBT transformer training completed',
            {'accuracy': eval_metrics.get('accuracy', 0)}
        )
        
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        monitor.record_event('training_error', f'Training failed: {str(e)}')
        raise
    
    finally:
        # Cleanup
        logger.info("\n[Cleanup] Releasing resources...")
        hqde_system.cleanup()
        
        # Stop monitoring
        monitor.stop_monitoring()
        performance_report = monitor.get_performance_report()
        
        logger.info("\n" + "=" * 70)
        logger.info("Performance Summary")
        logger.info("=" * 70)
        
        current_metrics = performance_report.get('current_metrics', {})
        logger.info(f"CPU Usage: {current_metrics.get('cpu_percent', 0):.1f}%")
        logger.info(f"Memory Usage: {current_metrics.get('memory_percent', 0):.1f}%")
        logger.info(f"GPU Memory: {current_metrics.get('gpu_memory_used_gb', 0):.2f} GB")
        
        # Export metrics
        monitor.export_metrics("cbt_transformer_performance.json", format="json")
        logger.info("\n✓ Performance metrics exported to cbt_transformer_performance.json")
        
        logger.info("\n" + "=" * 70)
        logger.info("Demo completed successfully!")
        logger.info("=" * 70)


def demonstrate_lightweight_transformer():
    """Quick demo with lightweight transformer."""
    logger.info("\n" + "=" * 70)
    logger.info("Lightweight Transformer Demo (Fast Training)")
    logger.info("=" * 70)
    
    # Create small dataset
    train_texts, train_labels = create_cbt_sample_data(num_samples=500)
    test_texts, test_labels = create_cbt_sample_data(num_samples=100)
    
    # Simple tokenizer
    tokenizer = SimpleTokenizer(vocab_size=3000, max_seq_length=64)
    tokenizer.build_vocab(train_texts, min_freq=1)
    
    # Datasets
    train_dataset = CBTDataset(train_texts, train_labels, tokenizer, max_seq_length=64)
    test_dataset = CBTDataset(test_texts, test_labels, tokenizer, max_seq_length=64)
    
    # Loaders
    train_loader = TextDataLoader.create_text_loader(train_dataset, batch_size=32, num_workers=0)
    test_loader = TextDataLoader.create_text_loader(test_dataset, batch_size=32, num_workers=0)
    
    # Lightweight model
    model_kwargs = {
        'vocab_size': len(tokenizer.word2idx),
        'num_classes': 10,
        'd_model': 64,
        'nhead': 2,
        'num_encoder_layers': 2,
        'dim_feedforward': 256,
        'dropout_rate': 0.1,
        'max_seq_length': 64
    }
    
    # Fast training config
    training_config = make_transformer_training_config(
        ensemble_mode='independent',
        learning_rate=1e-3,
        warmup_epochs=1
    )
    
    # Create system
    hqde_system = create_hqde_system(
        model_class=LightweightTransformerClassifier,
        model_kwargs=model_kwargs,
        num_workers=2,
        training_config=training_config
    )
    
    # Quick training
    logger.info("Training lightweight ensemble (5 epochs)...")
    start_time = time.time()
    hqde_system.train(train_loader, num_epochs=5, validation_loader=test_loader)
    training_time = time.time() - start_time
    
    # Evaluate
    eval_metrics = hqde_system.evaluate(test_loader)
    
    logger.info(f"\n✓ Training completed in {training_time:.2f} seconds")
    logger.info(f"  Test Accuracy: {eval_metrics.get('accuracy', 0):.2%}")
    
    # Cleanup
    hqde_system.cleanup()
    
    logger.info("✓ Lightweight demo completed!")


def main():
    """Main demo function."""
    try:
        # Run main CBT transformer demo
        demonstrate_cbt_transformer_hqde()
        
        # Optional: Run lightweight demo
        # demonstrate_lightweight_transformer()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
