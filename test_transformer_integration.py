"""
Quick integration test for HQDE transformer extension.

This script verifies that all transformer components are properly integrated.
"""

import sys
import torch

def test_imports():
    """Test that all transformer components can be imported."""
    print("Testing imports...")

    try:
        from hqde import (
            TransformerTextClassifier,
            LightweightTransformerClassifier,
            CBTTransformerClassifier,
            SimpleTokenizer,
            TextClassificationDataset,
            CBTDataset,
            TextDataLoader,
            make_transformer_training_config,
            make_cbt_training_config
        )
        print("âœ“ All imports successful")
        return True
    except ImportError as e:
        print(f"âœ— Import failed: {e}")
        return False


def test_tokenizer():
    """Test tokenizer functionality."""
    print("\nTesting tokenizer...")

    try:
        from hqde.utils.text_data_utils import SimpleTokenizer

        tokenizer = SimpleTokenizer(vocab_size=1000, max_seq_length=64)
        texts = ["Hello world", "This is a test", "Tokenizer works"]
        tokenizer.build_vocab(texts, min_freq=1)

        # Test encoding
        encoded = tokenizer.encode("Hello world", padding=True, truncation=True)
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert encoded['input_ids'].shape[0] == 64

        # Test decoding
        decoded = tokenizer.decode(encoded['input_ids'])

        print(f"âœ“ Tokenizer works (vocab size: {len(tokenizer.word2idx)})")
        return True
    except Exception as e:
        print(f"âœ— Tokenizer test failed: {e}")
        return False


def test_models():
    """Test transformer model instantiation."""
    print("\nTesting models...")

    try:
        from hqde.models.transformers import (
            TransformerTextClassifier,
            LightweightTransformerClassifier,
            CBTTransformerClassifier
        )

        # Test standard transformer
        model1 = TransformerTextClassifier(
            vocab_size=1000,
            num_classes=2,
            d_model=64,
            nhead=2,
            num_encoder_layers=2
        )

        # Test lightweight transformer
        model2 = LightweightTransformerClassifier(
            vocab_size=1000,
            num_classes=2,
            d_model=32,
            nhead=2
        )

        # Test CBT transformer
        model3 = CBTTransformerClassifier(
            vocab_size=1000,
            num_classes=10,
            d_model=64,
            nhead=2,
            num_encoder_layers=2
        )

        # Test forward pass
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output1 = model1(input_ids, attention_mask)
        output2 = model2(input_ids, attention_mask)
        output3 = model3(input_ids, attention_mask)

        assert output1.shape == (batch_size, 2)
        assert output2.shape == (batch_size, 2)
        assert output3.shape == (batch_size, 10)

        print("âœ“ All models work correctly")
        return True
    except Exception as e:
        print(f"âœ— Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset creation."""
    print("\nTesting datasets...")

    try:
        from hqde.utils.text_data_utils import (
            SimpleTokenizer,
            TextClassificationDataset,
            CBTDataset
        )

        # Create tokenizer
        tokenizer = SimpleTokenizer(vocab_size=1000, max_seq_length=32)
        texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
        tokenizer.build_vocab(texts, min_freq=1)

        # Test TextClassificationDataset
        labels = [0, 1, 0]
        dataset1 = TextClassificationDataset(texts, labels, tokenizer)
        assert len(dataset1) == 3
        sample1 = dataset1[0]
        assert 'input_ids' in sample1
        assert 'attention_mask' in sample1
        assert 'labels' in sample1

        # Test CBTDataset
        dataset2 = CBTDataset(texts, labels, tokenizer)
        assert len(dataset2) == 3
        sample2 = dataset2[0]
        assert 'input_ids' in sample2

        print("âœ“ Datasets work correctly")
        return True
    except Exception as e:
        print(f"âœ— Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_configs():
    """Test training configuration presets."""
    print("\nTesting training configs...")

    try:
        from hqde.utils.transformer_presets import (
            make_transformer_training_config,
            make_cbt_training_config,
            make_lightweight_transformer_config,
            make_large_transformer_config
        )

        config1 = make_transformer_training_config()
        config2 = make_cbt_training_config()
        config3 = make_lightweight_transformer_config()
        config4 = make_large_transformer_config()

        # Verify required keys
        required_keys = ['ensemble_mode', 'learning_rate', 'optimizer']
        for config in [config1, config2, config3, config4]:
            for key in required_keys:
                assert key in config, f"Missing key: {key}"

        print("âœ“ Training configs work correctly")
        return True
    except Exception as e:
        print(f"âœ— Training config test failed: {e}")
        return False


def test_hqde_integration():
    """Test HQDE system with transformer."""
    print("\nTesting HQDE integration...")

    try:
        from hqde import create_hqde_system
        from hqde.models.transformers import LightweightTransformerClassifier
        from hqde.utils.text_data_utils import (
            SimpleTokenizer,
            TextClassificationDataset,
            TextDataLoader
        )
        from hqde.utils.transformer_presets import make_transformer_training_config

        # Create small dataset
        texts = ["text " + str(i) for i in range(20)]
        labels = [i % 2 for i in range(20)]

        tokenizer = SimpleTokenizer(vocab_size=100, max_seq_length=16)
        tokenizer.build_vocab(texts, min_freq=1)

        dataset = TextClassificationDataset(texts, labels, tokenizer)
        loader = TextDataLoader.create_text_loader(
            dataset, batch_size=4, shuffle=False, num_workers=0
        )

        # Create HQDE system
        model_kwargs = {
            'vocab_size': len(tokenizer.word2idx),
            'num_classes': 2,
            'd_model': 32,
            'nhead': 2,
            'num_encoder_layers': 1,
            'dropout_rate': 0.1
        }

        training_config = make_transformer_training_config(
            ensemble_mode='independent',
            learning_rate=1e-3
        )

        hqde_system = create_hqde_system(
            model_class=LightweightTransformerClassifier,
            model_kwargs=model_kwargs,
            num_workers=2,
            training_config=training_config
        )

        # Quick training (1 epoch)
        print("  Running quick training test (1 epoch)...")
        hqde_system.train(loader, num_epochs=1)

        # Test prediction
        predictions = hqde_system.predict(loader)
        assert predictions.shape[0] == len(texts)
        assert predictions.shape[1] == 2

        # Cleanup
        hqde_system.cleanup()

        print("âœ“ HQDE integration works correctly")
        return True
    except Exception as e:
        print(f"âœ— HQDE integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("HQDE Transformer Extension - Integration Tests")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Tokenizer", test_tokenizer),
        ("Models", test_models),
        ("Datasets", test_dataset),
        ("Training Configs", test_training_configs),
        ("HQDE Integration", test_hqde_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"âœ— {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "âœ“ PASS" if result else "âœ— FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nðŸŽ‰ All tests passed! Transformer extension is ready to use.")
        return 0
    else:
        print(f"\nâš ï¸  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
