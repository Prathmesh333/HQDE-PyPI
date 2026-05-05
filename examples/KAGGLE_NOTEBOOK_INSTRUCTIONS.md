# CBT DeBERTa HQDE Kaggle Notebook Instructions

## 📋 Overview

This notebook trains an ensemble of DeBERTa models using the HQDE (Hierarchical Quantum-inspired Distributed Ensemble) framework to classify cognitive distortions in therapy conversations.

## 🎯 Quick Start for Kaggle

### Step 1: Upload to Kaggle
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Click "File" → "Upload Notebook"
4. Upload `cbt_deberta_hqde_kaggle.ipynb`

### Step 2: Configure Hardware
1. Click "Settings" (right sidebar)
2. Under "Accelerator", select **GPU T4 x2**
3. Ensure you have **2 GPUs** enabled
4. Click "Save"

### Step 3: Run the Notebook
1. Click "Run All" or run cells sequentially
2. Wait for training to complete (~30-45 minutes)
3. Check results in the final cell

## 🔧 Hardware Requirements

- **GPUs**: 2x T4 (Kaggle free tier)
- **vCPUs**: 4
- **RAM**: ~30GB
- **Disk**: ~5GB

## 📊 Expected Results

- **Training Time**: 30-45 minutes
- **Expected Accuracy**: 85-90%
- **Expected F1-Score**: 85-90%

## 🧠 Cognitive Distortion Classes (10)

0. **All-or-Nothing Thinking** - Seeing things in black and white categories
1. **Overgeneralization** - Drawing broad conclusions from single events
2. **Mental Filter** - Focusing exclusively on negative details
3. **Disqualifying the Positive** - Rejecting positive experiences
4. **Jumping to Conclusions** - Making negative interpretations without evidence
5. **Magnification/Catastrophizing** - Exaggerating importance of problems
6. **Emotional Reasoning** - Assuming emotions reflect reality
7. **Should Statements** - Using "should" and "must" statements
8. **Labeling** - Attaching negative labels to self or others
9. **Personalization** - Taking responsibility for events outside control

## 🏗️ Architecture Details

### HQDE Ensemble Configuration
- **4 ensemble workers** (distributed across 2 GPUs)
- **Diverse configurations** per worker:
  - Different dropout rates: [0.1, 0.15, 0.2, 0.25]
  - Different learning rates: [1.5e-5, 2e-5, 2.5e-5, 3e-5]
- **Ensemble prediction**: Average of all worker predictions

### Model Configuration
- **Base Model**: microsoft/deberta-v3-base
- **Max Sequence Length**: 256 tokens
- **Batch Size**: 8 per GPU (effective batch size = 16)
- **Epochs**: 15
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine with warmup
- **Mixed Precision**: Enabled (AMP)

### Training Features
- ✅ Gradient clipping (max_norm=1.0)
- ✅ Mixed precision training (FP16)
- ✅ Cosine learning rate schedule with warmup
- ✅ Stratified train/val/test splits
- ✅ Ensemble diversity through hyperparameter variation

## 📁 Notebook Structure

### Cell 1: Title and Overview
Markdown cell with project description

### Cell 2: Install Packages
Installs required dependencies:
- transformers
- datasets
- accelerate
- scikit-learn
- pandas, matplotlib, seaborn

### Cell 3: Imports
Import all required libraries and check GPU availability

### Cell 4: Configuration
Set up all hyperparameters and configuration

### Cell 5: Dataset Generation
Generate 100 synthetic CBT therapy conversations (10 per class)

### Cell 6: Data Preparation
- Split data (train/val/test)
- Load tokenizer
- Create PyTorch datasets and dataloaders

### Cell 7: Model Definition
Define DeBERTa classifier architecture

### Cell 8: Ensemble Worker
Define ensemble worker class with training/evaluation methods

### Cell 9: Create Workers
Instantiate 4 ensemble workers with diverse configurations

### Cell 10: Training Loop
Train all workers for 15 epochs with ensemble evaluation

### Cell 11: Final Evaluation
- Test set evaluation
- Classification report
- Confusion matrix visualization
- Per-class accuracy
- Individual worker performance

## 🎨 Output Files

The notebook creates the following outputs:

- `./cbt_output/confusion_matrix.png` - Confusion matrix visualization

## 🔍 Monitoring Training

During training, you'll see:
- Progress bars for each worker
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Ensemble accuracy and F1-score per epoch
- Best model checkpoints

## 🐛 Troubleshooting

### Out of Memory (OOM)
If you encounter OOM errors:
1. Reduce `batch_size` from 8 to 4 or 6
2. Reduce `max_length` from 256 to 128
3. Reduce `num_workers` from 4 to 2

### Slow Training
If training is too slow:
1. Ensure 2 GPUs are enabled in Kaggle settings
2. Check GPU utilization in Kaggle's resource monitor
3. Reduce `num_epochs` from 15 to 10

### Low Accuracy
If accuracy is below expected:
1. Increase `num_epochs` to 20
2. Increase `max_length` to 512 (may require smaller batch size)
3. Try different learning rates

## 📚 Dataset Information

The dataset consists of 100 synthetic therapy conversations:
- **10 conversations per cognitive distortion class**
- **Balanced dataset** (10 samples per class)
- **Realistic therapy dialogue** with patient statements and emotions
- **Reasoning chains** showing how classification is determined

### Data Splits
- **Training**: 70 samples (70%)
- **Validation**: 10 samples (10%)
- **Test**: 20 samples (20%)

## 🚀 Next Steps

After running the notebook:

1. **Analyze Results**
   - Check confusion matrix for misclassifications
   - Review per-class accuracy
   - Compare individual worker vs ensemble performance

2. **Experiment with Hyperparameters**
   - Try different learning rates
   - Adjust dropout rates
   - Modify ensemble size

3. **Use Real Data**
   - Replace synthetic dataset with real therapy conversations
   - Adjust preprocessing for your data format
   - Retrain with larger dataset

4. **Deploy Model**
   - Save best worker models
   - Create inference pipeline
   - Build API or web interface

## 📖 References

- **DeBERTa**: [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
- **HQDE Framework**: Hierarchical Quantum-inspired Distributed Ensemble
- **CBT**: Cognitive Behavioral Therapy cognitive distortions

## 💡 Tips for Best Results

1. **Use the full 2x T4 GPU allocation** - This is optimized for 2 GPUs
2. **Let it train for all 15 epochs** - Early stopping may reduce accuracy
3. **Monitor ensemble accuracy** - Should improve over individual workers
4. **Check confusion matrix** - Identifies which classes are confused
5. **Save your work frequently** - Kaggle sessions can timeout

## ⚠️ Important Notes

- This notebook uses **synthetic data** for demonstration
- For production use, train on **real therapy conversation data**
- Ensure proper **data privacy and ethics** when using real patient data
- This is a **research/educational tool**, not for clinical diagnosis

## 📞 Support

If you encounter issues:
1. Check Kaggle GPU availability
2. Verify all cells run in order
3. Review error messages carefully
4. Adjust hyperparameters if needed

---

**Happy Training! 🎉**
