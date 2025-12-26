# Rock-Paper-Scissors: Deep Learning Architecture Comparison

A comprehensive comparison of 5 neural network architectures for hand gesture classification, trained and evaluated on Apple Silicon (M4 Pro).

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architectures Evaluated](#architectures-evaluated)
- [Installation](#installation)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Hardware & Performance](#hardware--performance)
- [Authors](#authors)

---

## 🎯 Project Overview

This project evaluates **5 different neural network architectures** for classifying hand gestures (Rock, Paper, Scissors) from images. The goal is to provide an empirical comparison of:

1. **Custom-designed networks** (ANN, CNN)
2. **Transfer learning** approaches (ResNet18)
3. **Efficient architectures** (SqueezeNet, DenseNet)

Each model is trained from scratch (or fine-tuned) using aggressive data augmentation and optimized hyperparameters found through grid search.

---

## 📂 Dataset

- **Source**: Rock-Paper-Scissors image dataset
- **Classes**: 3 (Rock, Paper, Scissors)
- **Split**:
  - Training: ~2,520 images
  - Validation: ~372 images
  - Test: 63 images
- **Image Size**: 300×300 RGB
- **Challenges**: White backgrounds, lighting variations, hand positioning diversity

### Data Augmentation
- **Geometric**: Horizontal flip, rotation (45°), affine transforms, perspective distortion
- **Color**: Jitter (brightness, contrast, saturation), random grayscale
- **Noise**: Random erasing, white background replacement
- **Normalization**: ImageNet stats for transfer learning, [0.5, 0.5, 0.5] for custom models

---

## 🏗️ Architectures Evaluated

### 1. Simple Artificial Neural Network (ANN)
- **Type**: Fully-connected multilayer perceptron
- **Layers**: Input (270K neurons) → FC(512) → FC(256) → Output(3)
- **Regularization**: Dropout (0.2), Batch Normalization
- **Parameters**: ~138M
- **Optimizer**: Adam (lr=1e-3)

### 2. Custom CNN
- **Type**: Convolutional Neural Network (from scratch)
- **Architecture**: 
  - 3 Conv blocks (16/32/64 filters)
  - Each block: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool → Dropout
  - Global Average Pooling
  - FC classifier
- **Parameters**: ~288K
- **Optimizer**: Adam (lr=1e-3)

### 3. Transfer Learning - ResNet18
- **Type**: Fine-tuned pretrained model
- **Modes Tested**:
  - **Head-Only**: Only final classifier trained
  - **Fine-Tune Layer4**: Last residual block + classifier
  - **Deep Fine-Tune**: Layers 3-4 + classifier
- **Parameters**: ~11.2M (10.5M trainable)
- **Optimizer**: Adam (lr=1e-4)
- **Best**: Deep Fine-Tune achieved **100% validation accuracy**

### 4. SqueezeNet (From Scratch)
- **Type**: Efficient CNN using Fire modules
- **Architecture**: 
  - Conv1 → Fire modules (squeeze-expand pattern)
  - 3 MaxPool stages
  - Conv classifier + Global Average Pooling
- **Parameters**: ~724K
- **Optimizer**: AdamW (lr=1e-4 to 5e-4)

### 5. DenseNet (Custom)
- **Type**: Densely connected convolutional network
- **Architecture**: 
  - Dense blocks with growth rate = 12
  - Transition layers (compression)
  - Block config: (4, 8, 16 layers)
- **Parameters**: ~450K (estimated)
- **Optimizer**: AdamW (lr=3e-4)

---

## 🚀 Installation

### Requirements
```bash
# Python 3.10+
pip install torch torchvision matplotlib seaborn numpy pandas scikit-learn psutil
```

### Hardware Requirements
- **GPU**: Apple Silicon (MPS backend) or CUDA-compatible GPU
- **RAM**: 16GB+ recommended
- **Storage**: ~2GB for dataset + checkpoints

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/rps-cnn-comparison.git
cd rps-cnn-comparison

# Download dataset (example)
# Place in ./Dataset/ with structure:
#   Dataset/
#     train/rock/, train/paper/, train/scissors/
#     valid/rock/, valid/paper/, valid/scissors/
#     test/rock/, test/paper/, test/scissors/

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Quick Start (Run All Experiments)
```python
# Import main module
from Project import *

# 1. Run ANN search
run_ann_search()

# 2. Run Custom CNN search
run_cnn_search()

# 3. Run ResNet experiments
run_resnet_experiments()

# 4. Run SqueezeNet search
run_squeezenet_FIXED()

# 5. Compare all results
display_all_results()
```

### Train Individual Model
```python
# Example: Train ResNet18 with specific config
train_dl, valid_dl, test_dl, classes = get_dataloaders(
    TRAIN_PATH, VALID_PATH, TEST_PATH,
    train_transform_imagenet_v2, valid_transform_imagenet, 
    batch_size=32
)

model = TransferLearningCNN(dropout_rate=0.6, fine_tune_mode='fine_tune_deep').to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

history, state, time = train_val_loop(
    model, train_dl, valid_dl, criterion, optimizer,
    epochs=20, patience=5, name="ResNet_Custom"
)
```

### Evaluate with Test-Time Augmentation
```python
results = evaluate_with_tta(model, test_dl, use_tta=True)
print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"F1-Score: {results['f1_macro']:.4f}")
```

---

## 📊 Results Summary

| Model | Val Acc | Test Acc | F1-Macro | Inference (ms) | Training Time | Parameters |
|-------|---------|----------|----------|----------------|---------------|------------|
| **ANN Custom** | 86.6% | 55.6% | 0.542 | 180.2 | 4.7 min | 138M |
| **CNN Custom** | 94.4% | 53.9% | 0.489 | 187.2 | 18.7 min | 288K |
| **ResNet18 (Deep)** | **100%** | **84.1%** | **0.834** | **8.9** | 2.9 min | 11.2M |
| **SqueezeNet** | ~97% | ~73% | ~0.73 | ~12 | 15.4 min | 724K |
| **DenseNet** | ~95% | ~73% | ~0.72 | ~15 | 21.6 min | 450K |

*Note: Test accuracy reflects model generalization with TTA enabled for CNN-based models.*

---

## 🔍 Key Findings

### 1. Transfer Learning Dominance
- **ResNet18 achieved near-perfect performance** (100% val, 84% test)
- Fine-tuning deeper layers (Layer3+4) significantly outperformed head-only training
- Transfer learning required **16× fewer epochs** than training from scratch

### 2. Overfitting in Custom Models
- ANN: Massive parameter count (138M) led to severe overfitting (86% val → 56% test)
- Custom CNN: Better than ANN but still struggled with generalization gap

### 3. Efficiency vs. Accuracy Trade-off
- **SqueezeNet**: Best efficiency (724K params, ~12ms inference)
- **ResNet18**: Best accuracy-speed balance (8.9ms, 84% test)
- **ANN**: Worst trade-off (slowest inference, lowest accuracy)

### 4. Impact of Data Augmentation
- **Critical transformation**: Replacing white backgrounds with random colors
- Models trained with aggressive augmentation showed 10-15% improvement
- Test-Time Augmentation (TTA) provided 2-5% boost in CNNs

### 5. Hardware Observations (Apple Silicon MPS)
- Training speedup: ~3-4× vs CPU
- Memory management crucial (unified RAM architecture)
- Batch size limitations: Max 128 for SqueezeNet on 16GB RAM

---

## 📁 Project Structure

```
rps-cnn-comparison/
│
├── Dataset/                    # Dataset (not included in repo)
│   ├── train/
│   ├── valid/
│   └── test/
│
├── checkpoints/                # Saved model weights
│   ├── 1_ANN_Custom_best.pt
│   ├── 2_CNN_Custom_best.pt
│   ├── 3_Transfer_ResNet_best.pt
│   ├── 4_SqueezeNet_best.pt
│   └── 5_DenseNet_best.pt
│
├── Project.ipynb              # Main notebook (corrected version)
├── old.md                     # Original experiments (reference)
├── Project.md                 # Exported markdown
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## ⚙️ Hardware & Performance

### Development Environment
- **Device**: MacBook Pro M4 Pro
- **RAM**: 48GB Unified Memory
- **GPU**: Apple M4 Pro (14-core)
- **Backend**: PyTorch MPS (Metal Performance Shaders)

### Performance Notes
- **Batch Size**: Limited by unified memory (32-128 typical)
- **Training Speed**: ~30-40 images/sec for ResNet18
- **Memory Usage**: Peak ~12GB during SqueezeNet training
- **Early Stopping**: Essential for preventing overfitting (patience=5-10)

---

## 🛠️ Troubleshooting

### Common Issues

**1. MPS Out of Memory**
```python
# Reduce batch size
batch_size = 32  # Instead of 128

# Or clear cache manually
torch.mps.empty_cache()
clear_memory()
```

**2. Model Stuck at 33% Accuracy (Random Guessing)**
- **Cause**: Missing `Fire` module definition for SqueezeNet
- **Fix**: Ensure all custom modules are defined before model instantiation
- **Verify**: Run `quick_sanity_check()` to test if model can overfit 1 batch

**3. ColorJitter Not Working**
- **Cause**: Applied after `ToTensor()` (operates on PIL Images only)
- **Fix**: Move `ColorJitter` before `ToTensor()` in transform pipeline

**4. Validation Accuracy Drops Suddenly**
- **Cause**: Learning rate too high or insufficient regularization
- **Fix**: Reduce LR by 10×, increase dropout rate, or add weight decay

---

## 📚 References

- **SqueezeNet**: [Iandola et al., 2016](https://arxiv.org/abs/1602.07360)
- **ResNet**: [He et al., 2015](https://arxiv.org/abs/1512.03385)
- **DenseNet**: [Huang et al., 2017](https://arxiv.org/abs/1608.06993)
- **PyTorch MPS Backend**: [Apple Developer Documentation](https://developer.apple.com/metal/pytorch/)

---

## 👥 Authors

- **Oriol** - Custom architectures, transfer learning experiments
- **Mei** - Data augmentation, hyperparameter tuning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [Laurence Moroney's RPS Dataset](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)
- Pretrained weights: PyTorch Model Zoo (ImageNet1K)
- Hardware support: Apple Silicon MPS backend

---

## 📧 Contact

For questions or collaboration:
- **Email**: [your.email@example.com]
- **GitHub Issues**: [Open an issue](https://github.com/yourusername/rps-cnn-comparison/issues)

---

**⭐ If you found this project useful, please consider giving it a star!**
