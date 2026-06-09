# ISIC Skin Lesion Classification (Few-Shot Learning Approach)

## Project Overview

This project is a deep learning-based **multi-class skin lesion classification system** that employs **Few-Shot Learning** and **MobileViT** architecture to learn skin lesion classification features from limited annotated data. The project supports multiple skin lesion datasets, including **ISIC-2018**, **HAM10000**, **SD-198**, and **MedMNIST**.

## Datasets Overview

### 1. ISIC-2018 Dataset
- **Scale**: ~10,015 training images + 1,512 test images
- **Number of Classes**: 7 skin lesion categories
  - MEL (Melanoma)
  - NV (Nevus)
  - BCC (Basal Cell Carcinoma)
  - AKIEC (Actinic Keratosis/Bowen's Disease)
  - BKL (Benign Keratosis)
  - DF (Dermatofibroma)
  - VASC (Vascular Lesion)
- **Storage Location**: `ISIC-2018-dataset/`
- **File Format**: JPEG images + CSV annotation files (one-hot encoded)

### 2. HAM10000 Dataset
- **Scale**: 10,015 dermatoscopic images from multiple sources
- **Source**: Aggregated from multiple skin lesion databases
- **Storage Location**: `HAM10000/`
- **Format**: Images + metadata CSV

### 3. SD-198 Dataset
- **Scale**: Skin disease sample collection
- **Storage Location**: `SD-198/`

### 4. MedMNIST Dataset
- **Type**: Medical imaging dataset
- **Purpose**: Model pre-training and transfer learning

## Detailed Dataset Information

### Data Organization Structure
```
ISIC-2018-dataset/
├── images/
│   └── ISIC2018_Task3_Training_Input/  # Training images
├── ISIC2018_Task3_Training_GroundTruth/  # Annotation CSV
├── splits/                               # Data split files
│   ├── train.txt / train.csv
│   ├── val.txt / val.csv
│   └── test.txt / test.csv
└── output/                               # Model outputs
```

### Data Preprocessing
- **Image Size**: Uniformly resized to 384×384 pixels
- **Normalization**: ImageNet standard normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Data Augmentation**:
  - Random rotation (±10°)
  - Random flipping (horizontal/vertical)
  - Random cropping
  - Color jittering
  - Cutmix/Mixup augmentation

### Data Splitting Strategy
The project provides pre-defined data splits (in `ISIC_Splits/` directory):
- **Training Set**: For basic model training (classes 0-4)
- **Validation Set**: For model validation and hyperparameter tuning
- **Test Set**: For Few-Shot evaluation on new classes (classes 5-6)

## Code Information

### Core Code Modules

#### 1. Model Architecture (`mobilevit.py`)
- **Model**: MobileViT (Mobile Vision Transformer)
- **Supported Versions**: XS, S (lightweight models)
- **Features**:
  - Combines efficiency of MobileNet with expressiveness of Transformer
  - Low parameter count, suitable for edge device deployment
  - Pre-trained weight loading support

#### 2. Few-Shot Learning Model (`fsl_model.py`)
Implements ProtoNet (Prototypical Network) and Dynamic Prototype Calibration Module (DPCM):

**DPCM (Dynamic Prototype Calibration Module)**
```python
class DPCM(nn.Module):
    """Dynamically adjusts contribution weights of support set samples to prototype"""
    - Input: Support set features
    - Output: Dynamically calibrated prototype vector
```

**MobileViT-ProtoNet**
```python
class MobileViT_ProtoNet(nn.Module):
    """Prototypical network based on MobileViT"""
    - Encoder: MobileViT backbone network
    - Prototype Generation: DPCM or mean pooling
    - Distance Metric: Euclidean distance or cosine similarity
```

#### 3. Dataset Processing (`dataset.py`)
- **ISIC2018Dataset**: Load ISIC-2018 dataset
- **CategoriesSampler**: Few-Shot sampler (N-way K-shot sampling)
- Features:
  - Flexible class filtering (support specifying classes for training/testing)
  - Dynamic sampling of support and query sets
  - Multiple sampling strategies

#### 4. Generic Dataset Processing (`generic_dataset.py`)
Generic interface for loading other dataset formats.

#### 5. Training Scripts

| File Name | Function | Description |
|-----------|----------|-------------|
| `train_hybrid_mobilevit.py` | MobileViT + DPCM | Main hybrid model training |
| `train_hybrid_dpcm.py` | DPCM Module Training | Dynamic prototype calibration training |
| `train_hybrid_mobilenet_v3.py` | MobileNet V3 Training | Baseline model comparison |
| `train_hybrid_resnet.py` | ResNet Training | Deep network baseline |
| `train_ablation_isic.py` | Ablation Study (Training) | Component contribution analysis |
| `train_generalization.py` | Generalization Performance Test | Evaluate model on new classes |
| `train_baseline_comparison.py` | Baseline Comparison | MobileNet V2 vs V3 comparison |

#### 6. Testing and Evaluation Scripts

| File Name | Function |
|-----------|----------|
| `check.py` | Check weight file and model compatibility |
| `judge.py` | Model performance evaluation (accuracy, F1, AUC) |
| `sd198_expert_test.py` | SD-198 dataset expert testing |

#### 7. Analysis Scripts

| File Name | Function |
|-----------|----------|
| `data_preprocess.py` | Data preprocessing and cleaning |
| `data-split.py` | Generate training/validation/test data splits |
| `dataset_analysis.py` | Data statistics and analysis |
| `figure_2.py` | Visualization script |
| `cuda.py` | CUDA environment check |
| `MedMNIST.py` | MedMNIST dataset loader |

### Pre-trained Weights

| File | Size | Purpose |
|------|------|---------|
| `pretrain_weights/mobilevit_xs.pt` | ~5MB | MobileViT-XS pre-trained weights |
| `pretrain_weights/mobilevit_s.pt` | ~15MB | MobileViT-S pre-trained weights |
| `pretrain_weights/mobilevit_xxs.pt` | ~3MB | MobileViT-XXS pre-trained weights |

### Training Checkpoints

| Directory | Contained Models | Performance |
|-----------|-----------------|-------------|
| `checkpoints_xxs/` | Smallest model (XXS) | Optimal DPCM model |
| `checkpoints_xs/` | Lightweight model (XS) | Balanced performance |
| `checkpoints_baselines/` | MobileNet V2/V3 | Baseline comparison |
| `checkpoints_hybrid_xxs/` | Hybrid XXS | Optimized version |
| `checkpoints_mobilenetv3_dpcm/` | MobileNet V3 + DPCM | Fusion model |
| `checkpoints_resnet/` | ResNet Baseline | Deep network |

## Usage Guide

### Prerequisites

#### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support, at least 4GB VRAM
- **CPU**: Multi-core processor (8+ cores recommended)
- **Memory**: At least 16GB RAM
- **Storage**: At least 50GB (for datasets)

#### Operating System
- Windows 10/11 or Linux (Ubuntu 20.04+)
- Python 3.8+

### Environment Dependencies

#### Python Libraries
```
# Core dependencies
torch>=1.12.0  # PyTorch deep learning framework
torchvision>=0.13.0  # Computer vision tools
pytorch-cuda=11.8  # CUDA acceleration

# Data processing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
PIL (Pillow>=8.3.0)

# Training and evaluation
tqdm>=4.62.0  # Progress bar
matplotlib>=3.4.0  # Visualization

# Optional
opencv-python>=4.5.0
tensorboard>=2.6.0  # Training monitoring
```

#### Installation

1. **Create Conda Virtual Environment**
   ```bash
   conda create -n isic-fsl python=3.10
   conda activate isic-fsl
   ```

2. **Install PyTorch (CUDA 11.8)**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install Other Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install numpy pandas scikit-learn pillow tqdm matplotlib opencv-python tensorboard
   ```

### Quick Start

#### 1. Data Preparation

**Download Datasets**
- ISIC-2018: https://challenge.isic-archive.com/
- HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Place data in corresponding directories

**Preprocess Data**
```bash
python data_preprocess.py
python data-split.py
```

#### 2. Model Training

**Basic MobileViT-XS Training**
```bash
python train_hybrid_mobilevit.py \
    --model_type xs \
    --num_classes 7 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

**Train with DPCM Module**
```bash
python train_hybrid_dpcm.py \
    --use_dpcm True \
    --hidden_dim 128 \
    --epochs 100
```

**Ablation Study**
```bash
python train_ablation_isic.py \
    --ablation_type no_dpcm  # Test performance without DPCM
```

**Baseline Comparison**
```bash
python train_baseline_comparison.py \
    --model mobilenet_v2 \
    --k_shot 5
```

#### 3. Model Evaluation

**Single Model Evaluation**
```bash
python judge.py \
    --model_path checkpoints_xxs/best_dpcm_5shot.pth \
    --test_csv ISIC_Splits/isic_test_split.csv
```

**Generalization Performance Test (New Classes)**
```bash
python train_generalization.py \
    --test_classes [5, 6] \
    --n_way 2 \
    --k_shot 5
```

**SD-198 Dataset Testing**
```bash
python sd198_expert_test.py \
    --model_path checkpoints_xxs/best_model.pth
```

#### 4. Check Weight Compatibility

```bash
python check.py
```
This script verifies the compatibility of pre-trained weights with the model definition.

#### 5. CUDA Environment Check

```bash
python cuda.py
```
Verify GPU availability and CUDA version.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_type` | 'xs' | MobileViT model size (xs/s/xxs) |
| `--num_classes` | 7 | Number of classification classes |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 0.001 | Learning rate |
| `--use_dpcm` | True | Whether to use DPCM module |
| `--k_shot` | 5 | K value in Few-Shot (1 or 5) |
| `--n_way` | 5 | N value in Few-Shot (number of classes) |
| `--device` | 'cuda:0' | Computing device |

### Project Directory Structure

```
ISIC_project/
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Dependency list
│
├── ==================== Model Definitions ====================
├── mobilevit.py                       # MobileViT backbone network
├── fsl_model.py                       # Few-Shot Learning model
├── generic_dataset.py                 # Generic dataset loader
│
├── ==================== Data Processing ====================
├── dataset.py                         # ISIC dataset loader
├── data_preprocess.py                 # Data preprocessing
├── data-split.py                      # Data split script
├── dataset_analysis.py                # Data analysis
│
├── ==================== Training Scripts ====================
├── train_hybrid_mobilevit.py          # MobileViT + DPCM main training
├── train_hybrid_dpcm.py               # DPCM specialized training
├── train_hybrid_mobilenet_v3.py       # MobileNet V3 training
├── train_hybrid_resnet.py             # ResNet training
├── train_ablation_isic.py             # Ablation study
├── train_baseline_comparison.py       # Baseline comparison
├── train_generalization.py            # Generalization test
├── train_generalization....1.py       # Generalization backup
│
├── ==================== Evaluation Scripts ====================
├── check.py                           # Weight compatibility check
├── judge.py                           # Performance evaluation
├── sd198_expert_test.py               # SD-198 testing
│
├── ==================== Utility Scripts ====================
├── cuda.py                            # CUDA environment check
├── MedMNIST.py                        # MedMNIST dataset loader
├── figure_2.py                        # Visualization script
│
├── ==================== Dataset Directories ====================
├── ISIC-2018-dataset/                 # ISIC-2018 official data
├── HAM10000/                          # HAM10000 skin lesion data
├── SD-198/                            # SD-198 dataset
├── ISIC_Splits/                       # Pre-defined data splits
│
├── ==================== Pre-trained Weights ====================
├── pretrain_weights/
│   ├── mobilevit_xs.pt
│   ├── mobilevit_s.pt
│   └── mobilevit_xxs.pt
│
├── ==================== Training Checkpoints ====================
├── checkpoints_xxs/
├── checkpoints_xs/
├── checkpoints_baselines/
├── checkpoints_hybrid_xxs/
├── checkpoints_mobilenetv3_dpcm/
└── checkpoints_resnet/
```

## Method Steps

### 1. Data Preprocessing Stage

```
Raw Dataset
    ↓
Data Loading and Validation (data_preprocess.py)
    ↓
Image Normalization and Augmentation (resize, normalize)
    ↓
Data Splitting (train/val/test) (data-split.py)
    ↓
Ready to Use
```

### 2. Model Training Stage

#### Stage A: Base Model Pre-training (Optional)
```
MobileViT Backbone + Full Classification Head
    ↓
Train on ISIC-2018 Training Set
    ↓
Obtain Base Feature Extraction Capability
```

#### Stage B: Few-Shot Learning Fine-tuning
```
Load Pre-trained MobileViT
    ↓
Remove Classification Head, Get Feature Vectors
    ↓
Generate Dynamic Prototypes using DPCM
    ↓
Optimize on K-shot Support Set
    ↓
Evaluate on Query Set
```

### 3. Workflow Diagram

```
┌─────────────────┐
│   Query Set     │
│  (New Classes)  │
└────────┬────────┘
         │
    ┌────▼──────────────────┐
    │  MobileViT Feature    │
    │  Extraction           │
    │ (From Pre-trained     │
    │  Backbone)            │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │   Support Set         │
    │   Feature Extraction  │
    │ (K Labeled Samples)   │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  DPCM Prototype       │
    │  Generation           │
    │ (Dynamic Weight       │
    │  Calibration)         │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  Distance Calculation │
    │  and Matching         │
    │ (Query vs Prototype)  │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │   Classification      │
    │   Output              │
    │ (N-way softmax)       │
    └──────────────────────┘
```

### 4. Core Algorithm Steps

#### DPCM (Dynamic Prototype Calibration Module)

```
Input: Support Set Features X_s ∈ ℝ^(K×D)
       (K samples, each D-dimensional)

Step 1: Attention Computation
  a_i = AttentionNet(x_i)  ∀i ∈ [1,K]
  where AttentionNet is MLP: D → 128 → 1

Step 2: Weight Normalization
  w_i = softmax(a_i)  ∀i ∈ [1,K]

Step 3: Dynamic Prototype Generation
  p = Σ(w_i * x_i)
  (Weighted Sum: Higher weight samples contribute more to prototype)

Output: Dynamic Prototype Vector p ∈ ℝ^D
```

#### Inference Process (Few-Shot Classification)

```
For N-way K-shot Problem:

Step 1: Encoding
  q = Encoder(query_image)      # Query sample feature
  s_k = Encoder(support_k)      ∀k ∈ [1,K]  # Support set features

Step 2: Prototype Generation (for each class)
  p_c = DPCM([s_1^c, s_2^c, ..., s_K^c])  ∀c ∈ [1,N]

Step 3: Similarity Computation
  dist_c = ||q - p_c||_2  ∀c ∈ [1,N]

Step 4: Classification
  scores = -dist_c (or cos_similarity)
  prediction = argmin(dist_c)
```

### 5. Training Configuration

```python
# Data Configuration
n_way = 5              # Number of classes
k_shot = 5             # Support set samples per class
query_per_class = 15   # Query samples per class

# Model Configuration
feature_dim = 384      # MobileViT-XS output dimension
hidden_dim_dpcm = 128  # DPCM hidden layer dimension
use_dpcm = True        # Enable DPCM

# Training Configuration
epochs = 100
batch_size = 32
learning_rate = 0.001
scheduler = 'cosine'   # Cosine annealing learning rate
optimizer = 'adam'

# Evaluation Configuration
k_shot_eval = 5        # K value for evaluation
n_way_eval = 5         # N value for evaluation
num_episodes = 600     # Number of evaluation episodes
```

## References

### Core Methodology

1. **Few-Shot Learning**
   - Snell, J., Swersky, K., & Zemel, R. S. (2017). 
     "Prototypical Networks for Few-shot Learning."
     NIPS 2017.

2. **MobileViT Architecture**
   - Mehta, S., & Rastegari, M. (2022).
     "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer."
     ICCV 2021.

3. **MobileNet V3**
   - Howard, A., et al. (2019).
     "Searching for MobileNetV3."
     ICCV 2019.

### Datasets

4. **ISIC-2018 Challenge**
   - Codella, N. C. F., et al. (2019).
     "Skin Lesion Analysis Toward Melanoma Detection: 
     ISIC 2018 Challenge."
     arXiv:1902.03368

5. **HAM10000 Dataset**
   - Tschandl, P., Rosendahl, C., & Kittler, H. (2018).
     "The HAM10000 Dataset: A Large Collection of Multi-Source 
     Dermatoscopic Images of Common Pigmented Skin Lesions."
     Scientific Data, 5, 180161.

### Related Medical Applications

6. **Skin Lesion Classification Survey**
   - Esteva, A., et al. (2019).
     "Dermatologist-level Classification of Skin Cancer with 
     Deep Neural Networks."
     Nature Medicine, 25(2), 302-308.

## Open Source License

### Project License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024-2025 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE OR ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Dataset Licenses

- **ISIC-2018**: https://www.isic-archive.com/ (CC BY-NC-SA 4.0)
- **HAM10000**: Kaggle Dataset License
- **MedMNIST**: CC BY 4.0

### Third-party Components

- **PyTorch**: BSD License
- **NumPy/Pandas**: BSD License
- **scikit-learn**: BSD License
- **Pillow**: HPND License
- **OpenCV**: Apache 2.0 License

## FAQ (Frequently Asked Questions)

### Q1: How to load pre-trained weights?
A: Use the `MobileViT` class in `mobilevit.py`:
```python
model = MobileViT(model_type='xs', num_classes=7)
model.load_state_dict(torch.load('pretrain_weights/mobilevit_xs.pt'))
```

### Q2: How to adjust Few-Shot N-way K-shot parameters?
A: Modify the sampler parameters in the training script:
```python
sampler = CategoriesSampler(
    labels, n_way=5, k_shot=5, query_per_class=15
)
```

### Q3: What to do if GPU memory is insufficient?
A: Try the following:
- Reduce `batch_size` (from 32 to 16 or 8)
- Use a smaller model (`xxs` or `xs`)
- Enable gradient accumulation or mixed precision training

### Q4: How to fine-tune on a new dataset?
A: Modify the data loading path in `dataset.py`, or inherit the `Dataset` class to implement a custom data loader.

### Q5: Does it support multi-GPU training?
A: Yes, you can use `DataParallel` or `DistributedDataParallel`:
```python
model = nn.DataParallel(model, device_ids=[0, 1])
```

## Contact and Feedback

- **Bug Reports**: Submit issues in the project Issues section
- **Feature Suggestions**: Welcome to submit Pull Requests
- **Academic Discussion**: [Your Email/Website]

## Acknowledgments

Thanks to the following resources for their support:
- ISIC Foundation for providing datasets
- PyTorch community
- Open-source contributions from the deep learning research community
