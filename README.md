# শব্দতরী: Bengali Regional Dialect ASR
### AI-FICATION 2025 (Datathon) - Team DejaView Solution

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.47.0-FFD21E?logo=huggingface)](https://huggingface.co/)

**Team Members:** [Ruwad Naswan](https://github.com/rwd51) • [Shadab Tanjeed](#) • [Abrar Zahin Raihan](https://github.com/azraihan)

For more details, look into the [presentation](https://github.com/azraihan/Team-DejaView---CUET-ETE-Televerse-AIFication-2025-Datathon-Solution/blob/main/presentation/ai_fication_presentation.pdf) and the [paper](#).

---

## 🎯 Problem Statement

Transcribe **20 regional Bangladeshi dialects** into standard Bangla text with high accuracy despite phonetic variations and diverse acoustic conditions.

### Dataset Overview
- **Training:** 3,350 audio files
- **Test:** 450 audio files
- **Dialects:** 20 regional variations
- **Format:** 16 kHz, mono WAV
- **Total Duration:** 3.90 hours
- **Avg Duration:** 4.2 seconds/sample
- **Vocabulary:** 590 unique words
- **Avg Words/Sample:** 5.4

---

## 🏆 Solution Overview

Our solution achieved **0.93509 NLS (Public)** and **0.91782 NLS (Private)** on the leaderboard through:

1. **Four Fine-tuned Whisper Medium Models** with different training strategies
2. **Advanced Audio Preprocessing** (denoising, normalization, padding)
3. **Data Augmentation** (on-the-fly per epoch)
4. **Weighted Sampling** for dialect balance
5. **External Dataset Integration** (RegSpeech12 with LLM-based standardization)
6. **ROVER Ensemble** for final predictions

---

## 📊 Model Architecture

### Base Model
**BengaliAI Whisper Medium** - Pre-trained on Bengali ASR

### Four Model Variants

| Model | Training Strategy | Public NLS | Private NLS |
|-------|------------------|------------|-------------|
| **Model 1** | Encoder-Only Fine-tuning (Decoder Frozen) | 0.91996 | 0.89496 |
| **Model 2** | Full Fine-tuning (Standard → Standard) | 0.90581 | 0.89294 |
| **Model 3** | Regional Classifier Adapter | 0.91488 | 0.89959 |
| **Model 4** | Full Fine-tuning (Regional → Regional) | 0.91096 | 0.88103 |
| **Ensemble** | ROVER (4 models) | **0.93509** | **0.91782** |

### Model 1: Encoder-Only Fine-tuning

![Frozen Decoder Fine-tuning Architecture](images/Frozen%20Decoder%20Fine-tuning.png)

This approach trains only the encoder layers while keeping the decoder frozen, adapting the acoustic encoder to regional dialects while preserving the decoder's language model capabilities.

---

## 🔧 Data Preprocessing

### Audio Preprocessing
- **Input Format:** 16 kHz, mono WAV (standardized)
- **Denoising:** Dynamic spectral gating via `librosa` + `noisereduce`
- **Normalization:** Peak normalization to -3 dB for consistency
- **Padding:** 3.5s silence added to short clips (<10s)
- **Zero-length Filtering:** Clips <1s eliminated
- **No Trimming:** Long clips kept as-is to preserve dialectal features

### Text Preprocessing
- **Quality:** Pre-cleaned standard Bangla text (UTF-8 verified)
- **No Additional Processing:** Foreign words and noise already handled

### Feature Engineering
- **Log-Mel Spectrograms** extracted directly from preprocessed audio for Whisper input

---

## 🎨 Data Augmentation

### On-The-Fly Augmentation
Applied **during training** (fresh augmentations each epoch):
- **Time Stretching** (0.9-1.1x)
- **Pitch Shifting** (±2 semitones)
- **Gaussian Noise Injection** (0.001-0.01 amplitude)
- **Volume Adjustment** (-6 to +6 dB)
- **Probability:** 50% per sample

---

## ⚖️ Balanced Sampling Strategy

### Problem
Severe class imbalance across 20 dialects:
- **Largest:** Chittagong (401 samples)
- **Smallest:** Khulna (21 samples)

### Solution: Weighted Random Sampling
```
Weight_region = Total_Samples / (Number_of_Classes × Region_Count)
```
- Khulna receives **20× higher weight** than Chittagong
- Sampler **resamples every epoch** automatically
- Ensures fair dialect representation during training

---

## 📦 External Dataset Integration

### RegSpeech12 Dataset
- **Problem:** Several dialects had <100 samples
- **Solution:** Added 290 samples from RegSpeech12 dataset
- **Conversion:** Regional dialect text → Standard Bangla using **Gemini 2.5 Flash**
- **Regions Enhanced:** Noakhali (+70), Barisal (+68), Comilla (+62), Chittagong (+30), Sylhet (+30), Rangpur (+30)

---

## 🧠 Model 3: Regional Classifier Adapter

### Architecture
Novel multi-task learning approach with encoder-level regional conditioning:

1. **Region Embedding:** Each region → 64-d learnable vector
2. **Projection:** 64-d → 1024-d to match encoder hidden states
3. **Adapter Injection:** Regional vector added to encoder outputs
4. **Stabilization:** LayerNorm applied for balance
5. **Auxiliary Classifier:** Mean-pooling → Linear layer → Region prediction

![Regional Classifier Adapter Architecture](images/Regional%20Classifier%20Adapter.png)

### Multi-task Loss
```
Total Loss = Loss_ASR + α · Loss_Region
```

### Impact
- **Before:** Silhouette Score = -0.3262 (overlapping clusters)
- **After:** Silhouette Score = **0.8778** (clear dialect separation)

---

## 🏋️ Training Configuration

### Hardware
- **GPU:** NVIDIA P100 (16GB)
- **Platform:** Kaggle Notebooks

### Hyperparameters
```python
{
    'batch_size': 1,
    'gradient_accumulation': 4,  # Effective batch size: 4
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'num_epochs': 6,
    'fp16': True,
    'optimizer': 'AdamW',
}
```

### Training Features
✅ **Encoder-only fine-tuning** (Model 1: ~25% parameters trainable)
✅ **On-the-fly augmentation** (fresh each epoch)
✅ **Weighted sampling** for class balance
✅ **Stratified train/val split** (encoder-only: 90/10, full fine-tune: 80/20)

---

## 🎯 Ensemble Strategy

### ROVER (Recognizer Output Voting Error Reduction)
- **Method:** Weighted voting selecting prediction most similar to all others
- **Models:** 4 variants of fine-tuned Whisper Medium

---

## 📈 Results & Analysis

### Evaluation Metric
**Normalized Levenshtein Similarity (NLS)**
```
NLS(r, p) = 1 - LevenshteinDistance(r, p) / max(|r|, |p|)
```

### Key Findings
- **No systematic failure patterns** (t-SNE analysis shows random error distribution)
- **OOV dialectal words** occasionally cause spelling mistakes
- **Frozen decoder approach** (Model 1) performed best individually

---

## 📁 Repository Structure

```
.
├── README.md
├── ai_fication_presentation.pdf                    # Presentation slides
├── DejaView_whispermedium_encoderonly_train.ipynb  # Model 1: Encoder-only
├── DejaView_whispermedium_standardBfinetune_train.ipynb  # Model 2: Full fine-tuning
├── DejaView_Whisper_Classifier_Training.ipynb      # Model 3: Regional adapter
├── DejaView_Whisper_STDProcessing_Training.ipynb   # Audio preprocessing + inference
```

---

## 🔬 Key Innovations

1. **Encoder-Only Fine-tuning:** Adapts acoustic encoder to dialects while preserving decoder's language model
2. **Regional Classifier Adapter:** Multi-task learning with encoder-level dialect conditioning
3. **LLM-based Dataset Augmentation:** Gemini 2.5 Flash for regional → standard text conversion
4. **On-the-Fly Augmentation:** Fresh augmentations each epoch (not pre-computed)
5. **Weighted Epoch Sampling:** Automatic resampling per epoch for balanced training

---

## 🔮 Future Directions

1. **Expand Dataset:** Collect more regional dialectal data for low-resource regions
2. **Address Imbalances:** Targeted data collection for gender and dialect balance
3. **ASR-LLM Projection Coupling:** Lightweight projection layers for acoustically-grounded text generation
4. **Dialectal Vocabulary Expansion:** Specialized tokenizer for regional words


---

## 📝 Citation

```bibtex
@misc{dejaview2025shobdotori,
  title={শব্দতরী: Where Dialects Flow into Bangla},
  author={Naswan, Ruwad and Tanjeed, Shadab and Raihan, Abrar Zahin},
  year={2025},
  organization={Team DejaView - CUET},
  note={AI-FICATION 2025 Competition Solution}
}
```