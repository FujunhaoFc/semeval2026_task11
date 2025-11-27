# SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in LLMs

This repository contains my implementation for SemEval-2026 Task 11, focusing on building models that can assess the formal validity of syllogistic arguments independent of their plausibility across multiple languages.

## 📌 Task Overview

**Challenge**: Large Language Models confuse formal logical validity with argument plausibility - they tend to:
- ✅ Overestimate validity of plausible arguments
- ❌ Underestimate validity of implausible but logically sound arguments

**Goal**: Build models that judge **logical structure**, not content truthfulness.

**Official Links**:
- 🌐 Website: https://sites.google.com/view/semeval-2026-task-11
- 💻 GitHub: https://github.com/neuro-symbolic-ai/semeval_2026_task_11
- 💬 Slack: [Join Channel](https://join.slack.com/t/semeval-2026task11/shared_invite/zt-3axe6sgpu-Im7YqfSOIs~GYSp_TTKmDQ)

## 📂 Repository Structure

```
semeval2026_task11/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── pilot_data/                        # Pilot data
│   └── syllogistic_reasoning_binary_pilot_en.json
├── train_data/                        # Training data
│   └── task1/
│       └── train_data.json           # 960 samples (480 valid, 480 invalid)
├── outputs_baseline/                  # Baseline experiment outputs
│   ├── best_model.pt
│   └── training_summary.txt
├── outputs_llm_generate/              # LLM augmentation experiment outputs
│   ├── best_model.pt
│   └── training_summary.txt
├── outputs_instruction_tuning/        # Instruction tuning outputs (planned)
├── generated_data.json                # LLM-generated synthetic data (2000 samples)
├── train_baseline.py                  # Baseline training script
├── train_with_llm_generation.py       # LLM data augmentation script
└── train_instruction_tuning.py        # Instruction tuning script
```

## 🎯 Subtask Focus

Currently focusing on **Subtask 1**: Binary validity classification (English)

| Subtask | Task | Languages | Key Challenge |
|---------|------|-----------|---------------|
| **1** | Binary validity classification | English | Mitigate content effect |
| 2 | Validity + relevant premise selection | English | Filter noisy premises |
| 3 | Binary validity classification | 12 languages | Cross-lingual generalization |
| 4 | Validity + premise selection | 12 languages | Multilingual + noise handling |

## 📊 Data Format

Each example contains a syllogism with validity and plausibility labels:

```json
{
  "id": "0",
  "syllogism": "Not all canines are aquatic creatures known as fish. It is certain that no fish belong to the class of mammals. Therefore, every canine falls under the category of mammals.",
  "validity": false,
  "plausibility": true
}
```

**Training Data Statistics**:
- Total samples: 960
- Valid syllogisms: 480 (50%)
- Invalid syllogisms: 480 (50%)
- Train/Val split: 85%/15% (816/144)

---

## 🧪 Experiments

### Experiment 1: Baseline (DeBERTa + LoRA + SCL)

**Method**: Supervised Contrastive Learning with DeBERTa-v3-large

| Component | Description |
|-----------|-------------|
| Base Model | `microsoft/deberta-v3-large` |
| Fine-tuning | LoRA (rank=16, alpha=32) |
| Loss | Cross-Entropy + Supervised Contrastive Loss (SCL) |
| Data Augmentation | 3x augmentation + 500 synthetic samples |

**Configuration**:
```yaml
Model: microsoft/deberta-v3-large
LoRA:
  rank: 16
  alpha: 32
  dropout: 0.15
Training:
  learning_rate: 3e-4
  batch_size: 16
  epochs: 20 (early stopped at 7)
  gradient_accumulation: 2
SCL:
  temperature: 0.07
  projection_dim: 256
  loss_weight: 0.5
Data:
  training_samples: 2948 (816 original + augmentation)
  validation_samples: 144
```

**Results**:
| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **96.53%** |
| Training Accuracy | 99.8% |
| Total Epochs | 7 (early stopping) |

---

### Experiment 2: LLM Data Augmentation

**Method**: Use LLM to generate additional training samples

| Component | Description |
|-----------|-------------|
| Base Model | `microsoft/deberta-v3-large` |
| LLM Provider | SiliconFlow (free API) |
| LLM Model | `Qwen/Qwen3-8B` |
| Generated Samples | 2000 (500 per category) |

**Four Categories Generated**:
1. **Valid + Plausible**: Correct logic with realistic content
2. **Valid + Implausible**: Correct logic with absurd content
3. **Invalid + Plausible**: Logical fallacies with realistic content
4. **Invalid + Implausible**: Logical fallacies with absurd content

**Configuration**:
```yaml
Model: microsoft/deberta-v3-large
LoRA:
  rank: 16
  alpha: 32
  dropout: 0.15
Training:
  learning_rate: 2e-4
  batch_size: 4  # Reduced due to SCL memory usage
  epochs: 20 (early stopped at 12)
  gradient_accumulation: 2
LLM Generation:
  provider: siliconflow
  model: Qwen/Qwen3-8B
  samples: 2000
  deduplication: enabled
Data:
  training_samples: 2816 (816 original + 2000 generated)
  validation_samples: 144 (original data only)
```

**Results**:
| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **98.61%** |
| Training Accuracy | 99.75% |
| Total Epochs | 12 (early stopping) |
| Improvement over Baseline | +2.08% |

**Training Progress**:
| Epoch | Train Acc | Val Acc | Status |
|-------|-----------|---------|--------|
| 1 | 76.28% | 64.58% | - |
| 2 | 94.28% | 79.17% | ↑ |
| 3 | 97.51% | 89.58% | ↑ |
| 4 | 98.65% | 92.36% | ↑ |
| 5 | 99.57% | 93.75% | ↑ |
| 6 | 99.72% | 95.83% | ↑ |
| 7 | 99.75% | **98.61%** | ✓ Best |
| 8-12 | ~100% | 97-98% | Early stopping |

---

### Experiment 3: LLM-Only Training (Negative Result)

**Purpose**: Test if LLM-generated data alone can train a model

**Setup**:
- Training: 2000 LLM-generated samples only
- Validation: 960 original samples (full dataset)

**Results**:
| Metric | Value |
|--------|-------|
| Validation Accuracy | **~50%** (random guessing) |
| Conclusion | LLM-generated data distribution differs significantly from real data |

**Analysis of Generated Data Quality**:
- Sampled 20 random generated samples
- Found ~20% label errors (e.g., invalid syllogisms labeled as valid)
- Qwen3-8B has systematic errors with complex logical structures
- Common mistake: Confusing "Undistributed Middle" fallacy with valid Darii form

**Key Insight**: LLM-generated data can **augment** but not **replace** real data.

---

### Experiment 4: Few-shot Instruction Tuning (Planned)

**Method**: Convert classification task to generation task with instruction prompts

**Approach**:
- Model: Qwen2.5-7B-Instruct with LoRA
- Framework: Unsloth for efficient fine-tuning
- Input: Syllogism + instruction on how to judge validity
- Output: VALID/INVALID with reasoning

**System Prompt Design**:
```
You are an expert in formal logic. Determine if the syllogism is 
logically VALID or INVALID. Focus ONLY on logical structure, 
ignore whether the content is believable.

Valid forms: Barbara, Celarent, Darii, Cesare...
Invalid patterns: Undistributed Middle, Illicit Major...
```

Status: **In Progress**

---

## 📈 Results Summary

| Experiment | Method | Val Accuracy | Notes |
|------------|--------|--------------|-------|
| Baseline | DeBERTa + LoRA + SCL | 96.53% | Strong baseline |
| LLM Augment | + 2000 Qwen3-8B samples | **98.61%** | +2.08% improvement |
| LLM Only | Train on generated data only | ~50% | Distribution mismatch |
| Instruction Tuning | Qwen2.5-7B + LoRA | TBD | In progress |

## 📈 Evaluation Metrics

- **Accuracy**: Correct validity predictions
- **Content Effect (CE)**: Measures bias towards plausibility
  - Intra-Plausibility CE: Bias within plausibility categories
  - Cross-Plausibility CE: Bias across plausibility values
  - Total CE = (Intra CE + Cross CE) / 2
- **Ranking Score**: `Accuracy / Total Content Effect` (higher is better)

**Goal**: High accuracy + Low content effect

## ⏰ Important Timeline

| Date | Event |
|------|-------|
| Sep 1, 2025 | Training data release ✅ |
| Oct 31, 2025 | Evaluation kit & output format release |
| Dec 1, 2025 | Test data & Codabench competition starts |
| Dec 10-31, 2025 | Practice phase |
| **Jan 1-31, 2026** | **Evaluation phase** |
| Feb 2026 | Paper submission deadline |

## 🚀 Quick Start

### Environment Setup

```bash
# Clone and setup
cd ~/semeval2026_task11
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Baseline

```bash
python train_baseline.py
```

### Run LLM Augmentation

```bash
export SILICONFLOW_API_KEY="your-key"
python train_with_llm_generation.py \
    --provider siliconflow \
    --model "Qwen/Qwen3-8B" \
    --llm_samples 2000 \
    --batch_size 4
```

### Run Instruction Tuning

```bash
python train_instruction_tuning.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --epochs 3 \
    --batch_size 4
```

## 📄 License

MIT License

---

**Last Updated**: November 2025