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
├── README.md                # This file
├── pilot_data/              # Pilot dataset (English)
├── train_data/              # Training data (released Sep 1, 2025)
└── evaluation_kit/          # Official evaluation scripts
```

## 🎯 Four Subtasks

| Subtask | Task | Languages | Key Challenge |
|---------|------|-----------|---------------|
| 1 | Binary validity classification | English | Mitigate content effect |
| 2 | Validity + relevant premise selection | English | Filter noisy premises |
| 3 | Binary validity classification | 12 languages | Cross-lingual generalization |
| 4 | Validity + premise selection | 12 languages | Multilingual + noise handling |

**Target Languages**: English, German, Spanish, French, Italian, Dutch, Portuguese, Russian, Chinese, Swahili, Bengali, Telugu

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

**Key Point**: Models must predict `validity`, ignoring `plausibility`.

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
| Sep 1, 2025 | Training data release |
| Oct 31, 2025 | Evaluation kit & output format release |
| Dec 1, 2025 | Test data & Codabench competition starts |
| Dec 10-31, 2025 | Practice phase |
| **Jan 1-31, 2026** | **Evaluation phase** |
| Feb 2026 | Paper submission deadline |

## 🚀 Current Status

- [x] Repository setup
- [x] Pilot data downloaded
- [x] Evaluation kit obtained
- [ ] Baseline implementation (in progress)
- [ ] Training scripts (planned)
- [ ] Evaluation pipeline (planned)



## 📄 License

MIT License

---

**Note**: This repository is under active development. Scripts and documentation will be updated progressively.
