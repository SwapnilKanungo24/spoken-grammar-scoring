# 🎙️ Spoken Grammar Scoring Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A multimodal regression system that scores the grammatical quality of spoken English audio on a continuous **0–5 scale**, combining Whisper transcripts with acoustic features and a Random Forest + Gradient Boosting ensemble.

Built for the **SHL Hiring Assessment 2026** competition.

## 📊 Results

| Metric (5-fold CV, 769 samples) | Score |
|---|---|
| **Out-of-fold RMSE** | **0.7513** |
| **Out-of-fold Pearson correlation** | **0.7965** |

Adding acoustic features cut RMSE from 0.9926 (text-only) to ~0.759, and the final ensemble improved it further to 0.7513.

![OOF Actual vs Predicted](images/oof_scatter.png)

## 🎯 Problem & Dataset

Given a 40–60 second spoken English audio sample, predict its grammar score (continuous, 0–5).

- **769** labeled training samples
- **216** unlabeled test samples
- **Metrics:** RMSE and Pearson correlation

> The raw competition dataset and audio files are not included in this repository.

## 🧠 Approach

```text
                    Spoken Audio
                         │
              ┌──────────┴──────────┐
              │                     │
          Whisper                 Librosa
       Transcription            Audio Analysis
              │                     │
              ▼                     ▼
      Linguistic Features      Acoustic Features
           (8)                      (33)
              │                     │
              └──────────┬──────────┘
                         ▼
                41 Combined Features
                         │
              ┌──────────┴──────────┐
              │                     │
       Random Forest        Gradient Boosting
              │                     │
              └──────────┬──────────┘
                         ▼
                 50/50 Ensemble
                         │
                         ▼
              Grammar Score (0–5)
```

The model uses both **what was said** (linguistic) and **how it was said** (acoustic).

## 🔧 Feature Engineering

### Linguistic features (8), from Whisper Base transcripts

| Feature | Description |
|---|---|
| Word Count | Total words in the transcript |
| Sentence Count | Number of detected sentences |
| Average Sentence Length | Average words per sentence |
| Average Word Length | Average characters per word |
| Vocabulary Diversity | Unique words / total words |
| Repeated Word Count | Number of repeated words |
| Repeated Word Ratio | Proportion of repeated words |
| Filler Count | Frequency of detected filler words |

### Acoustic features (33), from Librosa

| Group | Features |
|---|---|
| Duration | Audio duration |
| Energy | RMS mean, RMS std |
| Zero-crossing rate | Mean, std |
| Pitch | Mean, std |
| Spectral | 13 MFCC means, 13 MFCC standard deviations |

**Total:** 8 + 33 = **41 features**. Training matrix: 769 × 41. Test matrix: 216 × 41 (column order verified identical before prediction).

## 🤖 Models & Ensemble

| Parameter | Random Forest | Gradient Boosting |
|---|---|---|
| `n_estimators` | 300 | 300 |
| `max_depth` | 10 | 2 |
| `min_samples_leaf` | 2 | 5 |
| `max_features` | 1.0 | — |
| `learning_rate` | — | 0.03 |
| `loss` | — | `huber` |
| `random_state` | 42 | 42 |

```text
Final Prediction = 0.5 × RandomForest + 0.5 × GradientBoosting
```

## 📈 Evaluation

Five-fold cross-validation was used to estimate generalization.

| Model | Features | RMSE | Pearson | RMSE type |
|---|---|---|---|---|
| Random Forest | Text only (8) | 0.9926 | — | CV mean |
| Random Forest | Multimodal (41) | 0.7590 | 0.7915 | OOF |
| Gradient Boosting | Multimodal (41) | 0.7587 (±0.0381) | — | CV mean |
| **RF + GB ensemble (50/50)** | **Multimodal (41)** | **0.7513** | **0.7965** | **OOF** |

> **Note:** "CV mean" is the average RMSE across the five folds; "OOF" is computed on the pooled out-of-fold predictions. The two are close but not strictly identical metrics.

| Final ensemble | RMSE | Pearson |
|---|---|---|
| Out-of-fold (generalization estimate) | **0.7513** | **0.7965** |
| Training (optimistic, same data used for fitting) | 0.4764 | 0.9320 |

### Visual analysis

| Residuals | Feature Importance |
|---|---|
| ![Residuals](images/residuals.png) | ![Feature Importance](images/feature_importance.png) |

The notebook also contains the score distribution and model comparison plots.

## ⚠️ Limitations

- **Overfitting gap:** training RMSE (0.48) is well below OOF RMSE (0.75).
- **Small dataset:** 769 labeled samples limits how much the models can learn.
- **Transcription noise:** Whisper Base is a small model; transcription errors propagate into the linguistic features.
- **Shallow linguistic features:** filler and repetition detection is rule-based, and there are no syntax or grammar-error features.
- **Tuning bias:** hyperparameters were selected with the same cross-validation setup used for reporting, so the score may be slightly optimistic.

## 🔮 Future Work

- Larger Whisper models (small / medium) for cleaner transcripts
- Grammar-error features (e.g. LanguageTool) and transformer embeddings
- Nested cross-validation for an unbiased estimate
- Stacking with a learned meta-model instead of fixed 50/50 weights

## 🚀 Quick Start

```bash
git clone https://github.com/SwapnilKanungo24/spoken-grammar-scoring.git
cd spoken-grammar-scoring
pip install -r requirements.txt
```

> Whisper requires **FFmpeg** installed on your system (`sudo apt install ffmpeg` / `brew install ffmpeg`).

1. Open `SHL_Grammar_Scoring_Final_Clean.ipynb` (Kaggle or Jupyter).
2. Attach the competition dataset.
3. Enable GPU acceleration for Whisper.
4. Run all cells top to bottom. The notebook handles transcription, feature extraction, training, evaluation, and writes the predictions CSV.

## 📂 Repository Structure

```text
spoken-grammar-scoring/
├── SHL_Grammar_Scoring_Final_Clean.ipynb
├── images/
│   ├── oof_scatter.png
│   ├── residuals.png
│   └── feature_importance.png
├── outputs/
│   └── submission.csv
├── requirements.txt
├── LICENSE
└── README.md
```

## 🛠️ Tech Stack

Python · OpenAI Whisper · Librosa · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Kaggle / Jupyter · Git

## 👨‍💻 Author

**Swapnil Kanungo**, B.Tech in Computer Science and Information Technology

[GitHub](https://github.com/SwapnilKanungo24) · [LinkedIn](https://linkedin.com/in/swapnil-kanungo-181364324)

## 📜 License

Released under the [MIT License](LICENSE).
