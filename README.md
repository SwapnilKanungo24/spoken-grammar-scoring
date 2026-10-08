# 🎙️ Spoken Grammar Scoring Engine

> An AI-powered regression system for automatically scoring the grammatical quality of spoken English audio on a continuous 0–5 scale.

## 📌 Overview

Evaluating spoken grammar manually can be time-consuming and difficult to scale. This project develops a machine learning pipeline that analyzes spoken audio and predicts a continuous grammar score from **0 to 5**.

The system combines **speech transcription, linguistic analysis, and acoustic speech characteristics** to capture complementary information from each spoken response.

The project was developed as part of the **SHL Hiring Assessment 2026** competition.

---

## 🎯 Problem Statement

Given a spoken English audio sample, predict its grammar quality score on a continuous **0–5 scale**.

The dataset contains:

- **769 labeled training audio samples**
- **216 unlabeled test audio samples**
- Audio duration of approximately **40–60 seconds**
- Continuous grammar-quality scores

The primary evaluation metrics are:

- **RMSE (Root Mean Squared Error)**
- **Pearson Correlation**

---

## 🧠 Approach

The solution follows a multimodal feature engineering and ensemble learning pipeline.

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

---

## 🔊 1. Speech Transcription

The spoken audio was transcribed using **OpenAI Whisper (Base)**.

Whisper converts each spoken response into text, allowing linguistic characteristics to be extracted from the transcript.

Missing training transcripts were identified and reprocessed to ensure that the final feature matrix contained complete transcript-derived information.

---

## 📝 2. Linguistic Feature Engineering

Eight transcript-based features were extracted:

| Feature | Description |
|---|---|
| Word Count | Total number of words in the transcript |
| Sentence Count | Number of detected sentences |
| Average Sentence Length | Average number of words per sentence |
| Average Word Length | Average number of characters per word |
| Vocabulary Diversity | Ratio of unique words to total words |
| Repeated Word Count | Number of repeated words |
| Repeated Word Ratio | Proportion of repeated words |
| Filler Count | Frequency of detected filler words |

These features capture aspects of sentence structure, vocabulary usage, repetition, and speaking fluency.

---

## 🎵 3. Acoustic Feature Engineering

Audio characteristics were extracted using **Librosa**.

The **33 acoustic features** include:

- Audio duration
- RMS energy mean
- RMS energy standard deviation
- Zero-crossing rate mean
- Zero-crossing rate standard deviation
- Pitch mean
- Pitch standard deviation
- 13 MFCC means
- 13 MFCC standard deviations

These features provide information about speech delivery, acoustic variation, pitch characteristics, energy patterns, and spectral properties.

---

## 🔗 4. Multimodal Feature Set

The linguistic and acoustic features were combined into a single feature matrix.

```text
8 Linguistic Features
        +
33 Acoustic Features
        =
41 Total Features
```

| Dataset | Shape |
|---|---|
| Training feature matrix | 769 samples × 41 features |
| Test feature matrix | 216 samples × 41 features |

The training and test feature columns were explicitly checked to ensure identical ordering before prediction.

---

## 🤖 5. Machine Learning Models

Two tree-based regression models were selected based on validation performance.

### Random Forest Regressor

```text
n_estimators     = 300
max_depth        = 10
min_samples_leaf = 2
max_features     = 1.0
random_state     = 42
```

### Gradient Boosting Regressor

```text
n_estimators     = 300
learning_rate    = 0.03
max_depth        = 2
min_samples_leaf = 5
loss             = "huber"
random_state     = 42
```

---

## ⚖️ 6. Ensemble Strategy

The final system combines predictions from both models using equal weighting.

```text
Final Prediction =
    0.5 × Random Forest Prediction
  + 0.5 × Gradient Boosting Prediction
```

The ensemble was selected because it achieved better out-of-fold performance than the individual models evaluated during experimentation.

---

## 📊 7. Model Evaluation

Five-fold cross-validation was used to estimate the model's generalization performance.

### Final Ensemble — Out-of-Fold Performance

| Metric | Score |
|---|---|
| OOF RMSE | **0.7513** |
| OOF Pearson Correlation | **0.7965** |

### Final Ensemble — Training Performance

| Metric | Score |
|---|---|
| Training RMSE | 0.4764 |
| Training Pearson Correlation | 0.9320 |

> **Note:** Training metrics can be optimistic because the model is evaluated on the same data used for fitting. The out-of-fold validation metrics are more representative of expected generalization performance.

---

## 📈 Model Development

Several feature and model configurations were evaluated during development. The main progression was:

```text
Text-only features
        │
        ▼
Audio features added
        │
        ▼
41-feature multimodal model
        │
        ▼
Random Forest + Gradient Boosting
        │
        ▼
50/50 Ensemble
        │
        ▼
OOF RMSE: 0.7513
Pearson: 0.7965
```

The final approach uses both:

- **What was spoken** — linguistic information
- **How it was spoken** — acoustic information

This provides a more comprehensive representation of spoken grammar quality than using transcript features alone.

---

## 📊 Visual Analysis

The accompanying notebook contains visualizations covering:

- Training grammar-score distribution
- Model performance comparison
- Out-of-fold actual vs. predicted scores
- Residual error analysis
- Feature importance from the final tree-based models

These visualizations were used to understand the dataset, compare models, and analyze prediction behavior.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Speech Recognition | OpenAI Whisper |
| Audio Processing | Librosa |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Development Environment | Kaggle Notebook, Jupyter |
| Version Control | Git, GitHub |

---

## 📂 Repository Structure

```text
spoken-grammar-scoring/
│
├── SHL_Grammar_Scoring_Final_Clean.ipynb
├── Swapnil_Kanungo.csv
└── README.md
```

> The raw competition dataset and audio files are not included in this repository.

---

## 🚀 Reproducibility

To reproduce the project:

1. Clone or download this repository.
2. Open `SHL_Grammar_Scoring_Final_Clean.ipynb`.
3. Attach the required competition dataset.
4. Enable GPU acceleration for Whisper.
5. Run the notebook from top to bottom.
6. The notebook performs transcription, feature extraction, model training, evaluation, and test prediction.
7. The final predictions are saved as a CSV submission file.

The notebook is organized as an end-to-end machine learning pipeline so that the complete workflow can be reproduced from preprocessing through final prediction.

---

## 💡 Key Takeaways

- Speech transcription provides useful linguistic information for automated grammar assessment.
- Acoustic characteristics provide complementary information beyond the transcript.
- Combining linguistic and acoustic features produces a stronger representation of spoken responses.
- Ensemble learning can improve predictive performance by combining different tree-based regression models.
- Out-of-fold validation provides a more reliable estimate of generalization than training performance alone.

---

## ⭐ Project Highlights

- 🎙️ Automated spoken grammar assessment
- 🧠 Whisper-based speech transcription
- 📝 Linguistic feature engineering
- 🎵 Acoustic feature engineering
- 🤖 Ensemble regression
- 📊 Five-fold cross-validation
- 📈 RMSE and Pearson correlation evaluation
- 🔬 Feature importance and residual analysis
- 🚀 End-to-end reproducible ML pipeline

---

## 👨‍💻 Author

**Swapnil Kanungo**
B.Tech — Computer Science and Information Technology

Interested in:

- Artificial Intelligence
- Machine Learning
- Data Science
- Backend Development

[GitHub](https://github.com/SwapnilKanungo24) · [LinkedIn](https://linkedin.com/in/swapnil-kanungo-181364324)

---

## 📜 License

This repository is intended for educational, portfolio, and research purposes.
