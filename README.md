# 📄 SmartDoc AI — Intelligent Research Paper Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-00599C?style=for-the-badge&logo=meta&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)

**An end-to-end NLP pipeline that automatically reads, understands, and analyzes scientific research papers.**

*Classification · Summarization · NER · Semantic Search · RAG Chatbot*

[🚀 Live Demo](#) · [📓 Notebook](#) · [📊 Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)

</div>

---

## 🎯 Project Overview

**SmartDoc AI** is an intelligent document analysis system built as part of a Machine Learning academic project. It transforms raw, lengthy research papers into directly actionable insights — automating the analysis work that would normally take a human several hours to complete.

> *The ultimate goal is to transform large, raw documents into directly actionable insights — automating the analysis work that would normally take a human several hours to complete.*

### ✨ Key Features

| Feature | Description | Technology |
|---|---|---|
| 📂 **Document Classification** | Automatically identify the scientific domain | TF-IDF + Linear SVM |
| 📝 **Auto Summarization** | Generate concise extractive summaries | TF-IDF sentence scoring |
| 🔍 **Semantic Search** | Find the most relevant papers for any query | FAISS + sentence-transformers |
| 🏷️ **Named Entity Recognition** | Extract authors, organizations, locations | spaCy |
| 🤖 **Deep Q&A Chatbot** | Answer any deep question about a paper | Groq (Llama 3.3 70B) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Git
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/benameur21/SmartDoc-AI.git
cd SmartDoc-AI
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 5. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 6. Download the dataset

Download `arxiv-metadata-oai-snapshot.json` from Kaggle :

👉 [arXiv Dataset — Cornell University](https://www.kaggle.com/datasets/Cornell-University/arxiv)

> ⚠️ This file is ~3GB and is NOT included in the repository.

### 7. Configure API key

Create a `secrets.toml` file in the project root :

```secrets
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

Get your free key at 👉 `console.groq.com`

### 8. Run the notebook

Open and run all cells in order :

```bash
jupyter notebook smartdoc.ipynb
```

This will generate all trained models in `models/` and plots in `reports/`.

> ⚠️ Run the notebook BEFORE launching the app — the app needs the saved models.

### 9. Launch the chatbot

```bash
streamlit run app.py
```

---
## 🧠 ML Pipeline

```
arXiv JSON Dataset (1.7M papers)
         ↓
    Sample 5,000 papers
         ↓
┌─────────────────────────────────────────────┐
│  Section 1 — Data Loading                   │
│  Section 2 — EDA (5 visualizations)         │
│  Section 3 — Text Preprocessing (7 steps)   │
│  Section 4 — Feature Engineering (22 feats) │
│  Section 5 — Model Comparison (4 models)    │
│  Section 6 — NER (spaCy)                    │
│  Section 7 — Summarization + RAG (FAISS)    │
│  Section 8 — Drafts & Experimentations      │
└─────────────────────────────────────────────┘
         ↓
   Streamlit Chatbot (app.py)
         ↓
   Groq LLM (llama-3.3-70b-versatile)
```
---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

##🙋‍♂️  Author
Built by Islem ben ameur.
---
