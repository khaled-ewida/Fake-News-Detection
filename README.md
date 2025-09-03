# 📰 Fake News Detection

## 📌 Overview
This project is a **Fake News Detection System** built with **Python**.  
It uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify news articles as either **FAKE** or **REAL**.  

The model is trained using **TF-IDF Vectorization** and a **Passive Aggressive Classifier** for efficient and fast classification.  
It also provides a **Confusion Matrix Heatmap** to visualize model performance.

---

## ⚙️ Features
- Preprocesses text data using **TF-IDF Vectorizer**.
- Classifies news articles into FAKE or REAL.
- Evaluates model accuracy.
- Generates a confusion matrix with heatmap visualization.

---

## 📂 Dataset
The project uses a dataset file named **`news.csv`**, which should be placed in the project root directory.  
It contains the following columns:
- **title** → The headline of the news article.  
- **text** → The body/content of the news article.  
- **label** → The class of the article (`FAKE` or `REAL`).  

---

## 🚀 Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/khaled-ewida/Fake-News-Detection.git
   cd Fake-News-Detection
