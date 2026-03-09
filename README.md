# AI System for Detecting Misinformation

## Overview
This project builds a machine learning system that detects misinformation in political statements. The model is trained on the LIAR dataset and classifies claims as **REAL** or **FAKE**.

The goal of this project is to explore how natural language processing (NLP) techniques can be used to automatically identify misleading or false statements.

---

## Dataset
The model is trained on the **LIAR Dataset**, which contains short political statements labeled with different truthfulness levels.

Original labels include:

- pants-fire  
- false  
- barely-true  
- half-true  
- mostly-true  
- true  

For this project, the labels were simplified into a binary classification:

| Original Label | Mapped Label |
|----------------|-------------|
| pants-fire | FAKE |
| false | FAKE |
| barely-true | FAKE |
| half-true | REAL |
| mostly-true | REAL |
| true | REAL |

---

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- TF-IDF Vectorization  
- Logistic Regression  

Future improvements will include transformer models such as BERT.

---

## Project Structure

```
misinformation-detector

data/
    train.tsv
    valid.tsv
    test.tsv
    clean_data.csv

src/
    preprocess.py
    train.py

models/
    model.pkl
    vectorizer.pkl

requirements.txt
README.md
```

---

## Data Preprocessing

The preprocessing pipeline performs:

- Lowercasing text  
- Removing URLs  
- Removing special characters  
- Mapping truthfulness labels to binary classes  

The cleaned dataset is saved as:

```
data/clean_data.csv
```

---

## Model

Baseline model:

- **Feature extraction:** TF-IDF vectorization  
- **Classifier:** Logistic Regression  

---

## Results

Baseline model performance:

```
Accuracy: ~0.61
```

The LIAR dataset is challenging because statements are short and context is limited.

---

## How to Run the Project

### 1 Install dependencies

```
pip install -r requirements.txt
```

### 2 Preprocess the dataset

```
python src/preprocess.py
```

### 3 Train the model

```
python src/train.py
```

---

## Future Improvements

- Fine-tune BERT for misinformation detection  
- Add a prediction script for testing new claims  
- Build a REST API for real-time predictions  
- Create a web interface for interactive use  

---

## Author

Machine learning project exploring NLP techniques for misinformation detection.
