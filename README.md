# 🚀 AI-ML-DL-GenAI Code Reference Repository

> The Ultimate AI Engineer Handbook for Machine Learning, Deep Learning, NLP, Generative AI, MLOps, and Interview Preparation.

---

# 📚 Table of Contents

* [1. Machine Learning](#1-machine-learning)
* [2. Supervised Learning](#2-supervised-learning)
* [3. Unsupervised Learning](#3-unsupervised-learning)
* [4. Hyperparameter Tuning](#4-hyperparameter-tuning)
* [5. ML Pipelines](#5-ml-pipelines)
* [6. Model Persistence](#6-model-persistence)
* [7. Deep Learning](#7-deep-learning)
* [8. Natural Language Processing](#8-natural-language-processing)
* [9. PyTorch](#9-pytorch)
* [10. Generative AI](#10-generative-ai)
* [11. MLOps](#11-mlops)
* [12. Project Structures](#12-project-structures)
* [13. Interview Quick Revision](#13-interview-quick-revision)

---

# 📦 Installation

```bash
pip install numpy pandas matplotlib seaborn plotly scipy scikit-learn xgboost lightgbm catboost optuna hyperopt

pip install tensorflow torch torchvision torchaudio

pip install nltk spacy gensim textblob sentence-transformers

pip install langchain langchain-community langchain-groq chromadb faiss-cpu

pip install openai google-generativeai anthropic

pip install mlflow dvc fastapi uvicorn streamlit

pip install crewai langgraph pyautogen
```

---

# 1️⃣ Machine Learning

## Basic Libraries

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import scipy
import os
import warnings

warnings.filterwarnings("ignore")
```

### Purpose

| Library    | Purpose                   |
| ---------- | ------------------------- |
| NumPy      | Numerical Computing       |
| Pandas     | Data Analysis             |
| Matplotlib | Visualization             |
| Seaborn    | Statistical Visualization |
| Plotly     | Interactive Visualization |
| SciPy      | Scientific Computing      |

---

## Data Loading

### CSV

```python
df = pd.read_csv("data.csv")
```

### Excel

```python
df = pd.read_excel("data.xlsx")
```

### JSON

```python
df = pd.read_json("data.json")
```

### SQL

```python
from sqlalchemy import create_engine

engine = create_engine("sqlite:///db.sqlite")
df = pd.read_sql("SELECT * FROM table_name", engine)
```

### Parquet

```python
df = pd.read_parquet("data.parquet")
```

### Pickle

```python
df = pd.read_pickle("data.pkl")
```

---

## Exploratory Data Analysis

```python
df.shape
df.info()
df.describe()

df.isnull().sum()

df.duplicated().sum()

df.nunique()

df.corr()
```

### Heatmap

```python
sns.heatmap(df.corr(), annot=True)
plt.show()
```

### Pairplot

```python
sns.pairplot(df)
plt.show()
```

### Distribution

```python
sns.histplot(df["column"], kde=True)
```

### Outlier Detection

```python
sns.boxplot(x=df["column"])
```

---

## Data Cleaning

### Missing Values

```python
df.fillna(df.mean(numeric_only=True), inplace=True)
```

### Remove Duplicates

```python
df.drop_duplicates(inplace=True)
```

### Outlier Removal (IQR)

```python
Q1 = df["col"].quantile(0.25)
Q3 = df["col"].quantile(0.75)

IQR = Q3 - Q1

df = df[
(df["col"] >= Q1 - 1.5*IQR) &
(df["col"] <= Q3 + 1.5*IQR)
]
```

### Type Conversion

```python
df["age"] = df["age"].astype(int)
```

---

## Data Preprocessing

### Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
```

```python
from sklearn.preprocessing import OneHotEncoder
```

```python
from sklearn.preprocessing import OrdinalEncoder
```

### Scaling

```python
from sklearn.preprocessing import (
StandardScaler,
MinMaxScaler,
RobustScaler,
Normalizer,
PowerTransformer,
QuantileTransformer
)
```

### Train Test Split

```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)
```

---

## Feature Engineering

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures
```

### SelectKBest

```python
from sklearn.feature_selection import SelectKBest
```

### Chi-Square

```python
from sklearn.feature_selection import chi2
```

### Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif
```

### Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold
```

### PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

### LDA

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

---

# 2️⃣ Supervised Learning

## Regression Algorithms

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)
```

### Ridge

```python
from sklearn.linear_model import Ridge
```

### Lasso

```python
from sklearn.linear_model import Lasso
```

### ElasticNet

```python
from sklearn.linear_model import ElasticNet
```

### Tree-Based Models

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
RandomForestRegressor,
ExtraTreesRegressor,
GradientBoostingRegressor,
AdaBoostRegressor
)
```

### XGBoost

```python
from xgboost import XGBRegressor
```

### LightGBM

```python
from lightgbm import LGBMRegressor
```

### CatBoost

```python
from catboost import CatBoostRegressor
```

### SVR

```python
from sklearn.svm import SVR
```

### KNN

```python
from sklearn.neighbors import KNeighborsRegressor
```

---

## Regression Metrics

```python
from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score
)

mae = mean_absolute_error(y_test,pred)

mse = mean_squared_error(y_test,pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test,pred)
```

### Formulas

| Metric | Formula   |     |       |
| ------ | --------- | --- | ----- |
| MAE    | Σ         | y−ŷ | /n    |
| MSE    | Σ(y−ŷ)²/n |     |       |
| RMSE   | √MSE      |     |       |
| R²     | 1−SSR/SST |     |       |
| MAPE   | Σ(        | y−ŷ | /y)/n |

---

## Classification Algorithms

```python
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
RandomForestClassifier,
ExtraTreesClassifier,
GradientBoostingClassifier,
AdaBoostClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
```

---

## Classification Metrics

```python
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
confusion_matrix,
classification_report
)
```

---

# 3️⃣ Unsupervised Learning

## Clustering

```python
from sklearn.cluster import (
KMeans,
MiniBatchKMeans,
DBSCAN,
AgglomerativeClustering,
MeanShift
)
```

### GMM

```python
from sklearn.mixture import GaussianMixture
```

---

## Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
```

---

## Anomaly Detection

```python
from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from sklearn.neighbors import LocalOutlierFactor
```

---

# 4️⃣ Hyperparameter Tuning

## GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
```

## RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
```

## Optuna

```python
import optuna
```

## Hyperopt

```python
from hyperopt import hp, tpe, fmin
```

---

# 5️⃣ ML Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```

Example:

```python
pipeline = Pipeline([
("scaler", StandardScaler()),
("model", RandomForestClassifier())
])
```

---

# 6️⃣ Model Persistence

## Pickle

```python
import pickle

pickle.dump(model, open("model.pkl","wb"))
```

## Joblib

```python
import joblib

joblib.dump(model,"model.joblib")
```

---

# 7️⃣ Deep Learning

## TensorFlow Imports

```python
import tensorflow as tf
```

## Keras Imports

```python
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
Dense,
Dropout,
BatchNormalization,
Embedding,
Conv2D,
MaxPooling2D,
Flatten,
LSTM,
GRU
)

from tensorflow.keras.callbacks import (
EarlyStopping,
ModelCheckpoint,
ReduceLROnPlateau
)
```

---

## ANN Template

```python
model = Sequential([
Dense(128,activation="relu"),
Dropout(0.3),
Dense(64,activation="relu"),
Dense(1)
])

model.compile(
optimizer="adam",
loss="mse",
metrics=["mae"]
)

model.fit(X_train,y_train,epochs=20)

model.evaluate(X_test,y_test)

pred = model.predict(X_test)
```

---

## CNN Template

```python
model = Sequential([
Conv2D(32,(3,3),activation="relu"),
MaxPooling2D(),
Flatten(),
Dense(128,activation="relu"),
Dense(10,activation="softmax")
])
```

---

## Transfer Learning

```python
from tensorflow.keras.applications import (
VGG16,
ResNet50,
EfficientNetB0
)
```

---

## RNN

```python
SimpleRNN
LSTM
GRU
```

---

# 8️⃣ Natural Language Processing

## Libraries

```python
import nltk
import spacy
from textblob import TextBlob
from gensim.models import Word2Vec
```

---

## Text Preprocessing

```python
text = text.lower()
```

```python
from nltk.corpus import stopwords
```

```python
from nltk.stem import PorterStemmer
```

```python
from nltk.stem import WordNetLemmatizer
```

---

## Vectorization

### CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer
```

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
```

---

# 9️⃣ PyTorch

```python
import torch
import torch.nn as nn
```

## Dataset

```python
from torch.utils.data import Dataset, DataLoader
```

## Custom Dataset

```python
class MyDataset(Dataset):
    pass
```

## Training Loop

```python
for epoch in range(epochs):
    for x,y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x),y)
        loss.backward()
        optimizer.step()
```

---

# 🔟 Generative AI

## OpenAI

```python
from openai import OpenAI
```

## Groq

```python
from groq import Groq
```

## Gemini

```python
import google.generativeai as genai
```

## Claude

```python
from anthropic import Anthropic
```

## HuggingFace

```python
from transformers import AutoModelForCausalLM
```

---

## Prompt Engineering

### Zero Shot

```text
Answer the following question:
```

### Few Shot

```text
Example 1...
Example 2...
Now solve:
```

### Chain of Thought

```text
Think step by step.
```

### Role Based

```text
You are a Senior ML Engineer.
```

### Structured Output

```text
Return JSON format only.
```

---

## Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

```python
from langchain_openai import OpenAIEmbeddings
```

---

## Vector Databases

### ChromaDB

```python
import chromadb
```

### FAISS

```python
import faiss
```

### Pinecone

```python
from pinecone import Pinecone
```

### Weaviate

```python
import weaviate
```

---

## LangChain

```python
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
```

---

## RAG Starter

```python
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_groq import ChatGroq
```

Workflow:

```text
PDF
 ↓
Loader
 ↓
Splitter
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Retriever
 ↓
Groq Llama
 ↓
Answer
```

---

## AI Agents

### LangGraph

```python
from langgraph.graph import StateGraph
```

### CrewAI

```python
from crewai import Agent, Task, Crew
```

### AutoGen

```python
import autogen
```

---

# 1️⃣1️⃣ MLOps

## MLflow

```python
import mlflow
```

## DVC

```bash
dvc init
```

## FastAPI

```python
from fastapi import FastAPI
```

## Docker

```dockerfile
FROM python:3.11
```

## Streamlit

```python
import streamlit as st
```

---

# 1️⃣2️⃣ Project Structures

## ML Project

```text
ML_Project/
│
├── data/
├── notebooks/
├── src/
├── models/
├── configs/
├── logs/
└── app.py
```

## Deep Learning

```text
DL_Project/
├── dataset/
├── models/
├── training/
├── inference/
└── app.py
```

## NLP

```text
NLP_Project/
├── corpus/
├── preprocessing/
├── embeddings/
└── training/
```

## Generative AI

```text
GenAI_Project/
├── prompts/
├── vector_db/
├── llm/
├── chains/
└── app.py
```

## Multi Document RAG

```text
RAG_Project/
├── data/
├── embeddings/
├── chroma_db/
├── loaders/
├── chains/
├── app.py
└── requirements.txt
```

## AI Agent

```text
Agent_Project/
├── agents/
├── tools/
├── memory/
├── workflows/
└── main.py
```

---

# 1️⃣3️⃣ Interview Quick Revision

## Regression

| Algorithm         | Use Case              | Pros          | Cons          |
| ----------------- | --------------------- | ------------- | ------------- |
| Linear Regression | Continuous Prediction | Simple        | Linear Only   |
| Random Forest     | General Purpose       | Powerful      | Slow          |
| XGBoost           | Competitions          | High Accuracy | Tuning Needed |

---

## Classification

| Algorithm           | Use Case   | Pros     | Cons         |
| ------------------- | ---------- | -------- | ------------ |
| Logistic Regression | Binary     | Fast     | Linear       |
| Random Forest       | General    | Robust   | Large Models |
| XGBoost             | Production | Accurate | Complex      |

---

## Clustering

| Algorithm | Use Case        | Pros        | Cons              |
| --------- | --------------- | ----------- | ----------------- |
| KMeans    | Segmentation    | Fast        | Sensitive to K    |
| DBSCAN    | Noise Detection | No K Needed | Density Sensitive |

---

## Evaluation Metrics

| Metric   | Formula       | Use             |
| -------- | ------------- | --------------- |
| MAE      | Avg Error     | Regression      |
| RMSE     | sqrt(MSE)     | Regression      |
| Accuracy | Correct/Total | Classification  |
| F1       | Harmonic Mean | Imbalanced Data |

---

## Deep Learning

| ANN     | CNN    | RNN      | LSTM         | GRU         | Transformer |
| ------- | ------ | -------- | ------------ | ----------- | ----------- |
| Tabular | Images | Sequence | Long Context | Faster LSTM | LLMs        |

---

## Generative AI Components

| Component  | Purpose                 |
| ---------- | ----------------------- |
| Embeddings | Semantic Representation |
| Vector DB  | Similarity Search       |
| Retriever  | Fetch Context           |
| LLM        | Generate Answer         |
| Agent      | Multi-Step Reasoning    |

---

# ⭐ Best Practices

* Always perform EDA before modeling.
* Use Pipelines in production.
* Track experiments using MLflow.
* Use Feature Engineering carefully.
* Validate with Cross Validation.
* Prevent Data Leakage.
* Monitor Drift in Production.
* Use RAG to reduce hallucinations.
* Cache embeddings for performance.
* Containerize deployments with Docker.
* Automate CI/CD pipelines.
* Maintain reproducible environments.

---

# 🎯 Goal

This repository acts as a complete AI Engineer Handbook covering:

✅ Machine Learning
✅ Deep Learning
✅ NLP
✅ PyTorch
✅ Generative AI
✅ RAG
✅ AI Agents
✅ MLOps
✅ Interview Preparation

Happy Learning & Building 🚀
