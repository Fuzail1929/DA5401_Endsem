# DA5401_Endsem
- Name : Mohd Fuzail
- Roll no .CH22B080
- Date - 21-11-25
# 🧠 Metric–Response Fitness Prediction  
### Predicting LLM Judge Scores (0–10) from Metric & Text Embeddings

---

## 📌 Overview
This project builds a machine-learning system that predicts the *fitness score* (0–10) assigned by an LLM judge to a prompt–response pair, conditioned on a metric definition.

Each training example contains:

- **Metric name**
- **User prompt**
- **System prompt**
- **Model's generated response**
- **LLM judge score (0–10)**
- **Metric definition embedding (768 dimensions)**

The aim is to train a regression model that learns how well a given response satisfies a particular evaluation metric.

---

## 📂 Dataset

### Training Files
- train_data.json
- metric_names.json
- metric_name_embeddings.npy

### Test File


---

## 🧩 Data Processing Pipeline

### 1. Load Dataset
We load:
- Metric names  
- Metric embeddings  
- Training JSON (prompt + response + system prompt + score)  
- Map each `metric_name` → index  

### 2. Build Text Pair
text_pair = user_prompt + " " + response + " " + system_prompt

### 3. Text Embeddings
We use the multilingual SentenceTransformer:
googlegemma 

This produces a **512-dimensional embedding** per sample.

### 4. Similarity Features
We compute:
- **Cosine similarity** between metric + text embeddings  
- **Dot product**  
- **L2 distance**  

### 5. Additional Metadata Features
- Prompt length  
- Response length  
- Ratio of prompt/response lengths  

### 6. Final Feature Matrix
We concatenate:
768-dim metric embedding
512-dim text embedding
3 similarity features
3 metadata features
≈ 1542 total features

---

## 🔀 Cross-Validation (K-Fold)
We use:
KFold(n_splits=5, shuffle=True, random_state=42)
This ensures stable and robust RMSE estimation.

---

## 🌳 Model Training (XGBoost)

We use a generalization-optimized configuration:


params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.02,
    "max_depth": 7,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 2.0,
    "reg_alpha": 0.2,
    "gamma": 0.3,
    "tree_method": "hist",
    "random_state": 42
}

Training is done using:
5000 boosting rounds
Early stopping at 200 rounds
🧪 Inference on Test Set
Test data processing repeats the same steps as training:
Build text_pair
Compute text embedding
Extract metric embedding
Compute cosine, dot, L2
Compute metadata features
Build feature matrix
Predict using trained XGBoost model
Clip outputs to the valid range 0–10
