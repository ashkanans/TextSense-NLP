# **Sentence Classification with LSTM - HW1B**

## **Project Overview**
This project is part of **HW1B** and focuses on **sentence classification** using **LSTM-based deep learning models**. The goal is to implement an LSTM model and compare its performance against **baseline models**. The dataset consists of **train, news test, and tweets test sets**, where the model is trained on the **train set** and tested on **news and tweets sets**.

### **Objectives**
- Train a **BiLSTM classifier** on a text classification task.
- Compare its performance against **Random, Majority, and Stratified Baselines**.
- Use **different LSTM configurations** to analyze performance variations.
- Evaluate on separate **news** and **tweets** test sets.

## **📂 Project Structure**
```
project_root/
│── hw1b_train.py          # Train one LSTM model & save it
│── hw1b_evaluate.py       # Load & evaluate a trained LSTM model
│── pipeline.py            # Run multiple LSTM experiments & compare baselines
│── src/                   # Code modules (model, data loader, training, evaluation, baselines)
│── data/                  # Dataset files (train, test-news, test-tweets)
│── experiments/           # Stores trained LSTM models & results from pipeline.py
│── results/               # Stores evaluation results of trained models
│── models/                # Stores manually trained models from hw1b_train.py
│── README.md              # Project documentation
```

## **📜 How to Use the Scripts**

### **1️⃣ Training a Single Model: `hw1b_train.py`**
📌 **Purpose:** Train a single LSTM model based on a predefined configuration and save it.

#### **Run Command:**
```bash
python hw1b_train.py
```
#### **What Happens?**
- Loads the dataset.
- Trains the LSTM model.
- Saves the trained model to `models/lstm_model.pth`.

#### **Saves:**
- **Model Weights:** `models/lstm_model.pth`

---

### **2️⃣ Evaluating a Trained Model: `hw1b_evaluate.py`**
📌 **Purpose:** Load a trained model, evaluate it on test sets, compute baselines, and save results.

#### **Run Command:**
```bash
python hw1b_evaluate.py
```
#### **What Happens?**
- Loads the trained model from `models/lstm_model.pth`.
- Evaluates it on **news and tweets test sets**.
- Computes **baseline performance** for comparison.
- Saves evaluation results to `results/evaluation_results.json`.

#### **Saves:**
- **Evaluation Results:** `results/evaluation_results.json`

---

### **3️⃣ Running Multiple Experiments: `pipeline.py`**
📌 **Purpose:** Train multiple LSTM models with different configurations and compare their performance with baselines.

#### **Run Command:**
```bash
python pipeline.py
```
#### **What Happens?**
- Evaluates **baseline models** first.
- Trains multiple LSTM models with different hyperparameters.
- Evaluates each model on **news and tweets test sets**.
- Saves results for each experiment in a separate timestamped folder inside `experiments/`.

#### **Saves:**
- **Trained Models:** `experiments/YYYY-MM-DD_HH-MM-SS/model.pth`
- **Evaluation Results:** `experiments/YYYY-MM-DD_HH-MM-SS/model_info.json`
- **Baseline Results:** Stored in `model_info.json` for comparison.

---

## **Understanding the `experiments/`, `results/`, and `models/` Folders**

### **📂 `experiments/`**
- **Created by `pipeline.py`**
- Stores results of different **LSTM model experiments**.
- Each experiment has a **timestamped folder** with:
  - `model.pth`: Trained model weights.
  - `model_info.json`: LSTM and baseline evaluation metrics.

### **📂 `results/`**
- **Created by `hw1b_evaluate.py`**
- Stores final evaluation results of **one manually trained LSTM model**.
- Includes baselines comparison for easy performance analysis.

### **📂 `models/`**
- **Created by `hw1b_train.py`**
- Stores the manually trained **single LSTM model**.
- Used by `hw1b_evaluate.py` for evaluation.

---
## **📌 Best LSTM Model Configuration**
The best-performing LSTM model was trained with the following **hyperparameters**:

```yaml
Hidden Dimension: 256
Number of Layers: 2
Bidirectional: True
Dropout: 0.3
Embedding Dimension: 100
Learning rate: 0.0005
Epochs: 15
Model Path: models/lstm_model.pth
```

---
### **📊 Performance Comparison - Test News Set**
| **Model**             | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|----------------------|------------|------------|------------|------------|
| **Random Baseline**  | 49.6%      | 49.32%     | 49.27%     | 48.47%     |
| **Majority Baseline** | 63.8%      | 31.9%      | 50.0%      | 38.95%     |
| **Stratified Baseline** | 49.0%   | 43.65%     | 43.90%     | 43.73%     |
| **Word2Vec Baseline** | 57.8%     | 54.76%     | 54.86%     | 54.79%     |
| **LSTM Model (Best Config)** | **66.4%** | **62.94%** | **61.72%** | **62.01%** |

---

### **📊 Performance Comparison - Test Tweets Set**
| **Model**             | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|----------------------|------------|------------|------------|------------|
| **Random Baseline**  | 50.12%     | 50.13%     | 50.13%     | 50.12%     |
| **Majority Baseline** | 50.75%     | 25.38%     | 50.0%      | 33.67%     |
| **Stratified Baseline** | 49.8%  | 49.78%     | 49.78%     | 49.77%     |
| **Word2Vec Baseline** | 55.66%     | 55.66%     | 55.66%     | 55.65%     |
| **LSTM Model (Best Config)** | **63.34%** | **64.18%** | **63.51%** | **62.96%** |

---

### **Observations**
- **Word2Vec Baseline performs significantly better than Random and Stratified Baselines**.
- **Word2Vec outperforms Majority Baseline by ~5-10% in accuracy and F1-score**.
- **LSTM Model still remains the best performing** but shows a **clear advantage over Word2Vec** in both test sets.
- **Word2Vec Baseline captures sentence meaning better than frequency-based baselines**.

---
