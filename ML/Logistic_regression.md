# Logistic Regression – Google Machine Learning Crash Course Notes

## 🔍 What is Logistic Regression?

* A classification algorithm used to predict **binary** outcomes (e.g., spam/not spam, disease/no disease).
* Despite the name, it's used for **classification**, not regression.

---

## 📈 Core Concept

* Logistic regression predicts the **probability** that a given input belongs to class **1** (vs. class 0).
* Formula:

  ```
  z = w1*x1 + w2*x2 + ... + wn*xn + b
  ˆy = sigmoid(z) = 1 / (1 + e^(-z))
  ```

  where:

  * `sigmoid(z)` is the **sigmoid function**
  * `ˆy` is the predicted probability

---

## 🔄 Sigmoid Function

* Converts any real-valued number into the (0,1) interval.
* Graph is **S-shaped** (asymptotes at 0 and 1).

---

## ⚖️ Decision Boundary

* Predict 1 if `ˆy > 0.5`, else predict 0.
* Decision threshold can be adjusted for **precision-recall trade-off**.

---

## 💡 Loss Function: Log Loss (Cross-Entropy Loss)

For binary classification:

```
Loss = -[y*log(ˆy) + (1 - y)*log(1 - ˆy)]
```

* Penalizes incorrect confident predictions heavily.

---

## 🎯 Goal

* Minimize the **average log loss** across all training examples using gradient descent.

---

## ⚙️ Training via Gradient Descent

* Use gradients of the loss function w\.r.t weights to update them:

  ```
  wi := wi - η * ∂Loss/∂wi
  ```

  where `η` is the **learning rate**

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Curve

---

## 🧠 Logistic vs Linear Regression

| Feature             | Logistic Regression      | Linear Regression        |
| ------------------- | ------------------------ | ------------------------ |
| Output              | Probability (0 to 1)     | Continuous real value    |
| Target              | Categorical (binary)     | Continuous               |
| Activation Function | Sigmoid                  | None                     |
| Loss Function       | Cross-entropy (log loss) | Mean Squared Error (MSE) |

---

## ✅ When to Use Logistic Regression

* Binary classification tasks
* When you need probability estimates
* Fast and interpretable baseline model
