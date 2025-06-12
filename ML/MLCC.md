
# 📊 Linear Regression & Loss Functions

## 📘 What is Linear Regression?
Linear regression is a statistical method that models the relationship between a dependent variable and one or more independent variables.

---

## ⚠️ What is Loss?
Loss measures how wrong a model's predictions are by comparing them with actual labels.

### 🔢 Common Loss Functions

| Loss Type | Description | Formula |
|-----------|-------------|---------|
| **L1 Loss** | Sum of absolute differences | ![L1](https://latex.codecogs.com/svg.image?\sum_{i=1}^{N}\ |y_i\ -\ \hat{y}_i|) |
| **MAE** | Mean of L1 Loss | ![MAE](https://latex.codecogs.com/svg.image?\frac{1}{N}\ \sum_{i=1}^{N}\ |y_i\ -\ \hat{y}_i|) |
| **L2 Loss** | Sum of squared differences | ![L2](https://latex.codecogs.com/svg.image?\sum_{i=1}^{N}(y_i\ -\ \hat{y}_i)^2) |
| **MSE** | Mean of L2 Loss | ![MSE](https://latex.codecogs.com/svg.image?\frac{1}{N}\ \sum_{i=1}^{N}(y_i\ -\ \hat{y}_i)^2) |

> 🎯 MAE is less sensitive to outliers; MSE penalizes large errors more.

---

## 🎯 Outliers
Outliers are values that differ significantly from others. They can heavily influence MSE due to squaring of errors.

---

## 🚀 Gradient Descent
An iterative optimization algorithm that updates weights to minimize the loss:

```math
w = w - η * ∂L/∂w  
b = b - η * ∂L/∂b
```

- Moves in the direction of the **negative gradient**.
- Continues until loss converges (reaches a minimum).

---

## ⚙️ Hyperparameters vs Parameters

| Term | Meaning |
|------|---------|
| **Parameters** | Learned during training (weights, bias). |
| **Hyperparameters** | Set before training (learning rate, batch size, epochs). |

---

## ⚡ Learning Rate

- Controls **step size** in gradient descent.
- If too high → model diverges.  
- If too low → model converges slowly.  
- Ideal value helps converge efficiently.

> ℹ️ Example: Gradient = 2.5, LR = 0.01 → Step size = 0.025

---

## 📦 Batch Size

- Number of examples used before updating weights.

| Method | Description |
|--------|-------------|
| **Full Batch** | Uses all data once per epoch. |
| **SGD** | Uses 1 random example → noisy but fast updates. |
| **Mini-Batch** | Uses a small subset (e.g. 32, 64, 128) → balance speed & stability.

> 🧠 Larger batches reduce noise; smaller batches allow faster iteration but may be noisier.

---

## 🔁 Epochs

- 1 Epoch = model sees the **entire training dataset once**.
- Training usually requires **multiple epochs**.
- More epochs = better training (to a point), but more time.

> 🔄 For 1,000 examples, batch size 100 → 10 steps per epoch.

---

## 📌 Summary

- **Loss**: Measures prediction error (MAE/MSE).
- **Gradient Descent**: Optimizes model by minimizing loss.
- **Learning Rate**: Controls update size.
- **Batch Size**: Number of examples per training step.
- **Epochs**: Full passes over the dataset.

---

🧠 *Mastering these basics is crucial for ML model tuning and interview success!*
