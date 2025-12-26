# Support Vector Machine from Scratch (Lagrangian Dual)

This repository contains a **from-scratch implementation of a Hard-Margin Support Vector Machine (SVM)**,
derived step-by-step from the **primal optimization problem**, using:

- Lagrangian formulation
- Karush–Kuhn–Tucker (KKT) conditions
- Dual optimization
- Quadratic Programming (QP)

❗ No SVM libraries (e.g. `sklearn.svm`) are used.

---

## 📌 Motivation

Most SVM implementations rely on high-level libraries that hide the mathematical foundation.
This project explicitly demonstrates:

- How the primal problem is formulated
- How the Lagrangian is constructed
- How KKT conditions lead to the dual problem
- How the weight vector **w** and bias **b** are recovered from Lagrange multipliers **α**
- How the decision boundary and margins are manually plotted

This makes the project suitable for **educational and academic purposes**.

---

## 📐 Mathematical Formulation

### Primal Problem

\[
\min_{w,b} \frac{1}{2} \|w\|^2
\quad
\text{s.t. }
y_i (w^T x_i + b) \ge 1
\]

---

### Lagrangian

\[
\mathcal{L}(w,b,\alpha) =
\frac{1}{2} w^T w -
\sum_{i=1}^{N} \alpha_i \big(y_i (w^T x_i + b) - 1 \big)
\]

---

### KKT Conditions

- \( w = \sum_i \alpha_i y_i x_i \)
- \( \sum_i \alpha_i y_i = 0 \)
- \( \alpha_i \ge 0 \)

---

### Dual Problem (QP)

\[
\max_{\alpha}
\sum_i \alpha_i
-
\frac{1}{2}
\sum_i \sum_j
\alpha_i \alpha_j y_i y_j (x_i^T x_j)
\]

Subject to:

\[
\sum_i \alpha_i y_i = 0,
\quad
\alpha_i \ge 0
\]

---

## 🛠 Implementation Details

- The dual optimization problem is solved using **Quadratic Programming**
- Only the `cvxopt` library is used for QP solving
- The weight vector **w** and bias **b** are computed manually from the solution
- Decision boundary and margins are plotted explicitly

---
## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/hard_margin_svm.py
