import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# --------------------------------------------------
# 1) Dataset (linearly separable)
# --------------------------------------------------
X = np.array([
    [4, 4],
    [5, 3],
    [1, 1],
    [2, 0],
    [6,3],
    [0,0]
], dtype=float)

y = np.array([1, 1, -1, -1,-1,-1], dtype=float)
N = X.shape[0]

# --------------------------------------------------
# 2) Build QP matrices (Dual Problem)
# --------------------------------------------------
Q = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Q[i, j] = y[i] * y[j] * np.dot(X[i], X[j])

P = matrix(Q)
q = matrix(-np.ones(N))

G = matrix(-np.eye(N))     # α ≥ 0
h = matrix(np.zeros(N))

A = matrix(y.reshape(1, -1))
b = matrix(0.0)

# --------------------------------------------------
# 3) Solve QP → α
# --------------------------------------------------
solvers.options['show_progress'] = False
solution = solvers.qp(P, q, G, h, A, b)
alpha = np.array(solution['x']).flatten()

print("α values:")
print(alpha)

# --------------------------------------------------
# 4) Compute w from α (KKT)
# --------------------------------------------------
w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
print("\nw vector:")
print(w)

# --------------------------------------------------
# 5) Compute b using support vectors
# --------------------------------------------------
sv = alpha > 1e-5
b_vals = []

for i in range(N):
    if sv[i]:
        b_vals.append(y[i] - np.dot(w, X[i]))

b = np.mean(b_vals)
print("\nb value:")
print(b)

# --------------------------------------------------
# 6) Plot SVM (manual boundary)
# --------------------------------------------------
plt.figure(figsize=(6,6))

for i in range(N):
    if y[i] == 1:
        plt.scatter(X[i,0], X[i,1], c='blue', marker='o')
    else:
        plt.scatter(X[i,0], X[i,1], c='red', marker='x')

x_vals = np.linspace(0, 6, 200)
y_vals = -(w[0]*x_vals + b) / w[1]

plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')

# margins
y_margin1 = -(w[0]*x_vals + b - 1) / w[1]
y_margin2 = -(w[0]*x_vals + b + 1) / w[1]
plt.plot(x_vals, y_margin1, 'k--')
plt.plot(x_vals, y_margin2, 'k--')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.title("Hard-Margin SVM (Manual from Lagrangian)")
plt.show()