from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer,  GurobiOptimizer
import numpy as np

N = 10 # num of bins
I = 10 # num of items
B = 1 # weight of bin

weights = np.ones(I) * 0.49
weights = np.random.rand(I)
print(weights)

# define a problem
qp = QuadraticProgram()
for j in range(N):
    qp.binary_var(f"y{j}")

for i in range(I):
    for j in range(N):
        qp.binary_var(f"x{i}{j}")

for j in range(N):
    constraint = {}
    for i in range(I):
        constraint[f"x{i}{j}"] = weights[i].item()
    print('weight constraint', constraint)
    qp.linear_constraint(constraint, "<=", 1)

for i in range(I):
    constraint = {}
    for j in range(N):
        constraint[f"x{i}{j}"] = 1
    print('item in one bin constraint', constraint)
    qp.linear_constraint(constraint, "=", 1)

cplex_result = CplexOptimizer().solve(qp)
gurobi_result = GurobiOptimizer().solve(qp)

print("cplex")
print(cplex_result.prettyprint())
print()
print("gurobi")
print(gurobi_result.prettyprint())