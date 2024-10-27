from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer,  GurobiOptimizer
import numpy as np

num_bins = 5 # num of bins
num_items = 6 # num of items
B = 10 # weight of bin

weights = np.random.randint(1, B/2, num_items)
print(weights)

# define a problem
qp = QuadraticProgram()
for j in range(num_bins):
    qp.binary_var(f"y{j}")

for i in range(num_items):
    for j in range(num_bins):
        qp.binary_var(f"x{i}{j}")

to_be_minimized = {}
for j in range(num_bins):
    to_be_minimized[f'y{j}'] = 1
print('to_be_minimized', to_be_minimized)
qp.minimize(linear=to_be_minimized)

# weight constraint
for j in range(num_bins):
    weight_constraint = {}
    for i in range(num_items):
        weight_constraint[f"x{i}{j}"] = weights[i].item() # need to link x with y in a constraint?
    print('weight constraint', weight_constraint)
    qp.linear_constraint(weight_constraint, "<=", 10)

# one item per bin constraint
for i in range(num_items):
    item_per_bin_constraint = {}
    for j in range(num_bins):
        item_per_bin_constraint[f"x{i}{j}"] = 1
    print('item in one bin constraint', item_per_bin_constraint)
    qp.linear_constraint(item_per_bin_constraint, "=", 1)

# use at least one bin constraint
min_one_bin_constraint = {}
for j in range(num_bins):
    min_one_bin_constraint[f'y{j}'] = 1
qp.linear_constraint(min_one_bin_constraint, ">=", 1)

cplex_result = CplexOptimizer().solve(qp)
gurobi_result = GurobiOptimizer().solve(qp)

print("cplex")
for j in range(num_bins):
    print(f'y{j}', cplex_result.variables_dict[f'y{j}'])

for i in range(num_items):
    for j in range(num_bins):
        print(cplex_result.variables_dict[f"x{i}{j}"], end=' ')
    print()
print()
print("gurobi")
print(gurobi_result.prettyprint())