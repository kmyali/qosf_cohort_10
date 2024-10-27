from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer,  GurobiOptimizer
import numpy as np

# def consts and weights
num_bins = 4  # num of bins
num_items = 6 # num of items
B = 10 # weight of bin
weights = np.random.randint(1, B/2, num_items)
print('weights', weights)

# define all vars
qp = QuadraticProgram()
for j in range(num_bins):
    qp.binary_var(f"y{j}")

for i in range(num_items):
    for j in range(num_bins):
        qp.binary_var(f"x{i}{j}")

# cost function
to_be_minimized = {}
for j in range(num_bins):
    to_be_minimized[f'y{j}'] = 1
qp.minimize(linear=to_be_minimized)

# weight constraint
for j in range(num_bins):
    weight_constraint = {}
    for i in range(num_items):
        weight_constraint[(f"x{i}{j}", f'y{j}')] = weights[i].item() 
    qp.quadratic_constraint(quadratic=weight_constraint, sense="<=", rhs=10)

# use all items constraint
for i in range(num_items):
    use_all_items_constraint = {}
    for j in range(num_bins):
        use_all_items_constraint[(f"x{i}{j}", f'y{j}')] = 1
    qp.quadratic_constraint(quadratic=use_all_items_constraint, sense="=", rhs=1)

results = {}
# results['cplex'] = CplexOptimizer().solve(qp) # "Error: Model has non-convex quadratic constraint, index=3." "CPLEX cannot solve the model."
results['gurobi'] = GurobiOptimizer().solve(qp)

for name, result in results.items():
    print(name)
    for j in range(num_bins):
        print(f'y{j}', round(result.variables_dict[f'y{j}']))

    for i in range(num_items):
        for j in range(num_bins):
            print(round(result.variables_dict[f"x{i}{j}"]), end=' ')
        print()
    print()