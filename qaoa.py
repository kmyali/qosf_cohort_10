import pennylane.numpy as np
import matplotlib as plt
import numpy as npp
import math
import pennylane as qml
from pennylane import AdamOptimizer
shots = 500  # Number of samples used
dev = qml.device("default.qubit", shots=shots)

npp.set_printoptions(linewidth=150)

# def consts and weights
num_bins = 3
num_items = 4
matrix_len = num_bins+num_bins*num_items
max_weight = 10 # aka bin capacity
weights = [3,4,5,6]
# scaled_weights = [w/max_weight for w in weights]
# print(scaled_weights)
scaled_weights = weights

penalty_1 = 5
penalty_2 = 5
penalty_3 = 5

diagonal = [-1/2] * num_bins + [-penalty_1/2] * (num_bins * num_items)
Q_matrix = npp.diag(diagonal)

# The change of variable from x to z means adds extra constants. These constants ideally shouldnt impact the minimization.
H = 0
for i in range(len(diagonal)-num_bins):
    H +=  1/2*penalty_1 # from constraint 2

for _ in range(num_bins):
    H += 1/2 # from constraint 2

# constraint 1: weight constraint
for j in range(num_bins):
    for i in range(num_items-1):
        
        for k in range(i+1, num_items):
            row = (i*num_bins+j)+num_bins
            col = (k*num_bins+j)+num_bins
            value = scaled_weights[i]*scaled_weights[k]*penalty_2*1/4
            Q_matrix[row][col] = value
            
            Q_matrix[row][row] -= value
            Q_matrix[col][col] -= value
            H += value

# constraint 2: use all items once constraint
for i in range(num_items):
    for j in range(num_bins):

        for k in range(j+1, num_bins):
            row = (i*num_bins+j)+num_bins
            col = (i*num_bins+k)+num_bins
            Q_matrix[row][col] = 2*penalty_1*1/4

            Q_matrix[row][row] -= 1/2*penalty_1
            Q_matrix[col][col] -= 1/2*penalty_1
            H += 1/2*penalty_1 

# construct the Hamiltonian    
n_qubits = Q_matrix.shape[0]
for row in range(n_qubits):
    for col in range(row, n_qubits):
        if row == col:
            H += Q_matrix[row][row] * qml.PauliZ(row)
        elif Q_matrix[row][col] != 0:
            H += Q_matrix[row][col] * qml.PauliZ(row) @ qml.PauliZ(col)
print(H)

def qaoa_circuit(params):
    for param, wire in zip(params, H.wires):
        qml.RY(param, wires=wire)

@qml.qnode(dev)
def qaoa_expvalue(params):
    qaoa_circuit(params)
    return qml.expval(H)

cost_function = lambda params: qaoa_expvalue(params)

parameters = np.array([0.5 for i in range(len(H.wires))], requires_grad=True)

optimizer = AdamOptimizer()
cost_list = []
n_steps = 200

# optimize
for i in range(n_steps):
    parameters = optimizer.step(cost_function, parameters)
    cost_list.append(cost_function(parameters))
    print(cost_function(parameters))
print(parameters)

# Get the result from the ideal parameters
@qml.qnode(dev)
def get_x(params):
    for param, wire in zip(params, H.wires):
        qml.RY(param, wires=wire)
    return [qml.expval(qml.PauliZ(wire)) for wire in H.wires]
output = get_x(parameters)
print("Circuit output:", output)
