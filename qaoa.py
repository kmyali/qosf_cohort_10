import numpy as np
import math
import pennylane as qml
import pennylane.numpy as npp


shots = 500  # Number of samples used
dev = qml.device("default.qubit", shots=shots)

np.set_printoptions(linewidth=150)

# def consts and weights
num_bins = 3
num_items = 4
matrix_len = num_bins+num_bins*num_items
max_weight = 10 # aka bin capacity
weights = np.random.random(num_items)
weights = [3,4,5,6]
# scaled_weights = [w/max_weight for w in weights]
# print(scaled_weights)
scaled_weights = weights

penalty_1 = 2
penalty_2 = 2
penalty_3 = 5

diagonal = [1.0] * num_bins + [-1/2*penalty_1] * (num_bins * num_items)

Q_matrix = np.diag(diagonal)

H = 0
# for i in len(diagonal):
#     H +=  1/2*penalty_1 # constraint 2

# constraint 1: weight constraint
for j in range(num_bins):
    for i in range(num_items-1):
        
        for k in range(i+1, num_items):
            # should this be negative?
            Q_matrix[(i*num_bins+j)+num_bins][(k*num_bins+j)+num_bins] = scaled_weights[i]*scaled_weights[k]*penalty_2*1/4
            
            Q_matrix[(i*num_bins+j)+num_bins][(i*num_bins+j)+num_bins] -= 1/4*penalty_2
            Q_matrix[(k*num_bins+j)+num_bins][(k*num_bins+j)+num_bins] -= 1/4*penalty_2
            # H += 1/4*penalty_2*scaled_weights[i]*scaled_weights[k]

# constraint 2: use all items once constraint
for i in range(num_items):
    for j in range(num_bins):

        for k in range(j+1, num_bins):
            row = (i*num_bins+j)+num_bins
            col = (i*num_bins+k)+num_bins
            Q_matrix[row][col] = 2*penalty_1

            Q_matrix[row][row] -= 1/2*penalty_1
            Q_matrix[col][col] -= 1/2*penalty_1
            # H += 1/2*penalty_1 

n_qubits = Q_matrix.shape[0]
for row in range(n_qubits):
    for col in range(row, n_qubits):
        if row == col:
            H += Q_matrix[row][row] * qml.PauliZ(row)
        elif Q_matrix[row][col] != 0:
            H += Q_matrix[row][col] * qml.PauliZ(row) @ qml.PauliZ(col)
# print(H)

@qml.qnode(dev)
def circuit(params):
    for param, wire in zip(params, H.wires):
        qml.RY(param, wires=wire)
    return qml.sample()


params = np.array([5 for _ in range(len(H.wires))])
print(params)
opt = qml.AdamOptimizer(stepsize=0.5)
epochs = 200

for epoch in range(epochs):
    params = opt.step(circuit, params)
    print(params)
print(circuit(params))