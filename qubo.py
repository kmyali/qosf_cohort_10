import numpy as np

# def consts and weights
num_bins = 3
num_items = 4
max_weight = 10 # aka bin capacity
weights = [3,4,5,6]
scaled_weights = [w/max_weight for w in weights]

penalty_1 = 2
penalty_2 = 3

diagonal = [1.0] * num_bins + [penalty_1] * (num_bins * num_items)

# create Q for weights [3,4,5,6]
Q_matrix = np.diag(diagonal)

Q_matrix[3][4] = 2*penalty_1
Q_matrix[3][5] = 2*penalty_1
Q_matrix[3][6] = 0.12*penalty_2
Q_matrix[3][9] = 0.15*penalty_2
Q_matrix[3][12] = 0.18*penalty_2


Q_matrix[4][5] = 2*penalty_1
Q_matrix[4][7] = 0.12*penalty_2
Q_matrix[4][10] = 0.15*penalty_2
Q_matrix[4][13] = 0.18*penalty_2

Q_matrix[5][8] = 0.12*penalty_2
Q_matrix[5][11] = 0.15*penalty_2
Q_matrix[5][14] = 0.18*penalty_2

Q_matrix[6][7] = 2*penalty_1
Q_matrix[6][8] = 2*penalty_1
Q_matrix[6][9] = 0.2*penalty_2
Q_matrix[6][12] = 0.24*penalty_2

Q_matrix[7][8] = 2*penalty_1
Q_matrix[7][10] = 0.2*penalty_2
Q_matrix[7][13] = 0.24*penalty_2

Q_matrix[8][11] = 0.2*penalty_2
Q_matrix[8][14] = 0.24*penalty_2


Q_matrix[9][10] = 2*penalty_1
Q_matrix[9][11] = 2*penalty_1
Q_matrix[9][12] = 0.3*penalty_2

Q_matrix[10][11] = 2*penalty_1
Q_matrix[10][13] = 0.3*penalty_2

Q_matrix[11][14] = 0.3*penalty_2

Q_matrix[12][13] = 2*penalty_1
Q_matrix[12][14] = 2*penalty_1

Q_matrix[13][14] = 2*penalty_1

Q_matrix2 = np.diag(diagonal)

# constraint 1
for j in range(num_bins):
    for i in range(num_items-1):
        
        for k in range(i+1, num_items):
            # print(f'{i}{j} {k}{j} {scaled_weights[i]*scaled_weights[k]}')
            Q_matrix2[(i*num_bins+j)+num_bins][(k*num_bins+j)+num_bins] = scaled_weights[i]*scaled_weights[k]*penalty_2
            # print()

# constraint 2
for i in range(num_items):
    for j in range(num_bins):

        for k in range(j+1, num_bins):
            # print(f'{i}{j} {i}{k}')
            Q_matrix2[(i*num_bins+j)+num_bins][(i*num_bins+k)+num_bins] = 2*penalty_1

print(np.array_equal(Q_matrix, Q_matrix2))
# exit()
# from ILP solvers
x_opt = np.array([
    0, 1, 1, 
    0, 0, 1, 
    0, 1, 0, 
    0, 0, 1, 
    0, 1, 0])
min_cost = (x_opt.T @ Q_matrix @ x_opt)

    
def is_valid(x):
    # Use at least one bin.
    if not any(x[0:num_bins]):
        return False
    
    # each item must be in 1 bin. 
    for i in range(num_items):
        if not sum(x[num_bins+i*num_bins:num_bins+(i+1)*num_bins]) == 1:
            return False
    
    # ensure no items in unused bins
    for i in range(num_bins):
        if not x[i]:
            for j in range(num_bins, num_bins+num_bins*num_items, num_bins):
                if x[j+i]:
                    return False
    return True


for i in range(2**num_bins+num_bins*num_items):
    binary_str = np.binary_repr(i, width=num_bins+num_bins*num_items)
    x_arr = [int(b) for b in binary_str]
    x = np.asarray(x_arr)
    if is_valid(x):
        cost = (x.T @ Q_matrix @ x)
        if cost <= min_cost:
            print(f'cost {cost} for matrix {x} is less than min_cost {min_cost}')
