import numpy as np
np.set_printoptions(linewidth=150)
# def consts and weights
num_bins = 3
num_items = 4
matrix_len = num_bins+num_bins*num_items
max_weight = 10 # aka bin capacity
weights = [3,4,5,6]
scaled_weights = [w/max_weight for w in weights]

penalty_1 = 2
penalty_2 = 2

diagonal = [1.0] * num_bins + [penalty_1] * (num_bins * num_items)

Q_matrix = np.diag(diagonal)

# constraint 1
for j in range(num_bins):
    for i in range(num_items-1):
        
        for k in range(i+1, num_items):
            Q_matrix[(i*num_bins+j)+num_bins][(k*num_bins+j)+num_bins] = scaled_weights[i]*scaled_weights[k]*penalty_2

# constraint 2
for i in range(num_items):
    for j in range(num_bins):

        for k in range(j+1, num_bins):
            Q_matrix[(i*num_bins+j)+num_bins][(i*num_bins+k)+num_bins] = 2*penalty_1

    
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
            for j in range(num_bins, matrix_len, num_bins):
                if x[j+i]:
                    return False
    print('True')
    return True

min_cost = np.inf
for i in range(2**(matrix_len)):
    binary_str = np.binary_repr(i, width=matrix_len)
    x_arr = [int(b) for b in binary_str]
    x = np.asarray(x_arr)
    if is_valid(x):
        cost = (x.T @ Q_matrix @ x)
        if cost <= min_cost:
            min_cost = cost
            min_matrix = x.copy()
            # print(f'cost {cost} for matrix {min_matrix} is less than min_cost {min_cost}')

print(f'min_cost is {min_cost} for matrix {min_matrix}')