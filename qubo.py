import numpy as np

# def consts and weights
num_bins = 3
num_items = 4
max_weight = 10
# print('weights', weights)

penalty_1 = 5
penalty_2 = 5

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

Q_matrix[7][8] = 2*penalty_1
Q_matrix[7][10] = 0.2*penalty_2

Q_matrix[8][11] = 0.2*penalty_2

Q_matrix[9][10] = 2*penalty_1
Q_matrix[9][11] = 2*penalty_1
Q_matrix[9][12] = 0.3*penalty_2

Q_matrix[10][11] = 2*penalty_1
Q_matrix[10][13] = 0.3*penalty_2

Q_matrix[11][14] = 0.3*penalty_2

Q_matrix[12][13] = 2*penalty_1
Q_matrix[12][14] = 2*penalty_1

Q_matrix[13][14] = 2*penalty_1

# print(Q_matrix)
# exit()
x_opt = np.array([
    0, 1, 1, 
    0, 0, 1, 
    0, 1, 0, 
    0, 0, 1, 
    0, 1, 0])
min_cost = (x_opt.T @ Q_matrix @ x_opt)

    
def is_valid(x):
    basic = sum(x[3:6]) == 1 and sum(x[6:9]) == 1 and sum(x[9:12]) == 1 and sum(x[12:15]) == 1 and any(x[0:3])
    if not basic:
        return False
    
    for i in range(3):
        if not x[i]:
            for j in range(3,15,3):
                if x[j+i]:
                    return False
    return True


for i in range(2**15):
    binary_str = np.binary_repr(i, width=15)
    x_arr = [int(b) for b in binary_str]
    x = np.asarray(x_arr)
    if is_valid(x):
        cost = (x.T @ Q_matrix @ x)
        if cost <= min_cost:
            print(f'cost {cost} for matrix {x} is less than min_cost {min_cost}')


# min_cost = (x_opt.T @ Q_matrix @ x_opt)
# print(f"The minimum cost is  {min_cost}")