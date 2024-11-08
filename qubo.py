import numpy as np
np.set_printoptions(linewidth=150)
# def consts and weights
num_bins = 3
num_items = 4
matrix_len = num_bins+num_bins*num_items
max_weight = 10 # aka bin capacity
weights = np.random.random(num_items)
weights = [3,4,5,6] # comment this line to use random weights
# scaled_weights = [w/max_weight for w in weights]
# print(scaled_weights)
scaled_weights = weights

penalty_1 = 2
penalty_2 = 2
penalty_3 = 5

diagonal = [1.0] * num_bins + [penalty_1] * (num_bins * num_items)

Q_matrix = np.diag(diagonal)

# constraint 1: weight constraint
for j in range(num_bins):
    for i in range(num_items-1):
        
        for k in range(i+1, num_items):
            Q_matrix[(i*num_bins+j)+num_bins][(k*num_bins+j)+num_bins] = scaled_weights[i]*scaled_weights[k]*penalty_2

# constraint 2: use all items once constraint
for i in range(num_items):
    for j in range(num_bins):

        for k in range(j+1, num_bins):
            Q_matrix[(i*num_bins+j)+num_bins][(i*num_bins+k)+num_bins] = 2*penalty_1

# Note: adding or removing this constraint seems to have no effect on final answer.
# constraint 3: activate items in active bins
# for j in range(num_bins):
#     for i in range(num_items):
#         # Penalize each x_{ij} if y_j is 0 (inactive bin)
#         Q_matrix[j][num_bins + i * num_bins + j] = penalty_3


# Convert to Hermitian matrix if needed. Note: not needed because upper triangular is equivalent
# for r in range(Q_matrix.shape[0]):
#     for c in range(r+1, Q_matrix.shape[0]):
#         Q_matrix[r][c] = Q_matrix[r][c]/2
#         Q_matrix[c][r] = Q_matrix[r][c]


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
    return True

def brute_force():   
    min_cost = np.inf
    for i in range(2**(matrix_len)):
        binary_str = np.binary_repr(i, width=matrix_len)
        x_arr = [int(b) for b in binary_str]
        x = np.asarray(x_arr)
        if is_valid(x):
            cost = (x.T @ Q_matrix @ x)
            if cost <= min_cost:
                min_cost = cost
                min_x = x.copy()
                # print(f'cost {cost} for string {min_x} is less than min_cost {min_cost}')

    print(f'min_cost is {min_cost} for string {min_x}')


## Annealing ##

from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
import dimod

def annealing():
    """, then adding constraints to bqm."""
    penalty_1 = penalty_2 = 2
    penalty_3 = 2
    Q_matrix = np.diag(diagonal)

    # fill in qubo coefficients
    qubo = {(i, j): Q_matrix[i, j] for i in range(Q_matrix.shape[0]) for j in range(Q_matrix.shape[1])}
    bqm = dimod.BQM.from_qubo(qubo)
    bqm.relabel_variables({j: f'y{j}' for j in range(num_bins)})
    bqm.relabel_variables({i*num_bins+j+num_bins: f'x{i}{j}' for i in range(num_items) for j in range(num_bins)})
    
    # add constraints
    # weight constraint per bin
    for j in range(num_bins):
        bqm.add_linear_inequality_constraint(
            [(f"x{i}{j}", weights[i]*penalty_2) for i in range(num_items)],
            [7.29, 0.85], # from paper!
            "unbalanced",
            ub=max_weight*penalty_2,
            penalization_method="unbalanced",
        )
    
    # # use all items once constraint
    for i in range(num_items):
        bqm.add_linear_equality_constraint(
            [(f"x{i}{j}", 1) for j in range(num_bins)],
            penalty_1,
            -1)
    
    # Add "activate item only when bin is used" constraint
    for j in range(num_bins):
        for i in range(num_items):
            # Penalize each x_{ij} if y_j is 0
            bqm.add_interaction(f"y{j}", f"x{i}{j}", penalty_3)

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=5000)
    print(sampleset.first[0])

    for i in range(num_items):
        for j in range(num_bins):
            print(f'x{i}{j} ' + str(sampleset.first[0][f'x{i}{j}']))
        print()
    for j in range(num_bins):
        print(f'y{j}' + str(sampleset.first[0][f'y{j}']))

brute_force() 
annealing()


## Explanation:
# brute force works with mutliple test cases, seems to always return correct solution. 
# When testing with lower values for bins and items, I had to reduce the penalty

# Both annealing implementations dont seem to work. The returned output from the annealing2() seems to 
# not respect the constraints. As you can see below, the value for x00 and x01 is 1, implying item 0 
# is in both bins 0 and 1, which is incorrect. Also I was not able to figure out why all the y's are 0

# x00 1
# x01 1
# x02 0

# x10 0
# x11 0
# x12 1

# x20 1
# x21 0
# x22 1

# x30 0
# x31 1
# x32 0

# y00
# y10
# y20