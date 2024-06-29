# HMM.py
# Author: Charles Zhu
# Date: 3/3/24
# Description: 
# Predicts sequences most likely to overlap the body of a histone using a Hidden Markov model and a list of marked intervals

########## Imports ##########
from typing import List


########## Globals ##########

x = '' # list of emissions
alphabet = ['x', 'y', 'z', 'n']
states = [0, 1]

# parameters
transition = [
    [0.8, 0.2],
    [0.2, 0.8]
]
emission = [
    [0.1, 0.1, 0.05, 0.75],
    [0.25, 0.25, 0.45, 0.05]
]
n_iterations = 100
n_predictions = 50000

symbol_labels = {
    'x': 0,
    'y': 1,
    'z': 2,
    'n': 3
}


########## Algo ##########

def main():
    # read emission
    print('reading emissions!')
    read_data()

    # run baum-welch algorithm for n iterations
    print('running baum-welch algorithm!')
    emission_responsibility = run_baum_welch(n_iterations)

    # get intervals with the highest probability of overlapping histones
    print(f'making predictions!')
    histone_overlap_probabilities = emission_responsibility[1]
    predictions = sorted(range(len(x)), key=lambda i: histone_overlap_probabilities[i], reverse=True)[:n_predictions]
    predictions = sorted([i+1 for i in predictions])

    # output predictions
    print('outputting predictions!')
    with open('predictions.csv', 'w') as f:
        for interval in predictions:
            f.write(f'{interval}\n')
        f.close()


# reads emissions from file
def read_data():
    global x

    file_name = "input.fasta"
    with open(file_name) as f:
        emissions = f.read().split()[1:]
        x = ''.join(emissions)
        f.close()


# runs baum welch algorithm for n iterations
def run_baum_welch(n_iterations: int):
    global transition, emission

    for i in range(n_iterations):
        print(f'running iteration {i+1}/{n_iterations}!')

        print('estimating responsibility matrices...')
        responsibility1, responsibility2 = estimate_responsibility_matrices()

        print('estimating parameters...')
        transition, emission = estimate_parameters(responsibility1, responsibility2)

        print_matrix(transition, "Transition", 3)
        print_matrix(emission, "Emission", 3)

    return responsibility1


# generates responsibility matrices
def estimate_responsibility_matrices() -> List[List[float]]:

    # generate forward and backwards graphs
    forward = generate_forward_graph()
    backward = generate_backward_graph()

    # generate responsibility matrix 1
    n = len(x)
    l1 = len(states)
    r1 = [[0 for _ in range(n)] for _ in range(l1)] # |States| x n matrix

    for i in range(n):
        for k in range(l1):
            # for each state-symbol combo, calculate probability of the state + symbol using forward and backward graphs
            r1[k][i] = forward[k][i] * backward[k][i]

        # for each interval, normalize each value by dividing it by the state-symbol combinations at that interval
        total_interval_probability = sum([row[i] for row in r1])
        for k in range(l1):
            r1[k][i] = r1[k][i] / total_interval_probability


    # generate responsibility matrix 2
    r2 = [[[0 for _ in range(n-1)] for _ in range(l1)] for _ in range(l1)] # |States| x |States| x (n-1) matrix

    # calculate each cell value
    for i in range(n-1):
        for l in range(l1):
            for k in range(l1):
                # for each state-symbol combo, calculate probability of the state + symbol using forward and backward graphs
                r2[l][k][i] = forward[l][i] * weight(l, k, i+1) * backward[k][i+1]

         # for each interval, normalize each value by dividing it by the sum of the state-state combinations at that interval
        total_interval_probability = sum([r2[l][k][i] for k in range(l1) for l in range(l1)])
        for l in range(l1):
            for k in range(l1):
                r2[l][k][i] = r2[l][k][i] / total_interval_probability

    return r1, r2


# generates forward graph
def generate_forward_graph() -> List[List[float]]:
    n = len(x)
    l1 = len(states)
    g = [[0 for _ in range(n)] for _ in range(l1)]

    # initialize first column with equal transition probabilities
    probability = 1 / l1
    for k in range(l1):
        g[k][0] = probability * emission[k][symbol_labels[x[0]]]

    # fill matrix column by column, left to right, starting with 2nd column
    for i in range(1, n):
        for k in range(l1):
            # for each state-symbol combo, calculate sum of probabilities of going from previous states to current state and emitting current symbol
            g[k][i] = sum([ g[l][i-1] * weight(l, k, i) for l in range(l1) ])

        # for each column, normalize each value by dividing it by the sum of the column (to avoid really small probabilities)
        total_interval_probability = sum([row[i] for row in g])
        for k in range(l1):
            g[k][i] = g[k][i] / total_interval_probability
    
    return g


# generates backward graph
def generate_backward_graph() -> List[List[float]]:
    n = len(x)
    l1 = len(states)
    g = [[0 for _ in range(n)] for _ in range(l1)]

    # initialize last column with 1
    for k in range(l1):
        g[k][-1] = 1

    # fill matrix column by column, right to left, starting with second-to-last column
    for i in range(n-2, -1, -1):
        for k in range(l1):
            # for each state-symbol combo, calculate sum of probabilities of going from current state to next states and emitting next symbol
            g[k][i] = sum([ g[l][i+1] * weight(k, l, i+1) for l in range(l1) ])

        # for each column, normalize each value by dividing it by the sum of the column (to avoid really small probabilities)
        total_interval_probability = sum([row[i] for row in g])
        for k in range(l1):
            g[k][i] = g[k][i] / total_interval_probability
    
    return g


# returns weight of edge connecting (from_state, index-1) to (to_state, index)
def weight(from_state: int, to_state: int, index: int) -> float:
    return transition[from_state][to_state] * emission[to_state][symbol_labels[x[index]]]


# estimates parameters using responsibility matrices
def estimate_parameters(r1: List[List[float]], r2: List[List[List[float]]]):
    l1 = len(r1)
    l2 = len(alphabet)
    n = len(r1[0])

    # generate T
    T = [[0 for _ in range(l1)] for _ in range(l1)]
    for l in range(l1):
        for k in range(l1):
            T[l][k] = sum(r2[l][k])
    
    # estimate transition matrix
    new_transition = [[0 for _ in range(l1)] for _ in range(l1)]
    for l in range(l1):
        total_state_probability = sum(T[l])
        for k in range(l1):
            new_transition[l][k] = T[l][k] / total_state_probability

    # generate E
    E = [[0 for _ in range(l2)] for _ in range(l1)]
    for k in range(l1):
        for i in range(l2):
            E[k][i] = sum([ r1[k][j] for j in range(n) if symbol_labels[x[j]] == i ])

    # estimate emission matrix
    new_emission = [[0 for _ in range(l2)] for _ in range(l1)]
    for k in range(l1):
        total_state_probability = sum(E[k])
        for b in range(l2):
            new_emission[k][b] = E[k][b] / total_state_probability

    return new_transition, new_emission


# prints a matrix v nicely :D
def print_matrix(g: List[List[int]], title=None, decimals=4):
    print()
    if (title):
        print(f'{title}:')

    print('[')
    for i in range(len(states)):
        row = [f'{x:.{decimals}f}' for x in g[i]]
        print(f'  {states[i]}: {row},')
    print(']')
    print()


if __name__ == "__main__":
    main()