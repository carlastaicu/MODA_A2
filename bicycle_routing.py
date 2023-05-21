from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from desdeo_emo.EAs import NSGAIII
import numpy as np
import warnings
import pandas as pd
from utils import check_constraints, check_if_permutation, symmetric_matrix

warnings.filterwarnings("ignore")
from problem import BicycleProblem

def create_bicycle_problem(constraints, obj_weights=np.ones(5), variable_count: int = 20, 
                           nodes_num: int = 10, pfront: bool = True, pop_size=100, retrieve_from_dataset=True):
    if retrieve_from_dataset:
        dataset_path = './dataset/'
        distance_matrix = pd.read_csv(dataset_path+'distance_matrix.csv', sep=',', header=None).to_numpy()
        beauty_matrix = pd.read_csv(dataset_path+'beauty_matrix.csv', sep=',', header=None).to_numpy()
        roughness_matrix = pd.read_csv(dataset_path+'roughness_matrix.csv', sep=',', header=None).to_numpy()
        safety_matrix = pd.read_csv(dataset_path+'safety_matrix.csv', sep=',', header=None).to_numpy()
        slope_matrix = pd.read_csv(dataset_path+'safety_matrix.csv', sep=',', header=None).to_numpy()
    else:
        distance_matrix = symmetric_matrix(np.random.uniform(0, 5, size=(variable_count+1, variable_count+1)))
        beauty_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
        roughness_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
        safety_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
        slope_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)

    goal_nodes = np.random.choice(variable_count, size=nodes_num, replace=False)

    bicycle_problem = BicycleProblem(variable_count, pop_size, distance_matrix, beauty_matrix, 
                                     roughness_matrix, safety_matrix, slope_matrix, goal_nodes=goal_nodes)
    population, method = bicycle_problem.create_problem(obj_weights, constraints, pfront = pfront, sum_comfort = False)
    return bicycle_problem, population

def optimize_problem(population, pfront = True, obj_weights=np.ones(5)):
    evolver = NSGAIII(problem=None,
                    initial_population=population,
                    n_iterations=1,
                    n_gen_per_iter=100,
                    population_size=100)

    while evolver.continue_evolution():
        evolver.iterate()

    var, obj, _ = evolver.end()
    for i in range(len(obj[0])):
        if obj_weights[i] != 0:
            obj[:, i] = obj[:, i] / obj_weights[i]
    if pfront:
        obj[:, 1:] = -obj[:, 1:]
    obj[:, -2:] = (6 - obj[:, -2:]) * (obj_weights[-2:] != 0)

    return var, obj

if __name__ == "__main__":

    obj_weights = np.array([1,50,50,50,50])
    constraints = np.array([
        [None, None],
        [2.0, None], # Scenic beauty > 2 
        [2.0, None], # roughness > 2
        [4.0, None], # Safety < 2
        [4.0, None], # Slope < 2 
    ])
    variable_count = 20
    nodes_num = 10
    pfront = True

    bicycle_problem, population = create_bicycle_problem(constraints.copy(), obj_weights, variable_count, nodes_num)
    var_optimized, obj_optimized = optimize_problem(population, pfront, obj_weights)

    for path in var_optimized:
        if not check_if_permutation(path):
            print("solution is not a permutation")
    print("Minimum distance found: ", min(obj_optimized[:, 0]))



