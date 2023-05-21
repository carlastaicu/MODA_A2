from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from desdeo_emo.EAs import NSGAIII
import numpy as np
import warnings
import pandas as pd
from utils import check_constraints, check_if_permutation, symmetric_matrix

warnings.filterwarnings("ignore")
from problem import BicycleProblem

# Which objectives do we wish to optimize
# scenic beauty, roughness, safety, slope
# we want to minimize total distance and maximize comfort 
# (comfort is given by beauty, roughness, safety, slope)
obj_weights = np.array([1, 1, 1, 1, 1])
variable_count = 3  # Around 15 - 25 seems to be good enough
nodes_num = 2

# Set constraint for objectives, [lower, upper]
# If no constraint then set it to None
# Each row represents a objective function in the same order as in obj_gd 
# Notice that breaking constraints will result in a penalty and therefore we might get results that break the constraints
constraints = np.array([
    [None, None],
    [3.0, None], # Scenic beauty > 2 
    [3.0, None], # roughness > 2
    [2.0, None], # Inv Safety > 2
    [3.0, None], # Inv Slope > 2 
])

# How many 3d points should the hull be formed of
# more points => More complex problem : longer execution times
# Less points => More likely to fail in constructing the hull
pop_size = 100
pfront = False

# To create the problem we can call the gd_create method with the parameters defined earlier
# the pfront argument should be set to True if using the solve_pareto_front_representation method as it doesn't 
# take account minimizing/maximizing. For everything else we can set it to False
# The method returns a MOProblem and a scalarmethod instance which can be passed to different Desdeo objects
distance_matrix = symmetric_matrix(np.random.uniform(0, 50, size=(variable_count+1, variable_count+1)))

beauty_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
roughness_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
safety_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
slope_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
# distance_matrix = np.array([[0, 4.1, 3.9, 2.5],
#                             [4.1, 0, 1.4, 4.8],
#                             [3.9, 1.4, 0, 2.7],
#                             [2.5, 4.8, 2.7, 0]])
# beauty_matrix = np.array([[0, 2, 1, 3],
#                           [2, 0, 5, 2],
#                           [1, 5, 0, 4],
#                           [3, 2, 4, 0]])
# roughness_matrix = beauty_matrix.copy()
# safety_matrix = beauty_matrix.copy()
# slope_matrix = beauty_matrix.copy()

# goal_nodes = np.random.choice(variable_count, size=nodes_num, replace=False)
goal_nodes = np.array([1,2])

bicycle_problem = BicycleProblem(variable_count, pop_size, distance_matrix, beauty_matrix, roughness_matrix, safety_matrix, slope_matrix, goal_nodes=goal_nodes)

population, method = bicycle_problem.create_problem(obj_weights, pfront = pfront, sum_comfort = False)

# Two methods to solve the problem are shown below. Do not use them both at the same time!
# Use one, and comment out the other!

# Example on solving the pareto front : This might take some time so feel free to comment this out (lines 57 and 60).

# We will use the solve_pareto_front_representation method but one can change this to something else.
# The method takes the problem instance and a step size array

# The method will create reference points from nadir to ideal with these step sizes
# in this case : ref points = [[5,0,0,0], [4.5, 0, 0, 0], [4, 0, 0, 0] ... [5, 0.2, 0, 0] ... [0, 1, 1, 1]]
# large step sizes => less solutions but faster calculation
# step_sizes = np.array([.5, .2, .2, .2])[obj]

# # The method returns the decision vectors and corresponding objective vectors
# var, obj = solve_pareto_front_representation(problem, step_sizes, solver_method= method)

# Example on solving the pareto front using NSGA-III
evolver = NSGAIII(problem=None,
                  initial_population=population,
                  n_iterations=50,
                  n_gen_per_iter=100,
                  population_size=100)

while evolver.continue_evolution():
    evolver.iterate()


var, obj, _ = evolver.end()
m = "safety"
bicycle_problem.total_metric(var,  "safety", average=False,inverse=True)

if pfront:
    obj[:, 1:] = -obj[:, 1:]
obj[:, -2:] = (6 - obj[:, -2:]) * obj_weights[-2:]


print("var: ", var)
print("obj: ", obj)
print("min dist: ", min(obj[:,0]))
for path in var:
    if not check_if_permutation(path):
        print("solution is not a permutation")
check_constraints(bicycle_problem, var, constraints, mask=obj_weights>0)




