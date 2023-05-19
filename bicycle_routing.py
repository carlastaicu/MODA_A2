from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from desdeo_emo.EAs import NSGAIII
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from problem import BicycleProblem

def check_if_permutation(path):
    var_count = len(path)
    if np.sum(path > var_count) != 0:
        return False
    if np.sum(path < 1) != 0:
        return False
    _, counts = np.unique(path, return_counts=True)
    if np.sum(counts > 1) != 0:
        return False
    return True

def symmetric_matrix(matrix):
    matrix = (matrix + matrix.T) / 2
    for i in range(len(matrix)):
        matrix[i][i] = 0
    return matrix

def check_constraints(bicycle_problem, path):
    metrics = ["beauty", "roughness", "safety", "slope"]
    all_good = True
    for i, m in enumerate(metrics):
        value = bicycle_problem.total_metric(path, m)
        if constraints[i][0] is not None and np.sum(value < constraints[i][0]) != 0:
            print(f"Lower bound for metric {m} violated")
            all_good = False
        if constraints[i][1] is not None and np.sum(value > constraints[i][1]) != 0:
            print(f"Upper bound for metric {m} violated")
            all_good = True
    return all_good

# Which objectives do we wish to optimize
# scenic beauty, roughness, safety, slope
# we want to minimize total distance and maximize comfort 
# (comfort is given by beauty, roughness, safety, slope)
obj_weights = np.array([1, 1, 1, 1, 1])
variable_count = 15 # Around 15 - 25 seems to be good enough

# Set constraint for objectives, [lower, upper]
# If no constraint then set it to None
# Each row represents a objective function in the same order as in obj_gd 
# Notice that breaking constraints will result in a penalty and therefore we might get results that break the constraints
constraints = np.array([
    [3*variable_count, None], # Scenic beauty > 3 
    [3*variable_count, None], # roughness > 3
    [None, 2*variable_count], # Safety < 2
    [None, 3*variable_count], # Slope < 3 
])

# How many 3d points should the hull be formed of
# more points => More complex problem : longer execution times
# Less points => More likely to fail in constructing the hull
pop_size = 100

# To create the problem we can call the gd_create method with the parameters defined earlier
# the pfront argument should be set to True if using the solve_pareto_front_representation method as it doesn't 
# take account minimizing/maximizing. For everything else we can set it to False
# The method returns a MOProblem and a scalarmethod instance which can be passed to different Desdeo objects
distance_matrix = symmetric_matrix(np.random.uniform(0, 50, size=(variable_count+1, variable_count+1)))

beauty_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)))
roughness_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)))
safety_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)))
slope_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)))

bicycle_problem = BicycleProblem(variable_count, pop_size, distance_matrix, beauty_matrix, roughness_matrix, safety_matrix, slope_matrix)
population, method = bicycle_problem.create_problem(obj_weights, constraints, pfront = True)

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
                  n_iterations=10,
                  n_gen_per_iter=100,
                  population_size=100)

while evolver.continue_evolution():
    evolver.iterate()

var, obj, _ = evolver.end()

print("var: ", var)
print("obj: ", obj)
print("min dist: ", min(obj[:,0]))
for path in var:
    if not check_if_permutation(path):
        print("solution is not a permutation")
if not check_constraints(bicycle_problem, var):
    print("Some solution not fulfilling constraints")

# save the solution if you wish, make sure to change the name to not accidentally overwrite an existing solution.
# Saved solutions can be used later to visualize it
# The solution will be saved to modules/DataAndVisualization/'name'
# save("gdExample", obj, var, problem.nadir, problem.ideal)