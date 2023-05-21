from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from desdeo_emo.EAs import NSGAIII
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd

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

def symmetric_matrix(matrix, integers = False):
    matrix = (matrix + matrix.T) / 2
    if integers:
        matrix = matrix.astype(int)
    for i in range(len(matrix)):
        matrix[i][i] = 0
    return matrix

def check_constraints(bicycle_problem, path):
    metrics = ["distance", "beauty", "roughness", "safety", "slope"]
    all_good = np.array([True]*len(path))
    for i, m in enumerate(metrics):
        value = bicycle_problem.total_metric(path, m)
        if constraints[i][0] is not None:
            all_good = all_good & (value < constraints[i][0])
        if constraints[i][1] is not None:
            all_good = all_good & (value > constraints[i][1])
    return all_good

# Which objectives do we wish to optimize
# scenic beauty, roughness, safety, slope
# we want to minimize total distance and maximize comfort 
# (comfort is given by beauty, roughness, safety, slope)
obj_weights = np.array([1, 1, 0, 0, 0])
variable_count = 15  # Around 15 - 25 seems to be good enough
dataset_path = './dataset/'

# Set constraint for objectives, [lower, upper]
# If no constraint then set it to None
# Each row represents a objective function in the same order as in obj_gd 
# Notice that breaking constraints will result in a penalty and therefore we might get results that break the constraints
constraints = np.array([
    [None, None],
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
# distance_matrix = symmetric_matrix(np.random.uniform(0, 50, size=(variable_count+1, variable_count+1)))

# beauty_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
# roughness_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
# safety_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
# slope_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)

df = pd.read_csv(dataset_path+'distance_matrix.csv', sep=',', header=None)
distance_matrix = df.to_numpy()
df = pd.read_csv(dataset_path+'beauty_matrix.csv', sep=',', header=None)
beauty_matrix = df.to_numpy()
df = pd.read_csv(dataset_path+'roughness_matrix.csv', sep=',', header=None)
roughness_matrix = df.to_numpy()
df = pd.read_csv(dataset_path+'safety_matrix.csv', sep=',', header=None)
safety_matrix = df.to_numpy()
df = pd.read_csv(dataset_path+'safety_matrix.csv', sep=',', header=None)
slope_matrix = df.to_numpy()

bicycle_problem = BicycleProblem(variable_count, pop_size, distance_matrix, beauty_matrix, roughness_matrix, safety_matrix, slope_matrix)
population, method = bicycle_problem.create_problem(obj_weights, pfront = True)

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

print("var: ", var)
print("obj: ", obj)
print("min dist: ", min(obj[:,0]))
for path in var:
    if not check_if_permutation(path):
        print("solution is not a permutation")
check_constraints(bicycle_problem, var)

# Generate plots
# Scatterplot F1, F2
NUM_RANDOM_SAMPLES = 100
bicycle_problem = BicycleProblem(variable_count, NUM_RANDOM_SAMPLES, distance_matrix, beauty_matrix, roughness_matrix,safety_matrix, slope_matrix)
random_samples = np.array([np.random.permutation(range(1, variable_count+1)) for _ in range(NUM_RANDOM_SAMPLES)])
random_distances = bicycle_problem.total_distance(random_samples)
random_beauty = bicycle_problem.total_beauty(random_samples)
random_roughness = bicycle_problem.total_roughness(random_samples)
random_safety = bicycle_problem.total_safety(random_samples)
random_slope = bicycle_problem.total_slope(random_samples)


plt.figure(figsize=(6, 6))
list_obj = ["beauty", "roughness", "safety", "slope"]
for i, obj_name in enumerate(list_obj, start=1):
    plt.subplot(2, 2, i)
    plt.scatter(random_distances, locals()["random_"+obj_name])
    plt.scatter(obj[:, 0], obj[:, i])
    plt.xlabel("Total distance")
    plt.ylabel(f"Total {obj_name}")
plt.savefig("plots/pfront_plot.png", dpi=800)

# import plotly.express as px
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

data_pareto = pd.DataFrame(obj, columns=["distance"] + list_obj)
data_pareto_follow_constraints = data_pareto[check_constraints(bicycle_problem, var)]
data_pareto_follow_constraints["label"] = "Pareto follow constraints"
data_pareto_not_follow_constraints = data_pareto[np.invert(check_constraints(bicycle_problem, var))]
data_pareto_not_follow_constraints["label"] = "Pareto not follow constraints"
data_random = pd.DataFrame(np.column_stack([random_distances, random_beauty, random_roughness, random_safety, random_slope]),
                           columns=["distance"] + list_obj)
data_random_follow_constraints = data_random[check_constraints(bicycle_problem, random_samples)]
data_random_follow_constraints["label"] = "Random follow constraints"
data_random_not_follow_constraints = data_random[np.invert(check_constraints(bicycle_problem, random_samples))]
data_random_not_follow_constraints["label"] = "Random not follow constraints"
data = pd.concat([data_random_follow_constraints, data_random_not_follow_constraints,
                  data_pareto_not_follow_constraints, data_pareto_follow_constraints])
data.iloc[:, :-1] = data.iloc[:, :-1].apply(normalize)
columns_to_check = [i for i, obj_weight in enumerate(obj_weights) if obj_weight != 0]
interested_data = data.iloc[:, columns_to_check + [-2]]

plt.figure()
pd.plotting.parallel_coordinates(data, 'label', color=["#F11200", "#8A0101", "#109300", "#5DEC00"])
plt.savefig("plots/parallel.png", dpi=800)
plt.show()
