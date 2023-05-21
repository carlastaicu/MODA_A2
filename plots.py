from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from desdeo_emo.EAs import NSGAIII
from problem import BicycleProblem
import numpy as np
import warnings
import matplotlib.pyplot as plt
from utils import symmetric_matrix, scatter_plot, parallel_plot, scatter_plot_2obj, parallel_plot_2obj

warnings.filterwarnings("ignore")
from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation

def create_bicycle_problem(constraints, obj_weights=np.ones(5), 
                           variable_count: int = 20, nodes_num: int = 10, pfront: bool = True, pop_size=100):
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

def optimize_problem(bicycle_problem, population, pfront = True, obj_weights=np.ones(5)):
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

def create_random_samples(bicycle_problem, variable_count, obj_weights=np.ones(5), pop_size=100):
    var_random = np.array([np.random.permutation(range(1, variable_count+1)) for _ in range(pop_size)])
    random_distances = bicycle_problem.total_distance(var_random)
    random_beauty = bicycle_problem.total_beauty(var_random)
    random_roughness = bicycle_problem.total_roughness(var_random)
    random_safety = bicycle_problem.total_safety(var_random)
    random_slope = bicycle_problem.total_slope(var_random)
    obj_random = np.vstack([random_distances, random_beauty, random_roughness,
                            random_safety, random_slope]).T
    obj_random[:, -2:] = (6 - obj_random[:, -2:]) * (obj_weights[-2:] != 0)
    return var_random, obj_random

def plot_nominal_config(constraints, variable_count, nodes_num, pfront):

    # obj_weights = np.ones(5, dtype=int)
    obj_weights = np.array([1,50,50,50,50])

    problem, population = create_bicycle_problem(constraints.copy(), obj_weights, variable_count, nodes_num)
    var_optimized, obj_optimized = optimize_problem(problem, population, pfront, obj_weights)
    var_random, obj_random = create_random_samples(problem, variable_count, obj_weights)
    scatter_plot(obj_optimized, obj_random, "nominal_config_scatter")
    parallel_plot(var_optimized, var_random, obj_optimized, obj_random, constraints, obj_weights, "nominal_config_parallel")

def plot_different_weights(constraints, variable_count, nodes_num, pfront):
    obj_weights = np.array([1,100,1,100,1])

    problem, population = create_bicycle_problem(constraints.copy(), obj_weights, variable_count, nodes_num)
    var_optimized, obj_optimized = optimize_problem(problem, population, pfront, obj_weights)
    var_random, obj_random = create_random_samples(problem, variable_count, obj_weights)
    scatter_plot(obj_optimized, obj_random, "diff_weights_scatter")
    parallel_plot(var_optimized, var_random, obj_optimized, obj_random, constraints, obj_weights, "diff_weights_parallel")


def plot_2obj(constraints, variable_count, nodes_num, pfront):
    obj_weights = np.ones(5)
    problem, population = create_bicycle_problem(constraints.copy(), obj_weights, variable_count, nodes_num)
    var_random, obj_random = create_random_samples(problem, variable_count, obj_weights)
    # obj_weights = np.ones(5, dtype=int)
    vars_optimized = []
    objs_optimized = []
    for i in range(1, 5):
        obj_weights = np.array([1,0,0,0,0])
        obj_weights[i] = 1
        problem, population = create_bicycle_problem(constraints.copy(), obj_weights, variable_count, nodes_num)
        var_optimized, obj_optimized = optimize_problem(problem, population, pfront, obj_weights)
        vars_optimized.append(var_optimized)
        objs_optimized.append(obj_optimized)
    parallel_plot_2obj(vars_optimized, var_random, objs_optimized, obj_random, constraints, "1v1_parallel")
    scatter_plot_2obj(objs_optimized, obj_random, "1v1_scatter")

def plot_sizes(constraints, variable_count, pfront):
    nodes_num = 5
    obj_weights = np.array([1,50,50,50,50])

    problem, population = create_bicycle_problem(constraints.copy(), obj_weights, variable_count, nodes_num)
    var_optimized, obj_optimized = optimize_problem(problem, population, pfront, obj_weights)
    var_random, obj_random = create_random_samples(problem, variable_count, obj_weights)
    scatter_plot(obj_optimized, obj_random, "small_scatter")
    parallel_plot(var_optimized, var_random, obj_optimized, obj_random, constraints, obj_weights, "small_parallel")

if __name__ == "__main__":
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
    plot_2obj(constraints, variable_count, nodes_num, pfront)
    plot_nominal_config(constraints, variable_count, nodes_num, pfront)
    plot_different_weights(constraints, variable_count, nodes_num, pfront)
    plot_sizes(constraints, variable_count, pfront)
    # plt.figure(figsize=(6, 6))
    # for i in range(1, 5):
    #     obj_weights = [1, 0, 0, 0, 0]
    #     obj_weights[i] = 1
    #     run_experiment(obj_weights=obj_weights)
    #     print("Finished experiment ", i)
    # plt.savefig("plots/pareto_fronts_individual.png")
    # plt.show()
