from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from desdeo_emo.EAs import NSGAIII
from problem import BicycleProblem
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation

def run_experiment(obj_weights=(1, 1, 0, 0, 0), variable_count: int = 15, pop_size=100):
    constraints = np.array([
        [3 * variable_count, None],  # Scenic beauty > 3
        [3 * variable_count, None],  # roughness > 3
        [None, 2 * variable_count],  # Safety < 2
        [None, 3 * variable_count],  # Slope < 3
    ])

    distance_matrix = np.random.uniform(0, 50, size=(variable_count + 1, variable_count + 1))
    beauty_matrix = np.random.randint(1, 5, size=(variable_count + 1, variable_count + 1))
    roughness_matrix = np.random.randint(1, 5, size=(variable_count + 1, variable_count + 1))
    safety_matrix = np.random.randint(1, 5, size=(variable_count + 1, variable_count + 1))
    slope_matrix = np.random.randint(1, 5, size=(variable_count + 1, variable_count + 1))
    bicycle_problem = BicycleProblem(variable_count, pop_size, distance_matrix, beauty_matrix, roughness_matrix,
                                     safety_matrix, slope_matrix)
    population, method = bicycle_problem.create_problem(obj_weights, constraints, pfront=True)
    evolver = NSGAIII(problem=None, initial_population=population,
                      n_iterations=50,
                      n_gen_per_iter=100,
                      population_size=100)

    while evolver.continue_evolution():
        evolver.iterate()

    var, obj, _ = evolver.end()

    print("var: ", var)
    print("obj: ", obj)
    print("min dist: ", min(obj[:, 0]))

    # Generate plots
    NUM_RANDOM_SAMPLES = 1000
    bicycle_problem = BicycleProblem(variable_count, NUM_RANDOM_SAMPLES, distance_matrix, beauty_matrix,
                                     roughness_matrix, safety_matrix, slope_matrix)
    random_samples = np.array([np.random.permutation(range(1, variable_count + 1)) for _ in range(NUM_RANDOM_SAMPLES)])
    random_distances = bicycle_problem.total_distance(random_samples)
    random_beauty = bicycle_problem.total_beauty(random_samples)
    random_roughness = bicycle_problem.total_roughness(random_samples)
    random_safety = bicycle_problem.total_safety(random_samples)
    random_slope = bicycle_problem.total_slope(random_samples)

    list_obj = ["beauty", "roughness", "safety", "slope"]
    i = np.argmax(obj_weights[1:]) + 1
    obj_name = list_obj[i-1]
    plt.subplot(2, 2, i)
    plt.scatter(random_distances, locals()["random_" + obj_name])
    plt.scatter(obj[:, 0], obj[:, i])
    plt.xlabel("Total distance")
    plt.ylabel(f"Total {obj_name}")


if __name__ == "__main__":
    plt.figure(figsize=(6, 6))
    for i in range(1, 5):
        obj_weights = [1, 0, 0, 0, 0]
        obj_weights[i] = 1
        run_experiment(obj_weights=obj_weights)
        print("Finished experiment ", i)
    plt.savefig("plots/pareto_fronts_individual.png")
    plt.show()
