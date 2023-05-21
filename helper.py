import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bicycle_routing import check_constraints





# import plotly.express as px


# Generate plots
# Scatterplot F1, F2
NUM_RANDOM_SAMPLES = 100
bicycle_problem = BicycleProblem(variable_count, NUM_RANDOM_SAMPLES, distance_matrix, beauty_matrix, roughness_matrix,safety_matrix, slope_matrix, goal_nodes)
random_samples = np.array([np.random.permutation(range(1, variable_count+1)) for _ in range(NUM_RANDOM_SAMPLES)])
random_distances = bicycle_problem.total_distance(random_samples)
random_beauty = bicycle_problem.total_beauty(random_samples)
random_roughness = bicycle_problem.total_roughness(random_samples)
random_safety = bicycle_problem.total_safety(random_samples)
random_slope = bicycle_problem.total_slope(random_samples)


