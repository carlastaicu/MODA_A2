import numpy as np
import pandas as pd
from utils import symmetric_matrix
import os

variable_count = 20  # Around 15 - 25
distance_matrix = symmetric_matrix(np.random.uniform(0, 5, size=(variable_count+1, variable_count+1)))
if not os.path.exists('./dataset/'):
    os.mkdir('./dataset/')

DF = pd.DataFrame(distance_matrix)
DF.to_csv("dataset/distance_matrix.csv", index=False, header=False)

beauty_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
roughness_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
safety_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
slope_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)

DF = pd.DataFrame(beauty_matrix)
DF.to_csv("dataset/beauty_matrix.csv", index=False, header=False)
DF = pd.DataFrame(roughness_matrix)
DF.to_csv("dataset/roughness_matrix.csv", index=False, header=False)
DF = pd.DataFrame(safety_matrix)
DF.to_csv("dataset/safety_matrix.csv", index=False, header=False)
DF = pd.DataFrame(slope_matrix)
DF.to_csv("dataset/slope_matrix.csv", index=False, header=False)