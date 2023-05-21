import numpy as np
import pandas as pd

def symmetric_matrix(matrix, integers = False):
    matrix = (matrix + matrix.T) / 2
    if integers:
        matrix = matrix.astype(int)
    for i in range(len(matrix)):
        matrix[i][i] = 0
    return matrix
  
variable_count = 15  # Around 15 - 25 seems to be good enough
distance_matrix = symmetric_matrix(np.random.uniform(0, 50, size=(variable_count+1, variable_count+1)))

DF = pd.DataFrame(distance_matrix)
DF.to_csv("distance_matrix.csv", index=False, header=False)

beauty_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
roughness_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
safety_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)
slope_matrix = symmetric_matrix(np.random.randint(1, 5, size=(variable_count+1, variable_count+1)), integers=True)

DF = pd.DataFrame(beauty_matrix)
DF.to_csv("beauty_matrix.csv", index=False, header=False)
DF = pd.DataFrame(roughness_matrix)
DF.to_csv("roughness_matrix.csv", index=False, header=False)
DF = pd.DataFrame(safety_matrix)
DF.to_csv("safety_matrix.csv", index=False, header=False)
DF = pd.DataFrame(slope_matrix)
DF.to_csv("slope_matrix.csv", index=False, header=False)
