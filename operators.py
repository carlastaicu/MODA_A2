from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
import numpy as np
from random import shuffle

class Mutation:
    def __init__(self, mut_rate: float = 0.2):
        self.mut_rate = mut_rate

    def do(self, offspring: np.ndarray):
        offspring_copy = offspring.copy()
        pop_size = len(offspring_copy)
        if pop_size == 0:
            return offspring_copy
        var_count = len(offspring_copy[0])
        for i in range(0, pop_size):
            offspring_i = offspring_copy[i].copy()
            number_mut = int(self.mut_rate * var_count)  # Number of mutated bits
            mut_points = np.random.choice(var_count, number_mut, replace=False) # Randomly select the mutated bits
            perm = np.random.permutation(len(mut_points))
            for j, p in enumerate(mut_points):
                offspring_i[p] = offspring_copy[i][mut_points[perm[j]]]
            offspring_copy[i] = offspring_i
        return offspring_copy


class Crossover:
    def __init__(self, xover_rate: float = 0.4):
        self.xover_rate = xover_rate

    def do(self, pop, mating_pop_ids):
        pop_copy = pop.copy()
        pop_size = len(pop_copy)
        if mating_pop_ids is None:
            shuffled_ids = list(range(pop_size))
            shuffle(shuffled_ids)
        else:
            shuffled_ids = mating_pop_ids
        var_count = len(pop_copy[0])
        for i in range(0, pop_size, 2):
            number_xover = int(self.xover_rate * var_count)  # Number of crossover points
            xover_points = np.random.choice(var_count, number_xover, replace=False)  # Randomly select the crossover points
            values_xover = pop_copy[shuffled_ids[i]][xover_points]  # Values from the first parent to be swapped in the second parent
            k = 0  # Counter for the values to be swapped
            for j in range(var_count):
                if pop_copy[shuffled_ids[i+1]][j] in values_xover:  # If the value is in the first parent, swap it with the value from the second parent
                    tmp = pop_copy[shuffled_ids[i+1]][j]
                    pop_copy[shuffled_ids[i+1]][j] = values_xover[k]
                    pop_copy[shuffled_ids[i]][xover_points[k]] = tmp
                    k += 1
        return pop_copy