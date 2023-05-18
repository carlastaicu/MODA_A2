from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
import numpy as np
from random import shuffle

class Mutation:
    def __init__(self, mut_rate: float = 0.2):
        self.mut_rate = mut_rate

    def do(self, offspring: np.ndarray):
        # import pdb; pdb.set_trace()
        var_count = len(offspring[0])
        print("MUTATION")
        
        
        return offspring


class Crossover:
    def __init__(self, xover_rate: float = 0.4):
        self.xover_rate = xover_rate

    def do(self, pop, mating_pop_ids):
        print("CROSSOVER")
        pop_size = len(pop)
        if mating_pop_ids is None:
            shuffled_ids = list(range(pop_size))
            shuffle(shuffled_ids)
        else:
            shuffled_ids = mating_pop_ids
        n_features = len(pop[0])
        for i in range(0, pop_size, 2):
            number_xover = int(self.xover_rate * n_features)  # Number of crossover points
            xover_points = np.random.choice(list(range(n_features)), number_xover, replace=False)  # Randomly select the crossover points
            values_xover = pop[shuffled_ids[i]][xover_points]  # Values from the first parent to be swapped in the second parent
            k = 0  # Counter for the values to be swapped
            for j in range(n_features):
                if pop[shuffled_ids[i+1]][j] in values_xover:  # If the value is in the first parent, swap it with the value from the second parent
                    pop[shuffled_ids[i+1]][j] = values_xover[k]
                    k += 1
        return pop


# class SurrogatePopulation(Population, BasePopulation):
#     def __init__(
#         self, problem, pop_size: int, initial_pop, xover, mutation, recombination
#     ):
#         BasePopulation.__init__(self, problem, pop_size)
#         self.add(initial_pop)
#         self.xover = xover
#         self.mutation = mutation
#         self.recombination = recombination