from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
import numpy as np
from random import shuffle

class Mutation:
    def __init__(self, mut_rate: float = 0.2):
        self.mut_rate = mut_rate

    def do(self, offspring: np.ndarray):
        import pdb; pdb.set_trace()
        var_count = len(offspring[0])
        print("MUTATION")
        
        
        return offspring


class Crossover:
    def do(self, pop, mating_pop_ids):
        print("CROSSOVER")
        pop_size = len(pop)
        if mating_pop_ids is None:
            shuffled_ids = list(range(pop_size))
            shuffle(shuffled_ids)
        else:
            shuffled_ids = mating_pop_ids
        import pdb; pdb.set_trace()
        return pop[shuffled_ids]


# class SurrogatePopulation(Population, BasePopulation):
#     def __init__(
#         self, problem, pop_size: int, initial_pop, xover, mutation, recombination
#     ):
#         BasePopulation.__init__(self, problem, pop_size)
#         self.add(initial_pop)
#         self.xover = xover
#         self.mutation = mutation
#         self.recombination = recombination