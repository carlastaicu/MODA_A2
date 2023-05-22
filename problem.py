import numpy as np
from desdeo_tools.solver import ScalarMethod
from scipy.optimize import minimize
from desdeo_problem.problem.Objective import  _ScalarObjective
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem.problem.Variable import variable_builder
from desdeo_emo.population.SurrogatePopulation import SurrogatePopulation
from operators import Mutation, Crossover

import utils

class BicycleProblem:
    def __init__(self, var_count, pop_size, distance_matrix, beauty_matrix, roughness_matrix, safety_matrix, slope_matrix, goal_nodes):
        self.var_count = var_count
        self.pop_size = pop_size
        self.distance_matrix = distance_matrix
        self.beauty_matrix = beauty_matrix
        self.roughness_matrix = roughness_matrix
        self.safety_matrix = safety_matrix
        self.slope_matrix = slope_matrix
        self.goal_nodes = goal_nodes


    def create_goal_nodes_array(self, pop_size):
        goal_nodes_array = np.zeros((pop_size,self.var_count))
        goal_nodes_array[:, self.goal_nodes] = 1
        return goal_nodes_array

    # Defining the objective functions

    def total_metric(self, path, metric, inverse=False, average=True):
        # Keep track of which interest nodes have not been visited yet
        goal_nodes_copy = self.create_goal_nodes_array(len(path))
        matrix_str = metric + "_matrix"
        matrix = getattr(self, matrix_str)
        total = 0
        total += matrix[0][path[:, 0]]
        length = np.ones(path.shape[0])
        for i in range(path.shape[1]-1):
            goal_nodes_copy[np.arange(len(path)), path[:, i]-1] = 0
            mask = np.zeros(path.shape[0], dtype=matrix.dtype)
            mask[np.where(np.sum(goal_nodes_copy, axis=1))] = 1
            # The route is finished once all interest nodes are visited
            length += mask
            if np.sum(mask) == 0:
                break
            if not inverse:
                total += matrix[path[:, i],path[:, i+1]] * mask
            else:
                # Invert the comfort value to maximize all of them
                total += (6-matrix[path[:, i],path[:, i+1]]) * mask
        if average:
            total = total / length

        return total
    
    # Minimize
    def total_distance(self, path):
        return self.total_metric(path, "distance", average=False)
    
    # Maximize
    def total_beauty(self, path):
        return self.total_metric(path, "beauty")
        
    # Maximize
    def total_roughness(self, path):
        return self.total_metric(path, "roughness")

    # Minimize
    def total_safety(self, path):
        return self.total_metric(path, "safety", inverse=True)

    # Minimize
    def total_slope(self, path):
        return self.total_metric(path, "slope", inverse=True)
    
    def total_comfort(self, path):
        return (self.total_beauty(path) + self.total_roughness(path)
                + self.total_safety(path) + self.total_slope(path))

    def create_problem(self, obj_weights = np.ones(5), constraints = None, pfront = False, sum_comfort = False):
        num_constraints = 5 if not sum_comfort else 2
        if constraints is not None:
            if constraints.shape[0] != num_constraints or constraints.shape[1] != 2:
                raise("invalid constraints")
            elif pfront: # Flip the values if pfront
                for i in range(0,num_constraints):
                    np.flip(constraints[i])
                    if constraints[i][0] is not None: constraints[i][0] = -constraints[i][0]
                    if constraints[i][1] is not None: constraints[i][1] = -constraints[i][1]
        else:
            constraints = np.array([None] * 2 * num_constraints).reshape((num_constraints,2))
        
        if type(obj_weights) is not np.ndarray:
            obj_weights = np.array(obj_weights)
            
        # objective functions
        distance = lambda path: self.total_distance(path) * obj_weights[0]
        beauty = lambda path: self.total_beauty(path) * obj_weights[1]
        rough = lambda path: self.total_roughness(path) * obj_weights[2]
        safe = lambda path: self.total_safety(path) * obj_weights[3]
        slope = lambda path: self.total_slope(path) * obj_weights[4]
        total_comfort = lambda path: (beauty(path) * obj_weights[1] + 
                                      rough(path) * obj_weights[2] + 
                                      safe(path) * obj_weights[3] + 
                                      slope(path) * obj_weights[4])

        obj1 = _ScalarObjective("total_distance", distance, maximize=False)
        obj1_pfront = _ScalarObjective("distance", lambda path: distance(path), maximize=False)
        if sum_comfort:
            obj2 = _ScalarObjective("total_comfort", total_comfort, maximize=True)
            obj2_pfront = _ScalarObjective("total_comfort_p", lambda path: -1*total_comfort(path), maximize=True)
            # List of objectives for MOProblem class
            objectives = np.array([obj1, obj2])
            objectives_pfront = np.array([obj1_pfront, obj2_pfront])
            obj_f = [self.total_distance, self.total_comfort]
        else:
            # Objectives for desdeo problem
            # Minimizing distance
            obj2 = _ScalarObjective("total_beauty", beauty, maximize=True)
            obj3 = _ScalarObjective("total_roughness", rough, maximize=True)
            obj4 = _ScalarObjective("total_safety", safe, maximize=True)
            obj5 = _ScalarObjective("total_slope", slope, maximize=True)

            # Objectives for pareto front solver variation. 
            obj2_pfront = _ScalarObjective("total_beauty_p", lambda path: -1*beauty(path), maximize=False)
            obj3_pfront = _ScalarObjective("total_roughness_p", lambda path: -1*rough(path), maximize=False)
            obj4_pfront = _ScalarObjective("total_safety_p", lambda path: -1*safe(path), maximize=False)
            obj5_pfront = _ScalarObjective("total_slope_p", lambda path: -1*slope(path), maximize=False)
            
            objectives = np.array([obj1, obj2, obj3, obj4, obj5])
            objectives_pfront = np.array([obj1_pfront, obj2_pfront, obj3_pfront, obj4_pfront, obj5_pfront])
            obj_f = [self.total_distance, self.total_beauty, self.total_roughness, self.total_safety, self.total_slope]


        objectives_count = len(objectives)


        initial_pop = np.array([np.random.permutation(range(1, self.var_count+1)) for _ in range(self.pop_size)])
        var_names = [f"vertex {i}" for i in range(1, self.var_count + 1)] 

        # set lower bounds for each variable
        lower_bounds = np.array([0] * self.var_count)

        # set upper bounds for each variable
        upper_bounds = np.array([self.var_count]  * self.var_count)

        # Create a list of Variables for MOProblem class
        variables = variable_builder(var_names, initial_pop[0], lower_bounds, upper_bounds)

        cons = []
        for i in range(5):
            lower, upper = constraints[i]
            if lower is not None:
                con = utils.constraint_builder(lambda path: obj_f[i](path), objectives_count, self.var_count, lower, True, f"c{i}l")
                cons.append(con)
            if upper is not None:
                con = utils.constraint_builder(lambda path: obj_f[i](path), objectives_count, self.var_count, upper, False, f"c{i}u")
                cons.append(con)
        
        # Create the problem
        # This problem object can be passed to various different methods defined in DESDEO
        problem = MOProblem(objectives=objectives, variables=variables, constraints=cons)
        problem_pfront = MOProblem(objectives = objectives_pfront, variables=variables, constraints=cons)

        xover = Crossover()
        mutation = Mutation()


        scipy_de_method = ScalarMethod(
            lambda x, _, **y: minimize(x, **y, x0 = np.random.rand(problem.n_of_variables)),
            method_args={"method":"SLSQP"},
            use_scipy=True
        )

        # Ideal and nadir points
        ideal = np.array([0, 5, 5, 5, 5])[range(len(obj_weights))]
        nadir = np.array([np.inf, 1, 1, 1, 1])[range(len(obj_weights))]

        # Pass ideal and nadir to the problem object
        problem.ideal = ideal
        problem.nadir = nadir

        problem_pfront.ideal = -1*ideal
        problem_pfront.nadir = nadir

        population = SurrogatePopulation(problem, 100, initial_pop, xover, mutation, recombination=None)
        population_pfront = SurrogatePopulation(problem_pfront, 100, initial_pop, xover, mutation, recombination=None)

        return (population_pfront if pfront else population), scipy_de_method

