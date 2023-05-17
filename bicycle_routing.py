from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from desdeo_emo.EAs import NSGAIII
from modules.utils import save
from modules.GeometryDesign.problem import create_problem
import numpy as np
import warnings
warnings.filterwarnings("ignore") # ignore warnings :)

# Creating geometry design problem : tent like buildings
# Which objectives do you wish to optimize
# scenic beauty, roughness, safety, slope
obj = np.array([
    True, True, True, True, #we want to minimize total distance and maximize comfort.
])

# Set constraint for objectives, [lower, upper]
# If no constraint then set it to None
# Each row represents a objective function in the same order as in obj_gd 
# Notice that breaking constraints will result in a penalty and therefore we might get results that break the constraints
constraints = np.array([
    [0.2, None], # Surface area > 0.2
    [.5, .8], # .5 < volume < .8. Even though we're not optimizing volume, we can set a constraint on it  
    [.4, None], #  min height > .4
    [None, 0.6], # floor area < .6 
])