from desdeo_problem.problem.Constraint import ScalarConstraint
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pandas as pd


def save(name, objectives, decision, nadir, ideal):
    np.savez(
        f"modules/DataAndVisualization/{name}.npz",
        obj = objectives,
        var = decision,
        nadir = nadir,
        ideal = ideal
    )
    print("Saved successfully")

def load(name):
    d = np.load(f"modules/DataAndVisualization/{name}.npz")
    if d['nadir'] is not None and d['ideal'] is not None:
        return d['obj'], d['var'], d['nadir'], d['ideal']
    else:
        return d['obj'], d['var'], None, None


# """
# From desdeo_problem documentation:
# The constraint should be defined so, that when evaluated, it should return a positive value, 
# if the constraint is adhered to, and a negative, if the constraint is breached.
# """
def constraint_builder(f, n_obj, n_var, bound, is_lower_bound = True, name= "c1"):
    c = lambda xs, _ys: f(xs) - bound if is_lower_bound else bound - f(xs) 
    return ScalarConstraint(name, n_var, n_obj, c)

def remove_xy_duplicates_w_lowest_z(arr):
 
    t = np.unique(arr[:,:2], axis = 0)
    t = np.append(t, np.zeros((t.shape[0], 1)), axis = 1)
    if t.shape[0] == arr.shape[0]: # All copied, so no duplicates
        return arr

    # some duplicates, get the ones with lowest z value
    for row in t:
        dupl = arr[np.all(row[:2] == arr[:,:2], axis = -1)] # Get the ones with same x y values
        row[2] = np.min(dupl[:,2]) # Set the z value to the lowest of z value of the rows with same x, y values
         
    return t

def check_if_permutation(path):
    var_count = len(path)
    if np.sum(path > var_count) != 0:
        return False
    if np.sum(path < 1) != 0:
        return False
    _, counts = np.unique(path, return_counts=True)
    if np.sum(counts > 1) != 0:
        return False
    return True

def symmetric_matrix(matrix, integers = False):
    matrix = (matrix + matrix.T) / 2
    if integers:
        diff = np.ones_like(matrix) * 0.2
        probs = np.random.randint(2, size=matrix.shape)
        # Round up or down with equal probability
        matrix = (matrix + 0.1 - diff * probs).astype(int)
    for i in range(len(matrix)):
        matrix[i][i] = 0
    return matrix

def check_constraints(path, obj, constraints, mask=[True]*5):
    metrics = ["distance", "beauty", "roughness", "safety", "slope"]
    all_good = np.array([True]*len(path))
    for i, m in enumerate(metrics):
        if mask[i]:
            value = obj[:,i]
            if m == "safety" or m == "slope":
                value = 6 - value
            if constraints[i][0] is not None:
                all_good = all_good & (value > constraints[i][0])
            if constraints[i][1] is not None:
                all_good = all_good & (value < constraints[i][1])
    return all_good

def scatter_plot(obj_optimized, obj_random, name):
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(2,2, figsize=(6,6))
    fig.tight_layout(pad=3.0)
    if "diff" in name:
        fig.suptitle("Comfort parameters with different weights \n Optimized vs random solutions", fontsize=15, y=0.98)
    elif "small" in name:
        fig.suptitle("Smaller problem (nodes=5) \n Optimized vs random solutions", fontsize=15, y=0.98)
    else:
        fig.suptitle("All comfort parameters with same weights \n Optimized vs random solutions", fontsize=15, y=0.98)
    list_obj = ["beauty", "roughness", "safety", "slope"]
    for i, obj_name in enumerate(list_obj):
        subplot = ax[i//2][i%2]
        subplot.scatter(obj_random[:, 0], obj_random[:, i+1], label="Random")
        subplot.scatter(obj_optimized[:, 0], obj_optimized[:, i+1], label="Pareto")
        subplot.set_xlabel("Total distance (km)")
        subplot.set_ylabel(f"Average {obj_name}")
        handles, labels = subplot.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0))
    fig.savefig("plots/" + name + ".png", dpi=800, bbox_inches='tight')

def scatter_plot_2obj(objs_optimized, obj_random, name):
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(2,2, figsize=(6,6))
    fig.tight_layout(pad=3.0)
    fig.suptitle("Optimizing one comfort parameter at a time \n Optimized vs random solutions", fontsize=15, y=0.98)
    list_obj = ["beauty", "roughness", "safety", "slope"]
    for i, obj_name in enumerate(list_obj):
        subplot = ax[i//2][i%2]
        subplot.scatter(obj_random[:, 0], obj_random[:, i+1], label="Random")
        subplot.scatter(objs_optimized[i][:, 0], objs_optimized[i][:, i+1], label="Pareto")
        subplot.set_xlabel("Total distance (km)")
        subplot.set_ylabel(f"Average {obj_name}")
        handles, labels = subplot.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0))
    fig.savefig("plots/" + name + ".png", dpi=800, bbox_inches='tight')

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def filter_data(var, obj, constraints, mask, list_obj, name, colors):
    new_colors = []
    data = pd.DataFrame(obj, columns=["distance"] + list_obj)
    constraints_mask = check_constraints(var, obj, constraints, mask=mask)
    data_follow = data[constraints_mask]
    data_follow["label"] = name + " follow constraints"
    data_not_follow = data[np.invert(constraints_mask)]
    data_not_follow["label"] = name + " not follow constraints"
    if len(data_not_follow) > 0:
        new_colors.append(colors[1])
    if len(data_follow) > 0:
        new_colors.append(colors[0])
    return data_follow, data_not_follow, new_colors
    

def parallel_plot(var_optimized, var_random, obj_optimized, obj_random, constraints, obj_weights, name):
    list_obj = ["beauty", "roughness", "safety", "slope"]
    data_pareto_follow_constraints, data_pareto_not_follow_constraints, col1 = filter_data(var_optimized, 
                                                                                          obj_optimized, constraints, obj_weights>0, 
                                                                                          list_obj, "Pareto", ["#5DEC00", "#109300"])
    data_random_follow_constraints, data_random_not_follow_constraints, col2 = filter_data(var_random, obj_random, constraints, obj_weights>0, 
                                                                                            list_obj, "Random",["#F11200", "#8A0101"])

    colors = col2 + col1
    data = pd.concat([data_random_not_follow_constraints, data_random_follow_constraints,
                    data_pareto_not_follow_constraints, data_pareto_follow_constraints])
    data.iloc[:, :-1] = data.iloc[:, :-1].apply(normalize)
    fig = plt.figure()
    if "diff" in name:
        fig.suptitle("Comfort parameters with different weights \n Optimized vs random solutions", fontsize=15, y=0.98)
    elif "small" in name:
        fig.suptitle("Smaller problem (nodes=5) \n Optimized vs random solutions", fontsize=15, y=0.98)
    else:
        fig.suptitle("All comfort parameters with same weights \n Optimized vs random solutions", fontsize=15, y=0.98)
    pd.plotting.parallel_coordinates(data, 'label', color=colors)
    fig.savefig("plots/" + name + ".png", dpi=800, bbox_inches='tight')

def parallel_plot_2obj(vars_optimized, var_random, 
                       objs_optimized, obj_random, constraints, name):
    list_obj = ["beauty", "roughness", "safety", "slope"]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(2,2, figsize=(6,6))
    fig.suptitle("Optimizing one comfort parameter at a time \n Optimized vs random solutions", fontsize=15, y=0.98)

    all_handles = []
    all_labels = []
    for i, obj_name in enumerate(list_obj):
        colors = None
        subplot = ax[i//2][i%2]
        mask = np.array([1,0,0,0,0])
        mask[i+1] = 1
        data_random_follow_constraints, data_random_not_follow_constraints, col2 = filter_data(var_random, obj_random, constraints, mask, 
                                                                                            list_obj, "Random", ["#F11200", "#8A0101"])
        data_pareto_follow_constraints, data_pareto_not_follow_constraints, col1 = filter_data(vars_optimized[i], 
                                                                                            objs_optimized[i], constraints, mask, 
                                                                                            list_obj, "Pareto", ["#5DEC00", "#109300"])
        colors = col2 + col1
        data = pd.concat([data_random_not_follow_constraints, data_random_follow_constraints,
                        data_pareto_not_follow_constraints, data_pareto_follow_constraints])
        data = data.iloc[:,[0,i+1,-1]]
        data.iloc[:, :-1] = data.iloc[:, :-1].apply(normalize)
        pd.plotting.parallel_coordinates(data, 'label', color=colors, ax = subplot)
        handles, labels = subplot.get_legend_handles_labels()
        for i in range(len(labels)):
            if labels[i] not in all_labels:
                all_handles.append(handles[i])
                all_labels.append(labels[i])
        subplot.get_legend().remove()
    fig.tight_layout(pad=3.0)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 0))
    fig.savefig("plots/" + name + ".png", dpi=800, bbox_inches='tight')
