import numpy as np
import pandas as pd
from SafeProblem import Problem
from SafeOptWrapper import run_safeopt,run_modified_safeopt

# Objective function
# WARNING: GPy has issues with negative function values. The RBF kernel assumes the prior mean is zero. 
# Since zero is safe, SafeOpt gets very confused.
def sphere_broken(x): 
    x_opt = np.array([-1,-2])
    x = np.asarray(x) - x_opt
    #    return -((x*x).sum(axis=0))
    return -np.inner(x, x)

def sphere(x): return 100 + sphere_broken(x)

def rosenbrock(x):
    z = np.asarray(x)
    z = z + 1
    b = 100
    r = np.sum(b * (z[1:] - z[:-1]**2.0)**2.0 + (1 - z[:-1])**2.0, axis=0)
    r = 100 - np.log(r) # We take the log reduce the gradient and get a smaller lipschitz
    return r

safe_sphere = Problem(name = "sphere_2D_75",
                      fun = sphere, bounds = [(-5., 5.),(-5., 5.)], percentile = 0.75,
                      default_safe_seeds = [18643, 118129,  18766, 101797,  64078,  63071, 108293,  19281,  29693, 125759])
#safe_sphere = Problem(fun = sphere, bounds = [(-5., 5.),(-8., 2.)], percentile = 0.75,
#                     default_safe_seeds = [86225, 199784,  86241, 180581, 137070, 136068, 188095,  86672,  96619, 209639])
#safe_rosenbrok = Problem(fun = rosenbrock, bounds = [(-3., 3.),(-3.,3.)], percentile = 0.5)
fun = safe_sphere
n_evals = 100
n_reps = 10
N_seeds = [1,5,10]

algos = dict(SafeOpt = run_safeopt, SafeOptMod = run_modified_safeopt)
algo = "SafeOpt"
for algo in algos:
    for n_seeds in N_seeds:
        df_reps = []
        for r in range(n_reps):
            np.random.seed(r)
            algos[algo](fun, n_seeds = n_seeds, n_evals = n_evals)
            df_y = pd.DataFrame(dict(t = np.arange(1, n_evals + 1), y = fun.Y))
            df_y['rep'] = r + 1
        df_reps.append(df_y)
        df_reps = pd.concat(df_reps)
        df_reps.to_csv(f"results-{algo}-{fun.name}-nseeds={n_seeds}.csv", index=False)
