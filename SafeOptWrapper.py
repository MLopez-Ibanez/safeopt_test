from safeopt import SafeOpt
import GPy

def run_modified_safeopt(fun, n_seeds, n_evals):
    return run_safeopt(fun, n_seeds, n_evals, modified = True)
    
def run_safeopt(fun, n_seeds, n_evals, modified = False):
    fun._init_counters()
    x_safe_seed, y_safe_seed = fun.get_default_safe_seeds(n_seeds)
    kernel = GPy.kern.RBF(input_dim=fun.xdim, ARD=True)
    #kernel = GPy.kern.Matern52(fun.xdim,ARD=True)
    # The statistical model of our objective function
    gp = GPy.models.GPRegression(x_safe_seed, y_safe_seed, kernel=kernel, noise_var = 0)
    if modified:
        lipschitz = None
    else:
        lipschitz = fun.lipschitz
        
    opt = SafeOpt(gp, parameter_set=fun.x_matrix, fmin=fun.safe_threshold,
                  lipschitz = lipschitz, beta = 2, # This was the default
                  threshold=fun.safe_threshold)

    assert n_evals > n_seeds
    for i in range(n_evals):
        # Obtain next query point
        x_next = opt.optimize()
        # Get a measurement from the real system
        y_meas = fun(x_next)
        # Add this to the GP model
        opt.add_new_data_point(x_next, y_meas)
        print(f'evals={opt.t}\tx_next={x_next}\ty={y_meas}\tsafe={y_meas >= opt.fmin}')
    return opt


