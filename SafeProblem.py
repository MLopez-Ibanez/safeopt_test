import numpy as np

def eval_on_grid(fun, xbound, n_steps):
    # This is similar but simpler than what safeopt.linearly_spaced_combinations does.
    x_1 = np.linspace(xbound[0][0], xbound[0][1], n_steps)
    x_2 = np.linspace(xbound[1][0], xbound[1][1], n_steps)
    X,Y = np.meshgrid(x_1, x_2)
    x_matrix = np.column_stack((X.ravel(),Y.ravel()))
    # safeopt.linearly_spaced_combinations returns x_matrix[:, ::-1]
    # Another option would be np.apply_along_axis(function, 1, array)
    y = np.apply_along_axis(fun, 1, x_matrix)
    return x_1, x_2, x_matrix, y

def estimate_lipschitz(f, xbound, n_steps):
    x_1, x_2, x_matrix, y = eval_on_grid(f, xbound=xbound, n_steps = n_steps)
    g1, g2 = np.gradient(y.reshape(n_steps, n_steps), x_1, x_2)
    return max(np.abs(g1).max(), np.abs(g2).max())

class Problem:
    def __init__(self, fun, bounds, percentile, default_safe_seeds):
        self.fun = fun
        self.bounds = bounds
        self.xdim = len(bounds)
        assert percentile >= 0 and percentile < 0.9
        self.percentile = percentile
        n_steps = 500
        self.x_1, self.x_2, self.x_matrix, self.y = eval_on_grid(fun, bounds, n_steps)
        self.safe_threshold = np.quantile(self.y, percentile)
        print(f'Safe Threshold ({self.percentile}) = {self.safe_threshold}')
        self.lipschitz = estimate_lipschitz(fun, bounds, n_steps)
        print(f'Lipschitz constant = {self.lipschitz}')
        self.opt_x = None
        self.opt_y = None
        self.default_safe_seeds = default_safe_seeds
        self._init_counters()
        
    def _init_counters(self):
        self.n_evaluations = 0
        self.n_unsafe = 0
    
    def is_safe(self, y):
        return y >= self.safe_threshold
        
    def _calculate_optimal(self):
        assert self.opt_x is None
        opt_pos = np.argmax(self.y)
        self.opt_y = self.y[opt_pos] 
        self.opt_x = self.x_matrix[opt_pos,:]
        assert self.opt_y == self.fun(self.opt_x)
        assert self.is_safe(self.opt_y)
        
    def get_optimal_x(self):
        if self.opt_x is None:
            self._calculate_optimal()
        return self.opt_x
    
    def get_optimal_y(self):
        if self.opt_y is None:
            self._calculate_optimal()
        return self.opt_y

    def get_default_safe_seeds(self, n=1):
        x_safe_seed = self.x_matrix[self.default_safe_seeds[:n],:]
        y_safe_seed = self.y[self.default_safe_seeds[:n]]
        y_safe_seed = y_safe_seed[:,None]
        print(f'Safe seeds:\n X = {x_safe_seed}\n y = {y_safe_seed}')
        assert np.all(self.is_safe(y_safe_seed))
        return x_safe_seed, y_safe_seed
        
    def get_uniform_safe_seeds(self, rng, n):
        safe_region = (self.y > self.safe_threshold) & (self.y < np.quantile(self.y, 0.9))
        safe_idx = np.where(safe_region)[0]
        safe_idx = rng.choice(safe_idx, size = n, replace=False)
        x_safe_seed = self.x_matrix[safe_idx,:]
        y_safe_seed = self.y[safe_idx]
        y_safe_seed = y_safe_seed[:,None]
        print(f'Safe seeds:\n X = {x_safe_seed}\n y = {y_safe_seed}\n idx = {safe_idx}')
        assert np.all(self.is_safe(y_safe_seed))
        return x_safe_seed, y_safe_seed 
        
    def __call__(self, x):
        self.n_evaluations += 1
        y = self.fun(x)
        self.n_unsafe += int(~self.is_safe(y))
        return y
