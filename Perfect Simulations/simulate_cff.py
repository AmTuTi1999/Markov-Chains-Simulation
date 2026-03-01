from src.pipelines.continuous_kernel_pipelines import run_cff_simulation
from src.utils.decay import InfinitePolynomialTheta, InfiniteExponentialTheta




if __name__ == "__main__":
    window = (0,5)
    max_regen_search_depth = 5000
    theta_args_poly = {
    "theta_generator": InfinitePolynomialTheta,
    "theta0": 0.00000001,
    "alphas":[0.9], 
    "rhos":[11],
    'decay_type': 'polynomial',
    'C': 2.5,
    }
    
    # Run both truncated and non-truncated
    results = run_cff_simulation(
        window=(0,  0),
        theta_args=theta_args_poly,
        max_regen_search_depth=5000,
        num_validation_samples=10000000,
        validate_with_simulation=True,
        run_truncated=True,
        run_non_truncated=True,
    )
    

    theta_args_exp = {
    "theta_generator": InfiniteExponentialTheta,
    "theta0": 0.00000001,
    "alphas":[0.9], 
    "rhos":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'decay_type': 'exponential',
    'C': 2.5,
    }
    
    # Run both truncated and non-truncated
    results = run_cff_simulation(
        window=(0, 0),
        theta_args=theta_args_exp,
        max_regen_search_depth=5000,
        num_validation_samples=10000000,
        validate_with_simulation=True,
        run_truncated=True,
        run_non_truncated=True,
    )
    