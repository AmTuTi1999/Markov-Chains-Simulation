from src.pipelines.continuous_kernel_pipelines import run_cff_simulation
from src.utils.decay import InfinitePolynomialTheta, InfiniteExponentialTheta




if __name__ == "__main__":
    window = (0,5)
    max_regen_search_depth = 5000
    theta_args_exp = {
    "theta_generator": InfinitePolynomialTheta,
    "theta0": 0.00000001,
    "alphas":[0.9], 
    "rhos":[2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    run_cff_simulation(
        window=window, 
        theta_args=theta_args_exp, 
        max_regen_search_depth=max_regen_search_depth,
    )

    theta_args_poly = {
    "theta_generator": InfiniteExponentialTheta,
    "theta0": 0.00000001,
    "alphas":[0.9], 
    "rhos":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    run_cff_simulation(
        window=window, 
        theta_args=theta_args_poly, 
        max_regen_search_depth=max_regen_search_depth,
    )