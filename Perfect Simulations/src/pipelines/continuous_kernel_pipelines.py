import time
import os

from src.simulators.cff import BinaryAutoregressiveSimulator
from src.utils.visualize import plot_and_save_figure


def run_cff_simulation(
    window: tuple,
    theta_args: dict,
    max_regen_search_depth: int = 100,
):
    theta_generator = theta_args.get("theta_generator")
    theta0 = theta_args.get("theta0", 0.00000001)
    alphas = theta_args.get("alphas")
    rhos = theta_args.get("rhos")
    
    times_dct = {}  
    regen_times_dct = {}
    biases_dct = {}
    lookback_bound = {}
    actual_biases_dct = {}
    max_regen_search_depth = max_regen_search_depth
    for alpha in alphas:
        time_list, regen_times, biases, lookback, actual_biases = [], [], [], [], []
        for rho in rhos:
            print(f"rho={rho:.1f}")
            
            theta_seq = theta_generator(alpha=alpha, rho=rho) 
            start_time = time.time()
            sim = BinaryAutoregressiveSimulator(theta0=theta0, theta_seq=theta_seq, max_regen_search_depth=max_regen_search_depth)
            perfect_sample, regen_time = sim.perfect_sample(window=window)
            elapsed_time = time.time() - start_time

            time_list.append((rho, elapsed_time))
            regen_times.append((rho, regen_time))
            biases.append((rho, sim.conditional_lookback_expectation()))
            actual_biases.append((rho, sim.compute_user_impatience_bias_given_limit()))
            lookback.append((rho, sim.analytic_lookback_bound()))
        times_dct[alpha] = time_list
        regen_times_dct[alpha] = regen_times
        biases_dct[alpha] = biases
        lookback_bound[alpha] = lookback
        actual_biases_dct[alpha] = actual_biases

    filename_suffix = theta_generator.name
    # Build a project-relative, filesystem-safe path (avoid leading '/' and ':' in name)
    filename = os.path.join("results", "cff", f"Regen_time_vs_rho_{filename_suffix}.png")
        # If an absolute path was provided, make it relative to current working directory
    if os.path.isabs(filename):
        filename = os.path.join(os.getcwd(), filename.lstrip(os.sep))

    parent_dir = os.path.dirname(filename)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    
    fig = plot_and_save_figure(
        x=times_dct,
        y=regen_times_dct,
        z =None,
        title="Regeneration Time vs rho",
        xlabel="rho (upper bound of memory decay)",
        ylabel="Regeneration Time (s)",
    )

    fig.savefig(filename)
    print(f"Figure saved to {filename}")

    filename = os.path.join("results", "cff", f"Lookback_Bound_vs_rho_{filename_suffix}.png")
        # If an absolute path was provided, make it relative to current working directory
    if os.path.isabs(filename):
        filename = os.path.join(os.getcwd(), filename.lstrip(os.sep))
    fig = plot_and_save_figure(
        x=times_dct,
        y=biases_dct,
        z=lookback_bound,
        xlabel="rho (upper bound of memory decay)",
        ylabel="Empirical / Analytic Bound",
        title="Empirical vs Analytic Lookback Expectation Bound",
        label_1="Empirical Bound",
        label_2="Analytic Bound",
    )
    fig.savefig(filename)
    print(f"Figure saved to {filename}")

    filename = os.path.join("results", "cff", f"User_Impatience_Bias_vs_rho_{filename_suffix}.png")
        # If an absolute path was provided, make it relative to current working directory
    if os.path.isabs(filename):
        filename = os.path.join(os.getcwd(), filename.lstrip(os.sep))
    fig = plot_and_save_figure(
        x=times_dct,
        y=actual_biases_dct,
        z=None,
        xlabel="rho (upper bound of memory decay)",
        ylabel="User Impatience Bias",
        title="User Impatience Bias vs rho",
    )
    fig.savefig(filename)
    print(f"Figure saved to {filename}")