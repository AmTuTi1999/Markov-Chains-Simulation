import time
import os


from src.simulators.gallo import GalloSimulator
from src.utils.visualize import plot_and_save_figure

def run_gallo_simulation(
    window: tuple,
    args: dict,
    max_depth: int = 8,
):
    start_time = time.time()
    alphas = args.get("alphas")
    lookback, time_list = [], []
    for alpha in alphas:
        
        sim = GalloSimulator(
            alpha=alpha,
            epsilon=args['epsilon'],
            alphabet=args['alphabet'],
            reference_string=args['reference_string'],
            max_depth=max_depth,
        )
        regen_time, perfect_samples = sim.perfect_sample(
            window=window,
        )
        lookback.append(sim.analytic_lookback_expectation())
        time_list.append((alpha, regen_time))
        elapsed_time = time.time() - start_time
        print(
            f"Alpha: {alpha}, Lookback Expectation: {lookback[-1]:.2f}, Regeneration Time: {regen_time:.2f}s, Total Time: {elapsed_time:.2f}s"
        )
    filename = os.path.join("results", "gallo", "Gallo_Lookback_vs_alpha.png")
        # If an absolute path was provided, make it relative to current working directory
    if os.path.isabs(filename):
        filename = os.path.join(os.getcwd(), filename.lstrip(os.sep))
    parent_dir = os.path.dirname(filename)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    fig = plot_and_save_figure(
        x={ 0: [(alpha, lookback[i]) for i, alpha in enumerate(alphas)] },
        y={ 0: [(alpha, lookback[i]) for i, alpha in enumerate(alphas)] },
        z=None,
        title="Gallo Lookback Expectation vs Alpha",
        xlabel="Alpha",
        ylabel="Lookback Expectation",
        filename=filename,
    )

    fig.savefig(filename)
    print(f"Figure saved to {filename}")