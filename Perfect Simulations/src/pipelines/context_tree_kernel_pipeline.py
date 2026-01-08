import time


from src.simulators.gallo import GalloSimulator
#from src.utils.visualize import plot_and_save_figure

def run_gallo_simulation(
    window: tuple,
    alpha: float,
    epsilon: float,
    alphabet: list,
    reference_string: list,
    max_depth: int = 8,
):
    start_time = time.time()
    sim = GalloSimulator(
        alpha=alpha,
        epsilon=epsilon,
        alphabet=alphabet,
        reference_string=reference_string,
        max_depth=max_depth,
    )
    perfect_sample, regen_time, _ = sim.perfect_sample(
        window=window,
    )
    elapsed_time = time.time() - start_time

    print(f"Gallo perfect sample in window {window}: {perfect_sample}")
    print(f"Regeneration time: {regen_time}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")