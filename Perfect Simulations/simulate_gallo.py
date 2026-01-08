# import os
from src.pipelines.context_tree_kernel_pipeline import run_gallo_simulation


if __name__ == "__main__":

    # ---- basic setup ----
    window = (2, 12)
    alphabet = [+1, -1]
    reference_string = [+1, +1, +1]


    # ---- max context depth ----
    max_depth = 20




    # ---- decay parameter sweep ----
    alphas = [0.1 * i for i in range(1, 10)]

    for alpha in alphas:
        print(f"\nRunning simulation with alpha = {alpha}")



        run_gallo_simulation(
            window=window,
            alpha=alpha,
            epsilon=0.2,
            alphabet=alphabet,
            reference_string=reference_string,
            max_depth=max_depth
        )
