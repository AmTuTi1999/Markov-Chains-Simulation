# import os
from src.pipelines.context_tree_kernel_pipeline import run_gallo_simulation


if __name__ == "__main__":

    # ---- basic setup ----
    window = (0, 10)
    alphabet = [+1, -1]
    reference_string = [+1, +1]


    # ---- max context depth ----
    max_depth = 50


    # ---- decay parameter sweep ----
    alphas = [0.0052 * i for i in range(1, 10)]

    run_gallo_simulation(
        window=window,
        args={
            "alphas": alphas,
            "alphabet": alphabet,
            "reference_string": reference_string,
        },
        max_depth=max_depth
    )
