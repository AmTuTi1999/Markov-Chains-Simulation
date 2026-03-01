from src.pipelines.context_tree_kernel_pipeline import run_gallo_simulation
import numpy as np


if __name__ == "__main__":
    args = {
        "alphabet": [-1, 1],
        "reference_string": [-1, 1],
        "epsilon": 0.3,
        "beta": 0.7,
        "alphas": np.linspace(0.0005, 0.020, 10),
    }

    print("\n" + "=" * 80)
    print("GALLO CONTEXT-TREE EXPERIMENTS (MC VALIDATION USES SIMULATED CHAIN)")
    print("=" * 80)

    # Practical notes:
    # - max_trie_depth controls the size of the constructed context set τ (can explode).
    # - max_depth is ONLY the truncation S used in bounds + truncated MC min(L', S).
    # - For non-truncated MC, increase burn-in if you see many infinities dropped.
    results = run_gallo_simulation(
        window=(0, 0),
        args=args,
        max_trie_depth=5,            
        max_depth=5000,              
        num_validation_samples=10000000,
        validate_with_simulation=True,
        run_truncated=True,
        run_non_truncated=True,
        mc_burn_in=20_000,           
        mc_seed=123,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


