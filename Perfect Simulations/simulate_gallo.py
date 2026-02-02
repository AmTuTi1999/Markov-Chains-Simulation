# import os
from src.pipelines.context_tree_kernel_pipeline import run_gallo_simulation
import numpy as np


if __name__ == "__main__":
    # Configuration matching your thesis
    args = {
        'alphabet': [-1, 1],
        'reference_string': [-1, 1],
        'epsilon': 0.3,  # Minimum transition probability
        'beta': 0.7,     # AR coefficient decay: a_i = exp(-i^β)
        'alphas': np.linspace(0.005, 0.09, 10),  # Growth parameter for lag function
    }
    
    print("\n" + "="*80)
    print("GALLO PERFECT SIMULATION EXPERIMENTS")
    print("="*80)
    
    # Main experiment
    results = run_gallo_simulation(
        window=(0, 10),
        args=args,
        max_trie_depth=5,       # Context tree depth (exponential complexity!)
        max_depth=5000,              # Truncation index
        num_validation_samples=10000,  # MC samples
        validate_with_simulation=True,
        run_truncated=True,
        run_non_truncated=True,
    )
    
    # Optional: Run comparative study
    print("\n" + "="*80)
    print("RUNNING COMPARATIVE STUDY (OPTIONAL)")
    print("="*80)

