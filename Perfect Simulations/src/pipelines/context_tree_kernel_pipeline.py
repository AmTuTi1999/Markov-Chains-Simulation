import time
import os
from src.simulators.gallo import GalloContextTreeSimulator
from src.utils.visualize import plot_and_save_figure

def run_gallo_simulation(
    window: tuple,
    args: dict,
    max_trie_depth: int = 8,
    max_depth: int = 50,  # Truncation index S
    num_validation_samples: int = 10000,
    validate_with_simulation: bool = True,
    run_truncated: bool = True,
    run_non_truncated: bool = False,
):
    """
    Run Gallo perfect simulation experiments matching CFF structure.
    
    Args:
        window: (m, n) - finite window for simulation
        args: dict with 'alphas', 'alphabet', 'reference_string', 'epsilon', 'beta'
        max_trie_depth: Maximum depth for context tree construction
        max_depth: Truncation index S for lookback expectation
        num_validation_samples: Number of MC samples for validation
        validate_with_simulation: If True, validate with MC sampling
        run_truncated: If True, run truncated analysis
        run_non_truncated: If True, run non-truncated analysis
    """
    # Extract parameters
    alphas = args.get("alphas")
    alphabet = args.get("alphabet")
    reference_string = args.get("reference_string")
    epsilon = args.get("epsilon", 0.3)
    beta = args.get("beta", 0.7)
    
    # Results storage
    results = {
        'truncated': {} if run_truncated else None,
        'non_truncated': {} if run_non_truncated else None,
    }
    
    print("=" * 80)
    print("GALLO EXPERIMENT: Context Tree Perfect Simulation")
    print(f"Window: [{window[0]}, {window[1]}], Truncation S={max_depth}")
    print(f"Reference string: {reference_string}, Epsilon: {epsilon}")
    print(f"Beta (AR decay): {beta}, Max trie depth: {max_trie_depth}")
    print(f"Alpha values: {alphas}")
    print("=" * 80)
    
    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================
    
    for kappa in [0.0]:
        print(f"\n{'='*70}")
        print(f"ALPHA = {kappa:.3f}")
        print(f"{'='*70}")
        
        # Initialize storage for this alpha
        if run_truncated:
            results['truncated'][kappa] = {
                'times': [],
                'analytical_bounds': [],
                'empirical_bounds': [],
                'mu_S_analytical': [],
                'mu_S_empirical': [],
                'tail_bounds': [],
                'user_impatience': [],
                'prob_exceed_S': [],
            }
        
        if run_non_truncated:
            results['non_truncated'][kappa] = {
                'times': [],
                'theoretical_bounds': [],
                'exact_values': [],
                'empirical_means': [],
                'tightness': [],
            }
        
        # Initialize simulator
        start_time = time.time()
        
        # ========================================================
        # TRUNCATED PERFECT SIMULATION
        # ========================================================
        for alpha in alphas:
            print(f"  [Alpha = {alpha:.3f}]")
            
            # Initialize simulator
            sim = GalloContextTreeSimulator(
                alpha=alpha,
                alphabet=alphabet,
                reference_string=reference_string,
                epsilon=epsilon,
                beta=beta,
                max_depth=max_depth,
                max_trie_depth=max_trie_depth,
                show_progress=False
            )
            
            if run_truncated:
                print("  [Truncated Analysis]")
                
                # Analytical bound
                trunc_analytical = sim.analytical_lookback_bound(truncated=True)
            
                # Validation
                if validate_with_simulation:
                    trunc_validation = sim.validate_analytical_bound(
                        num_samples=num_validation_samples,
                        truncated=True,
                    )
                    empirical_mu_S = trunc_validation['empirical_mean']
                    empirical_total = empirical_mu_S + trunc_analytical['tail_bound']
                else:
                    empirical_mu_S = None
                    empirical_total = None
                
                # Store results
                results['truncated'][kappa]['times'].append(
                    (alpha, time.time() - start_time)
                )
                results['truncated'][kappa]['analytical_bounds'].append(
                    (alpha, trunc_analytical['total_bound'])
                )
                results['truncated'][kappa]['mu_S_analytical'].append(
                    (alpha, trunc_analytical['mu_S_analytical'])
                )
                results['truncated'][kappa]['tail_bounds'].append(
                    (alpha, trunc_analytical['tail_bound'])
                )
                results['truncated'][kappa]['user_impatience'].append(
                    (alpha, sim.compute_user_impatience_bias_given_limit())
                )
                results['truncated'][kappa]['prob_exceed_S'].append(
                    (alpha, trunc_analytical['prob_exceed_S'])
                )
                
                if validate_with_simulation:
                    results['truncated'][kappa]['empirical_bounds'].append(
                        (alpha, empirical_total)
                    )
                    results['truncated'][kappa]['mu_S_empirical'].append(
                        (alpha, empirical_mu_S)
                    )
                
                # Print summary
                print(f"    Truncated E[L_n] ≤ {trunc_analytical['total_bound']:.2f}")
                print(f"    μ_S = {trunc_analytical['mu_S_analytical']:.2f}, "
                    f"Tail = {trunc_analytical['tail_bound']:.2e}")
                if validate_with_simulation:
                    print(f"    Empirical μ_S = {empirical_mu_S:.2f} "
                        f"(discrepancy: {trunc_validation['discrepancy']:.2f})")
            
            # ========================================================
            # NON-TRUNCATED PERFECT SIMULATION
            # ========================================================
            if run_non_truncated:
                print("  [Non-Truncated Analysis]")
                
                # Analytical bound
                non_trunc_analytical = sim.analytical_lookback_bound(
                    truncated=False,
                    compute_exact=True
                )
                
                # Validation (optional, can be slow for non-truncated)
                if validate_with_simulation:
                    non_trunc_validation = sim.validate_analytical_bound(
                        num_samples=num_validation_samples,
                        truncated=False,
                    )
                    empirical_mean = non_trunc_validation['empirical_mean']
                else:
                    empirical_mean = None
                
                # Store results
                results['non_truncated'][kappa]['times'].append(
                    (alpha, time.time() - start_time)
                )
                results['non_truncated'][kappa]['theoretical_bounds'].append(
                    (alpha, non_trunc_analytical['theoretical_bound'])
                )
                results['non_truncated'][kappa]['exact_values'].append(
                    (alpha, non_trunc_analytical['exact_value'])
                )
                results['non_truncated'][kappa]['tightness'].append(
                    (alpha, non_trunc_analytical['tightness'])
                )
                
                if validate_with_simulation:
                    results['non_truncated'][kappa]['empirical_means'].append(
                        (alpha, empirical_mean)
                    )
                
                # Print summary
                print(f"    Theoretical bound: {non_trunc_analytical['theoretical_bound']:.2f}")
                print(f"    Exact E[L_n]:      {non_trunc_analytical['exact_value']:.2f}")
                print(f"    Tightness:         {non_trunc_analytical['tightness']*100:.1f}%")
                if validate_with_simulation:
                    print(f"    Empirical E[L_n]:  {empirical_mean:.2f}")
        
    # ========================================================
    # GENERATE PLOTS AND SAVE RESULTS
    # ========================================================
    
    filename_suffix = f"Gallo_Alpha_{min(alphas):.3f}_to_{max(alphas):.3f}"
    decay_type = f"context_tree_beta_{beta}"
    
    results_dir = os.path.join("results", "gallo")
    os.makedirs(results_dir, exist_ok=True)
    
    # --------------------------------------------------------
    # TRUNCATED PLOTS
    # --------------------------------------------------------
    if run_truncated:
        _generate_truncated_plots(
            results['truncated'],
            results_dir,
            filename_suffix,
            max_depth,
            decay_type,
            validate_with_simulation
        )
        _save_truncated_results(
            results['truncated'],
            results_dir,
            filename_suffix,
            max_depth,
            decay_type,
            num_validation_samples,
            validate_with_simulation
        )
    
    # --------------------------------------------------------
    # NON-TRUNCATED PLOTS
    # --------------------------------------------------------
    if run_non_truncated:
        _generate_non_truncated_plots(
            results['non_truncated'],
            results_dir,
            filename_suffix,
            decay_type,
            validate_with_simulation
        )
        _save_non_truncated_results(
            results['non_truncated'],
            results_dir,
            filename_suffix,
            decay_type,
            num_validation_samples,
            validate_with_simulation
        )
    
    # --------------------------------------------------------
    # COMPARISON PLOT (if both run)
    # --------------------------------------------------------
    if run_truncated and run_non_truncated:
        _generate_comparison_plot(
            results['truncated'],
            results['non_truncated'],
            results_dir,
            filename_suffix,
            max_depth,
            decay_type
        )
    
    return results


# ============================================================
# HELPER FUNCTIONS: TRUNCATED PLOTS
# ============================================================

def _generate_truncated_plots(
    trunc_results,
    results_dir,
    filename_suffix,
    S,
    decay_type,
    validate
):
    """Generate all plots for truncated perfect simulation."""
    
    # Extract times dict for x-axis
    times_dct = {alpha: data['times'] for alpha, data in trunc_results.items()}
    
    # Plot 1: Empirical vs Analytical (if validation enabled)
    if validate:
        filename = os.path.join(results_dir, f"Truncated_Empirical_vs_Analytical_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=times_dct,
            y={alpha: data['empirical_bounds'] for alpha, data in trunc_results.items()},
            z={alpha: data['analytical_bounds'] for alpha, data in trunc_results.items()},
            xlabel=r"$\alpha$ (growth parameter)",
            ylabel=r"Expected Lookback $\mathbb{E}[L_n]$",
            title=f"Truncated: Empirical vs Analytical ({decay_type}, S={S})",
            label_1="Empirical (MC validation)",
            label_2="Analytical (exact)",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    # Plot 2: Analytical Bound Only
    filename = os.path.join(results_dir, f"Truncated_Analytical_Bound_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['analytical_bounds'] for alpha, data in trunc_results.items()},
        z=None,
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"$\mathbb{E}[L_n]$ Upper Bound",
        title=f"Truncated Analytical Bound ({decay_type}, S={S})",
        label_1="Analytical Bound",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Plot 3: Components (μ_S vs Tail)
    filename = os.path.join(results_dir, f"Truncated_Components_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['mu_S_analytical'] for alpha, data in trunc_results.items()},
        z={alpha: data['tail_bounds'] for alpha, data in trunc_results.items()},
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel="Bound Component",
        title=f"Truncated vs Tail Contributions ({decay_type}, S={S})",
        label_1=r"$\mu_S = \mathbb{E}[\psi_S(L_n)]$",
        label_2=r"$\mathbb{E}[(L_n - S)_+]$ (tail)",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Plot 4: User Impatience Bias
    filename = os.path.join(results_dir, f"Truncated_User_Impatience_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['user_impatience'] for alpha, data in trunc_results.items()},
        z=None,
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"Bias = $P(L_n > S) / P(L_n \leq S)$",
        title=f"User Impatience Bias (S={S})",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Plot 5: P(L_n > S)
    filename = os.path.join(results_dir, f"Truncated_Prob_Exceed_S_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['prob_exceed_S'] for alpha, data in trunc_results.items()},
        z=None,
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"$P(L_n > S)$",
        title=f"Probability of Exceeding Truncation (S={S})",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def _save_truncated_results(
    trunc_results,
    results_dir,
    filename_suffix,
    S,
    decay_type,
    num_samples,
    validate
):
    """Save numerical results for truncated perfect simulation."""
    
    results_file = os.path.join(results_dir, f"truncated_numerical_results_{filename_suffix}.txt")
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRUNCATED PERFECT SIMULATION RESULTS (GALLO)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Truncation index S: {S}\n")
        f.write(f"  Decay type: {decay_type}\n")
        f.write(f"  Validation samples: {num_samples if validate else 'N/A'}\n\n")
        
        for alpha, data in trunc_results.items():
            f.write(f"\nAlpha = {alpha:.3f}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'alpha':<10} {'μ_S':<14} {'Tail':<14} {'Total':<14} "
                   f"{'P(L>S)':<14} {'Bias':<14}\n")
            f.write("-" * 80 + "\n")
            
            for i, (alpha_val, _) in enumerate(data['analytical_bounds']):
                mu_S = data['mu_S_analytical'][i][1]
                tail = data['tail_bounds'][i][1]
                total = data['analytical_bounds'][i][1]
                prob = data['prob_exceed_S'][i][1]
                bias = data['user_impatience'][i][1]
                
                f.write(f"{alpha_val:<10.3f} {mu_S:<14.4f} {tail:<14.2e} {total:<14.4f} "
                       f"{prob:<14.2e} {bias:<14.2e}\n")
            
            if validate and len(data['mu_S_empirical']) > 0:
                f.write("\nValidation (Empirical vs Analytical):\n")
                f.write(f"{'alpha':<10} {'μ_S (emp)':<14} {'μ_S (anal)':<14} {'Discrepancy':<14}\n")
                f.write("-" * 80 + "\n")
                for i, (alpha_val, _) in enumerate(data['mu_S_empirical']):
                    mu_emp = data['mu_S_empirical'][i][1]
                    mu_anal = data['mu_S_analytical'][i][1]
                    disc = mu_anal - mu_emp
                    f.write(f"{alpha_val:<10.3f} {mu_emp:<14.4f} {mu_anal:<14.4f} {disc:<14.4f}\n")
    
    print(f"Saved: {results_file}")


# ============================================================
# HELPER FUNCTIONS: NON-TRUNCATED PLOTS
# ============================================================

def _generate_non_truncated_plots(
    non_trunc_results,
    results_dir,
    filename_suffix,
    decay_type,
    validate
):
    """Generate all plots for non-truncated perfect simulation."""
    
    times_dct = {alpha: data['times'] for alpha, data in non_trunc_results.items()}
    
    # Plot 1: Theoretical Bound vs Exact Value
    filename = os.path.join(results_dir, f"NonTruncated_Bound_vs_Exact_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['exact_values'] for alpha, data in non_trunc_results.items()},
        z={alpha: data['theoretical_bounds'] for alpha, data in non_trunc_results.items()},
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"Expected Lookback $\mathbb{E}[L_n]$",
        title=f"Non-Truncated: Exact vs Theoretical Bound ({decay_type})",
        label_1="Exact (summation)",
        label_2="Theoretical (Theorem 6.1.16)",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Plot 2: Tightness Ratio
    filename = os.path.join(results_dir, f"NonTruncated_Tightness_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['tightness'] for alpha, data in non_trunc_results.items()},
        z=None,
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"Tightness = Exact / Theoretical",
        title=f"Bound Tightness ({decay_type})",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Plot 3: Empirical validation (if enabled)
    if validate:
        filename = os.path.join(results_dir, f"NonTruncated_Empirical_Validation_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=times_dct,
            y={alpha: data['empirical_means'] for alpha, data in non_trunc_results.items()},
            z={alpha: data['exact_values'] for alpha, data in non_trunc_results.items()},
            xlabel=r"$\alpha$ (growth parameter)",
            ylabel=r"$\mathbb{E}[L_n]$",
            title=f"Non-Truncated: Empirical vs Exact ({decay_type})",
            label_1="Empirical (MC)",
            label_2="Exact (analytical)",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")


def _save_non_truncated_results(
    non_trunc_results,
    results_dir,
    filename_suffix,
    decay_type,
    num_samples,
    validate
):
    """Save numerical results for non-truncated perfect simulation."""
    
    results_file = os.path.join(results_dir, f"non_truncated_numerical_results_{filename_suffix}.txt")
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NON-TRUNCATED PERFECT SIMULATION RESULTS (GALLO)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Decay type: {decay_type}\n")
        f.write(f"  Validation samples: {num_samples if validate else 'N/A'}\n\n")
        
        for alpha, data in non_trunc_results.items():
            f.write(f"\nAlpha = {alpha:.3f}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'alpha':<10} {'Exact E[L_n]':<16} {'Theoretical':<16} "
                   f"{'Tightness':<16}\n")
            f.write("-" * 80 + "\n")
            
            for i, (alpha_val, _) in enumerate(data['exact_values']):
                exact = data['exact_values'][i][1]
                theoretical = data['theoretical_bounds'][i][1]
                tightness = data['tightness'][i][1]
                
                f.write(f"{alpha_val:<10.3f} {exact:<16.4f} {theoretical:<16.4f} "
                       f"{tightness:<16.2%}\n")
            
            if validate and len(data['empirical_means']) > 0:
                f.write("\nValidation (Empirical vs Exact):\n")
                f.write(f"{'alpha':<10} {'Empirical':<16} {'Exact':<16} {'Discrepancy':<16}\n")
                f.write("-" * 80 + "\n")
                for i, (alpha_val, _) in enumerate(data['empirical_means']):
                    emp = data['empirical_means'][i][1]
                    exact = data['exact_values'][i][1]
                    disc = exact - emp
                    f.write(f"{alpha_val:<10.3f} {emp:<16.4f} {exact:<16.4f} {disc:<16.4f}\n")
    
    print(f"Saved: {results_file}")


# ============================================================
# HELPER FUNCTION: COMPARISON PLOT
# ============================================================

def _generate_comparison_plot(
    trunc_results,
    non_trunc_results,
    results_dir,
    filename_suffix,
    S,
    decay_type
):
    """Generate comparison plot between truncated and non-truncated."""
    
    times_dct = {alpha: data['times'] for alpha, data in trunc_results.items()}
    
    filename = os.path.join(results_dir, f"Truncated_vs_NonTruncated_Comparison_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: trunc_results[alpha]['mu_S_analytical'] for alpha in trunc_results},
        z={alpha: non_trunc_results[alpha]['exact_values'] for alpha in non_trunc_results},
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"Expected Lookback $\mathbb{E}[L_n]$",
        title=f"Truncated (S={S}) vs Non-Truncated ({decay_type})",
        label_1=f"Truncated μ_S (S={S})",
        label_2="Non-Truncated (exact)",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

