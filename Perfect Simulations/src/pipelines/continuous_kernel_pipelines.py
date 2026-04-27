import time
import os

from src.simulators.cff import BinaryAutoregressiveSimulator
from src.utils.visualize import (
    plot_and_save_figure,
)


def run_cff_simulation(
    window: tuple,
    theta_args: dict,
    max_regen_search_depth: int = 5000,
    num_validation_samples: int = 100000000,
    validate_with_simulation: bool = True,
    run_truncated: bool = True,
    run_non_truncated: bool = False,
):
    """
    Run CFF simulation experiments and generate comparison plots.
    
    Args:
        window: Time window for perfect sampling
        theta_args: Dictionary with theta sequence parameters
        max_regen_search_depth: Truncation index S (for truncated version)
        num_validation_samples: Number of MC samples for validation
        validate_with_simulation: If True, validate analytical bounds with MC
        run_truncated: If True, run truncated perfect simulation analysis
        run_non_truncated: If True, run non-truncated perfect simulation analysis
    """
    theta_generator = theta_args.get("theta_generator")
    theta0 = theta_args.get("theta0", 0.00000001)
    alphas = theta_args.get("alphas")
    rhos = theta_args.get("rhos")
    decay_type = theta_args.get("decay_type", 'exponential')
    C_constant = theta_args.get("C", 2.5)
    
    # Results storage - organized by truncated/non-truncated
    results = {
        'truncated': {} if run_truncated else None,
        'non_truncated': {} if run_non_truncated else None,
    }
    
    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================
    
    theta_seq = theta_generator(alpha=0.9, rho=3)
    
    # Initialize simulator
    start_time = time.time()
    sim = BinaryAutoregressiveSimulator(
        theta0=theta0,
        theta_seq=theta_seq,
        max_regen_search_depth=max_regen_search_depth,
        show_progress=False
    )
    perfect_samples = sim.perfect_sample()

    print(perfect_samples)
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"ALPHA = {alpha:.2f}")
        print(f"{'='*70}")
        
        # Initialize storage for this alpha
        if run_truncated:
            results['truncated'][alpha] = {
                'times': [],
                'analytical_bounds': [],
                'empirical_bounds': [],
                'mu_S_analytical': [],
                'mu_S_empirical': [],
                'tail_bounds': [],
                'user_impatience': [],
                'prob_exceed_S': [],
                'tightness': [],
            }
        
        if run_non_truncated:
            results['non_truncated'][alpha] = {
                'times': [],
                'theoretical_bounds': [],
                'exact_values': [],
                'empirical_means': [],
                'tightness': [],
                'ci_low': [],
                'ci_high': [],
                'empirical_std_error': [],
            }
        
        for rho in rhos:
            print(f"\nrho = {rho:.2f}")
            
            # Create theta sequence
            theta_seq = theta_generator(alpha=alpha, rho=rho)
            
            # Initialize simulator
            start_time = time.time()
            sim = BinaryAutoregressiveSimulator(
                theta0=theta0,
                theta_seq=theta_seq,
                max_regen_search_depth=max_regen_search_depth,
                show_progress=False
            )
            
            # ========================================================
            # TRUNCATED PERFECT SIMULATION
            # ========================================================
            if run_truncated:
                print("  [Truncated Analysis]")
                
                # Analytical bound
                trunc_analytical = sim.analytical_lookback_bound(
                    truncated=True,
                    decay_type=decay_type,
                    C=C_constant,
                    param=rho
                )
                
                # Validation
                if validate_with_simulation:
                    trunc_validation = sim.validate_analytical_bound(
                        num_samples=num_validation_samples,
                        truncated=True,
                        decay_type=decay_type,
                        C=C_constant,
                        param=rho
                    )
                    empirical_mu_S = trunc_validation['empirical_mean']
                    empirical_total = empirical_mu_S + trunc_analytical['tail_bound']
                else:
                    empirical_mu_S = None
                    empirical_total = None
                
                # Store results
                results['truncated'][alpha]['times'].append((rho, time.time() - start_time))
                results['truncated'][alpha]['analytical_bounds'].append(
                    (rho, trunc_analytical['total_bound'])
                )
                results['truncated'][alpha]['mu_S_analytical'].append(
                    (rho, trunc_analytical['mu_S_analytical'])
                )
                results['truncated'][alpha]['tail_bounds'].append(
                    (rho, trunc_analytical['tail_bound'])
                )
                results['truncated'][alpha]['user_impatience'].append(
                    (rho, sim.compute_user_impatience_bias_given_limit())
                )
                results['truncated'][alpha]['prob_exceed_S'].append(
                    (rho, trunc_analytical['prob_exceed_S'])
                )

                
                if validate_with_simulation:
                    results['truncated'][alpha]['empirical_bounds'].append(
                        (rho, empirical_total)
                    )
                    results['truncated'][alpha]['mu_S_empirical'].append(
                        (rho, empirical_mu_S)
                    )
                    results['truncated'][alpha]['tightness'].append(
                    (rho, trunc_validation['tightness'])
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
                    decay_type=decay_type,
                    C=C_constant,
                    param=rho,
                    compute_exact=True
                )
                
                # Validation (optional, can be slow for non-truncated)
                if validate_with_simulation:
                    non_trunc_validation = sim.validate_analytical_bound(
                        num_samples=num_validation_samples,
                        truncated=False,
                        decay_type=decay_type,
                        C=C_constant,
                        param=rho
                    )
                    empirical_mean = non_trunc_validation['empirical_mean']
                else:
                    empirical_mean = None
                
                # Store results
                results['non_truncated'][alpha]['times'].append(
                    (rho, time.time() - start_time)
                )
                results['non_truncated'][alpha]['theoretical_bounds'].append(
                    (rho, non_trunc_analytical['theoretical_bound'])
                )
                results['non_truncated'][alpha]['exact_values'].append(
                    (rho, non_trunc_analytical['exact_value'])
                )

                
                if validate_with_simulation:
                    results['non_truncated'][alpha]['empirical_means'].append(
                        (rho, empirical_mean)
                    )
                    results['non_truncated'][alpha]['tightness'].append(
                    (rho, non_trunc_validation['tightness'])
                    )
                    results['non_truncated'][alpha]['ci_low'].append(
                        (rho, non_trunc_validation.get('ci_low', float('nan')))
                    )
                    results['non_truncated'][alpha]['ci_high'].append(
                        (rho, non_trunc_validation.get('ci_high', float('nan')))
                    )
                    results['non_truncated'][alpha]['empirical_std_error'].append(
                        (rho, non_trunc_validation.get('empirical_std_error', float('nan')))
                    )
                
                # Print summary
                print(f"    Theoretical bound: {non_trunc_analytical['theoretical_bound']:.2f}")
                print(f"    Exact E[L_n]:      {non_trunc_analytical['exact_value']:.2f}")
                
                if validate_with_simulation:
                    print(f"    Empirical E[L_n]:  {empirical_mean:.2f}")
                    print(f"    Tightness:         {non_trunc_validation['tightness']*100:.1f}%")
                    print(f"    95% CI:            [{non_trunc_validation['ci_low']:.2f}, "
                          f"{non_trunc_validation['ci_high']:.2f}]")
    
    # ========================================================
    # GENERATE PLOTS AND SAVE RESULTS
    # ========================================================
    
    filename_suffix = f"{theta_generator.name}_{decay_type}"
    results_dir = os.path.join("results", "cff")
    os.makedirs(results_dir, exist_ok=True)
    
    # --------------------------------------------------------
    # TRUNCATED PLOTS
    # --------------------------------------------------------
    if run_truncated:
        _generate_truncated_plots(
            results['truncated'],
            results_dir,
            filename_suffix,
            max_regen_search_depth,
            decay_type,
            validate_with_simulation
        )
        _save_truncated_results(
            results['truncated'],
            results_dir,
            filename_suffix,
            max_regen_search_depth,
            decay_type,
            C_constant,
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
            C_constant,
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
            max_regen_search_depth,
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
            xlabel=r"$\rho$ (decay parameter)",
            ylabel=r"Expected Lookback $\mathbb{E}[L_n]$",
            title=f"Truncated: Empirical vs Analytical ({decay_type}, S={S})",
            label_1=r"Empirical $\hat{\mu}_S$",
            label_2=r"Analytical $\mu_S$",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    # Plot 2: Analytical Bound Only
    filename = os.path.join(results_dir, f"Truncated_Analytical_Bound_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['analytical_bounds'] for alpha, data in trunc_results.items()},
        z=None,
        xlabel=r"$\rho$ (decay parameter)",
        ylabel=r"$\mathbb{E}[L_n]$ Upper Bound",
        title=f"Truncated Analytical Bound ({decay_type}, S={S})",
        label_1=r"Analytical $\mu_S$",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Plot 3: Components (μ_S vs Tail)
    filename = os.path.join(results_dir, f"Truncated_Components_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['mu_S_analytical'] for alpha, data in trunc_results.items()},
        z={alpha: data['tail_bounds'] for alpha, data in trunc_results.items()},
        xlabel=r"$\rho$ (decay parameter)",
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
        xlabel=r"$\rho$ (decay parameter)",
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
        xlabel=r"$\rho$ (decay parameter)",
        ylabel=r"$P(L_n > S)$",
        title=f"Probability of Exceeding Truncation (S={S})",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

    # Plot 6: Tightness Ratio
    filename = os.path.join(results_dir, f"Truncated_Tightness_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['tightness'] for alpha, data in trunc_results.items()},
        z=None,
        xlabel=r"$\rho$ (decay parameter)",
        ylabel=r"Tightness = Empirical / Theoretical",
        title=f"Bound Tightness ({decay_type})",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

def _save_truncated_results(
    trunc_results,
    results_dir,
    filename_suffix,
    S,
    decay_type,
    C,
    num_samples,
    validate
):
    """Save numerical results for truncated perfect simulation."""
    
    results_file = os.path.join(results_dir, f"truncated_numerical_results_{filename_suffix}.txt")
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRUNCATED PERFECT SIMULATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Truncation index S: {S}\n")
        f.write(f"  Decay type: {decay_type}\n")
        f.write(f"  Constant C: {C}\n")
        f.write(f"  Validation samples: {num_samples if validate else 'N/A'}\n\n")
        
        for alpha, data in trunc_results.items():
            f.write(f"\nAlpha = {alpha:.2f}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'rho':<10} {'μ_S':<14} {'Tail':<14} {'Total':<14} "
                   f"{'P(L>S)':<14} {'Bias':<14}\n")
            f.write("-" * 80 + "\n")
            
            for i, (rho, _) in enumerate(data['analytical_bounds']):
                mu_S = data['mu_S_analytical'][i][1]
                tail = data['tail_bounds'][i][1]
                total = data['analytical_bounds'][i][1]
                prob = data['prob_exceed_S'][i][1]
                bias = data['user_impatience'][i][1]
                
                f.write(f"{rho:<10.3f} {mu_S:<14.4f} {tail:<14.2e} {total:<14.4f} "
                       f"{prob:<14.2e} {bias:<14.2e}\n")
            
            if validate:
                f.write("\nValidation (Empirical vs Analytical):\n")
                f.write(f"{'rho':<10} {'μ_S (emp)':<14} {'μ_S (anal)':<14} {'Discrepancy':<14}\n")
                f.write("-" * 80 + "\n")
                for i, (rho, _) in enumerate(data['mu_S_empirical']):
                    mu_emp = data['mu_S_empirical'][i][1]
                    mu_anal = data['mu_S_analytical'][i][1]
                    disc = mu_anal - mu_emp
                    f.write(f"{rho:<10.3f} {mu_emp:<14.4f} {mu_anal:<14.4f} {disc:<14.4f}\n")
    
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
        xlabel=r"$\rho$ (decay parameter)",
        ylabel=r"Expected Lookback $\mathbb{E}[L_n]$",
        title=f"Non-Truncated: Exact vs Theoretical Bound ({decay_type})",
        label_1="Exact (summation)",
        label_2="Theoretical (Theorem 5.1.22)",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Plot 2: Tightness Ratio
    filename = os.path.join(results_dir, f"NonTruncated_Tightness_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['tightness'] for alpha, data in non_trunc_results.items()},
        z=None,
        xlabel=r"$\rho$ (decay parameter)",
        ylabel=r"Tightness = Empirical Estimation / Theoretical Bound",
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
            z={alpha: data['theoretical_bounds'] for alpha, data in non_trunc_results.items()},
            xlabel=r"$\rho$ (decay parameter)",
            ylabel=r"$\mathbb{E}[L_n]$",
            title=f"Non-Truncated: Empirical Estimation vs Theoretical Bounds ({decay_type})",
            label_1="Empirical Estimation(MC)",
            label_2="Theoretical Bounds(Theorem 5.1.22)",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")


def _save_non_truncated_results(
    non_trunc_results,
    results_dir,
    filename_suffix,
    decay_type,
    C,
    num_samples,
    validate
):
    """Save numerical results for non-truncated perfect simulation."""
    
    results_file = os.path.join(results_dir, f"non_truncated_numerical_results_{filename_suffix}.txt")
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NON-TRUNCATED PERFECT SIMULATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Decay type: {decay_type}\n")
        f.write(f"  Constant C: {C}\n")
        f.write(f"  Validation samples: {num_samples if validate else 'N/A'}\n\n")
        
        for alpha, data in non_trunc_results.items():
            f.write(f"\nAlpha = {alpha:.2f}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'rho':<10} {'Exact E[L_n]':<16} {'Theoretical':<16} "
                   f"{'Tightness':<16}\n")
            f.write("-" * 80 + "\n")
            print(data.keys())
            for i, (rho, _) in enumerate(data['exact_values']):
                exact = data['exact_values'][i][1]
                theoretical = data['theoretical_bounds'][i][1]
                tightness = data['tightness'][i][1]

                
                f.write(f"{rho:<10.3f} {exact:<16.4f} {theoretical:<16.4f} "
                       f"{tightness:<16.2%}\n")
            
            if validate:
                
                f.write("\nValidation (Empirical vs Exact):\n")
                f.write(f"{'rho':<10} {'Empirical':<16} {'Exact':<16} {'Discrepancy':<16} {'CI_Low':<16} {'CI_High':<16} {'Std Error':<16}\n")
                f.write("-" * 80 + "\n")
                for i, (rho, _) in enumerate(data['empirical_means']):
                    emp = data['empirical_means'][i][1]
                    exact = data['exact_values'][i][1]
                    ci_low = data['ci_low'][i][1] if 'ci_low' in data else float('nan')
                    ci_high = data['ci_high'][i][1] if 'ci_high' in data else float('nan')
                    std_error = data['empirical_std_error'][i][1] if 'empirical_std_error' in data else float('nan')
                    disc = exact - emp
                    f.write(f"{rho:<10.3f} {emp:<16.4f} {exact:<16.4f} {disc:<16.4f} {ci_low:<16.4f} {ci_high:<16.4f} {std_error:<16.4f}\n")
    
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
        xlabel=r"$\rho$ (decay parameter)",
        ylabel=r"Expected Lookback $\mathbb{E}[L_n]$",
        title=f"Truncated (S={S}) vs Non-Truncated ({decay_type})",
        label_1=f"Truncated μ_S (S={S})",
        label_2="Non-Truncated (exact)",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")