import time
import os
import numpy as np
from src.simulators.garcia import GarciaContextTreeSimulator
from src.utils.visualize import (
    plot_and_save_figure,
)

def run_garcia_simulation(
    window: tuple,
    args: dict,
    max_blocks: int = 100,
    max_block_size: int = 50,
    num_validation_samples: int = 1000,
    validate_with_simulation: bool = True,
    run_coalescence_analysis: bool = True,
):
    """
    Run Garcia perfect simulation experiments matching CFF structure.
    
    Args:
        window: (m, n) - finite window for simulation (for compatibility)
        args: dict with 'skeleton', 'alpha_sequence', 'alphabet', 'epsilon'
        max_blocks: Maximum number of blocks to generate
        max_block_size: Maximum size of each block
        num_validation_samples: Number of MC samples for validation
        validate_with_simulation: If True, validate with MC sampling
        run_coalescence_analysis: If True, analyze coalescence times
    """
    # Extract parameters
    skeleton = args.get("skeleton")
    # alpha_sequence = args.get("alpha_sequence")
    alphabet = args.get("alphabet")
    epsilon = args.get("epsilon", 0.1)
    alpha_params = args.get("alpha_params", [0.1, 0.2, 0.3])  # For varying skeleton complexity
    
    # Results storage
    results = {
        'coalescence': {} if run_coalescence_analysis else None,
    }
    
    print("=" * 80)
    print("GARCIA EXPERIMENT: Locally Continuous Perfect Simulation")
    print(f"Window: [{window[0]}, {window[1]}] (compatibility)")
    print(f"Alphabet: {alphabet}, Epsilon: {epsilon}")
    print(f"Max blocks: {max_blocks}, Max block size: {max_block_size}")
    print(f"Skeleton size: {len(skeleton)}")
    print("=" * 80)
    
    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================
    
    for kappa in [0.0]:  # Can be extended for different parameter regimes
        print(f"\n{'='*70}")
        print(f"REGIME = {kappa:.3f}")
        print(f"{'='*70}")
        
        # Initialize storage for this regime
        if run_coalescence_analysis:
            results['coalescence'][kappa] = {
                'times': [],
                'mean_coalescence': [],
                'std_coalescence': [],
                'max_coalescence': [],
                'mean_blocks': [],
                'feasibility_metrics': [],
                'E_bar_theta': [],
                'prod_sum': [],
                'sum_complement': [],
            }
        
        start_time = time.time()
        
        # ========================================================
        # COALESCENCE TIME ANALYSIS
        # ========================================================
        for alpha_param in alpha_params:
            print(f"  [Alpha parameter = {alpha_param:.3f}]")
            
            # Generate alpha sequences based on parameter
            # For each context, α_k^v should converge to 1
            alpha_seq_dict = {}
            for ctx in skeleton:
                # Example: α_k^v = 0.5 + 0.45 * (1 - exp(-alpha_param * k))
                alpha_seq_dict[ctx] = [
                    0.5 + 0.45 * (1 - np.exp(-alpha_param * k)) 
                    for k in range(20)
                ]
            
            # Initialize simulator
            sim = GarciaContextTreeSimulator(
                alphabet=alphabet,
                skeleton=skeleton,
                alpha_sequence=alpha_seq_dict,
                epsilon=epsilon,
                max_blocks=max_blocks,
                max_block_size=max_block_size,
                show_progress=False
            )
            
            if run_coalescence_analysis:
                print("  [Coalescence Time Analysis]")
                
                # Compute expectation bound
                bound_info = sim.expectation_bound_Lambda()
                
                # Validation with simulation
                if validate_with_simulation:
                    validation = sim.validate_perfect_simulation(
                        num_samples=num_validation_samples
                    )
                    mean_coal = validation['mean_coalescence_time']
                    std_coal = validation['std_coalescence_time']
                    max_coal = validation['max_coalescence_time']
                    mean_blocks = validation.get('num_blocks', 0)
                else:
                    mean_coal = None
                    std_coal = None
                    max_coal = None
                    mean_blocks = None
                
                # Store results
                results['coalescence'][kappa]['times'].append(
                    (alpha_param, time.time() - start_time)
                )
                results['coalescence'][kappa]['E_bar_theta'].append(
                    (alpha_param, bound_info['E_bar_theta'])
                )
                results['coalescence'][kappa]['prod_sum'].append(
                    (alpha_param, bound_info['prod_sum'])
                )
                results['coalescence'][kappa]['sum_complement'].append(
                    (alpha_param, bound_info['sum_complement'])
                )
                results['coalescence'][kappa]['feasibility_metrics'].append(
                    (alpha_param, bound_info['feasible'])
                )
                
                if validate_with_simulation:
                    results['coalescence'][kappa]['mean_coalescence'].append(
                        (alpha_param, mean_coal)
                    )
                    results['coalescence'][kappa]['std_coalescence'].append(
                        (alpha_param, std_coal)
                    )
                    results['coalescence'][kappa]['max_coalescence'].append(
                        (alpha_param, max_coal)
                    )
                    results['coalescence'][kappa]['mean_blocks'].append(
                        (alpha_param, mean_blocks)
                    )
                
                # Print summary
                print(f"    Feasibility: {bound_info['feasible']}, "
                      f"Regime: {bound_info['regime']}")
                print(f"    E[θ̄[0]] = {bound_info['E_bar_theta']:.2f}")
                print(f"    Σ Π A_k = {bound_info['prod_sum']:.2e}, "
                      f"Σ (1-A_k) = {bound_info['sum_complement']:.2e}")
                
                if validate_with_simulation:
                    print(f"    Mean Λ[0] = {mean_coal:.2f} ± {std_coal:.2f}")
                    print(f"    Max Λ[0] = {max_coal:.0f}, Mean blocks = {mean_blocks:.1f}")
    
    # ========================================================
    # GENERATE PLOTS AND SAVE RESULTS
    # ========================================================
    
    filename_suffix = f"Garcia_Alpha_{min(alpha_params):.3f}_to_{max(alpha_params):.3f}"
    decay_type = "locally_continuous"
    
    results_dir = os.path.join("results", "garcia")
    os.makedirs(results_dir, exist_ok=True)
    
    # --------------------------------------------------------
    # COALESCENCE ANALYSIS PLOTS
    # --------------------------------------------------------
    if run_coalescence_analysis:
        _generate_coalescence_plots(
            results['coalescence'],
            results_dir,
            filename_suffix,
            decay_type,
            validate_with_simulation
        )
        _save_coalescence_results(
            results['coalescence'],
            results_dir,
            filename_suffix,
            decay_type,
            num_validation_samples,
            validate_with_simulation
        )
    
    return results


# ============================================================
# HELPER FUNCTIONS: COALESCENCE PLOTS
# ============================================================

def _generate_coalescence_plots(
    coal_results,
    results_dir,
    filename_suffix,
    decay_type,
    validate
):
    """Generate all plots for coalescence time analysis."""
    
    # Extract times dict for x-axis
    times_dct = {alpha: data['times'] for alpha, data in coal_results.items()}
    
    # Plot 1: Mean Coalescence Time
    if validate:
        filename = os.path.join(results_dir, f"Coalescence_Mean_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=times_dct,
            y={alpha: data['mean_coalescence'] for alpha, data in coal_results.items()},
            z=None,
            xlabel=r"$\alpha$ parameter (skeleton complexity)",
            ylabel=r"Mean Coalescence Time $\mathbb{E}[|\Lambda[0]|]$",
            title=f"Garcia Perfect Simulation: Mean Coalescence Time ({decay_type})",
            label_1="Empirical E[|Λ[0]|]",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    # Plot 2: Feasibility Metrics
    filename = os.path.join(results_dir, f"Feasibility_Metrics_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={alpha: data['prod_sum'] for alpha, data in coal_results.items()},
        z={alpha: data['sum_complement'] for alpha, data in coal_results.items()},
        xlabel=r"$\alpha$ parameter",
        ylabel="Feasibility Metric",
        title=f"Garcia Feasibility Conditions ({decay_type})",
        label_1=r"$\sum_{k\geq 1} \prod_{j=0}^{k-1} A_k$ (condition i)",
        label_2=r"$\sum_{k\geq 0} (1 - A_k)$ (condition ii)",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Plot 3: E[θ̄[0]] vs Mean Coalescence
    if validate:
        filename = os.path.join(results_dir, f"Coalescence_vs_BarTheta_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=times_dct,
            y={alpha: data['E_bar_theta'] for alpha, data in coal_results.items()},
            z={alpha: data['mean_coalescence'] for alpha, data in coal_results.items()},
            xlabel=r"$\alpha$ parameter",
            ylabel="Expected Value",
            title=f"E[θ̄[0]] vs E[|Λ[0]|] ({decay_type})",
            label_1=r"$\mathbb{E}[\bar{\theta}[0]]$ (good coalescence)",
            label_2=r"$\mathbb{E}[|\Lambda[0]|]$ (blocked rescaled)",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    # Plot 4: Number of Blocks
    if validate:
        filename = os.path.join(results_dir, f"Mean_Blocks_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=times_dct,
            y={alpha: data['mean_blocks'] for alpha, data in coal_results.items()},
            z=None,
            xlabel=r"$\alpha$ parameter",
            ylabel="Mean Number of Blocks",
            title=f"Block Structure Analysis ({decay_type})",
            label_1="Mean |{B_k}|",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    # Plot 5: Coalescence Time Distribution (if validation data exists)
    if validate:
        filename = os.path.join(results_dir, f"Coalescence_Distribution_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=times_dct,
            y={alpha: data['mean_coalescence'] for alpha, data in coal_results.items()},
            z={alpha: data['max_coalescence'] for alpha, data in coal_results.items()},
            xlabel=r"$\alpha$ parameter",
            ylabel="Coalescence Time",
            title=f"Coalescence Time: Mean vs Max ({decay_type})",
            label_1="Mean |Λ[0]|",
            label_2="Max |Λ[0]|",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")


def _save_coalescence_results(
    coal_results,
    results_dir,
    filename_suffix,
    decay_type,
    num_samples,
    validate
):
    """Save numerical results for coalescence time analysis."""
    
    results_file = os.path.join(results_dir, f"coalescence_numerical_results_{filename_suffix}.txt")
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GARCIA PERFECT SIMULATION RESULTS: COALESCENCE TIME ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Decay type: {decay_type}\n")
        f.write(f"  Validation samples: {num_samples if validate else 'N/A'}\n\n")
        
        for alpha, data in coal_results.items():
            f.write(f"\nRegime = {alpha:.3f}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'alpha':<10} {'E[θ̄[0]]':<14} {'Σ Π A_k':<14} {'Σ(1-A_k)':<14} "
                   f"{'Feasible':<10}\n")
            f.write("-" * 80 + "\n")
            
            for i, (alpha_val, _) in enumerate(data['E_bar_theta']):
                E_theta = data['E_bar_theta'][i][1]
                prod_sum = data['prod_sum'][i][1]
                sum_comp = data['sum_complement'][i][1]
                feasible = data['feasibility_metrics'][i][1]
                
                f.write(f"{alpha_val:<10.3f} {E_theta:<14.4f} {prod_sum:<14.2e} "
                       f"{sum_comp:<14.2e} {str(feasible):<10}\n")
            
            if validate and len(data['mean_coalescence']) > 0:
                f.write("\nCoalescence Time Statistics:\n")
                f.write(f"{'alpha':<10} {'Mean |Λ[0]|':<14} {'Std |Λ[0]|':<14} "
                       f"{'Max |Λ[0]|':<14} {'Mean Blocks':<14}\n")
                f.write("-" * 80 + "\n")
                for i, (alpha_val, _) in enumerate(data['mean_coalescence']):
                    mean_coal = data['mean_coalescence'][i][1]
                    std_coal = data['std_coalescence'][i][1]
                    max_coal = data['max_coalescence'][i][1]
                    mean_blocks = data['mean_blocks'][i][1]
                    f.write(f"{alpha_val:<10.3f} {mean_coal:<14.4f} {std_coal:<14.4f} "
                           f"{max_coal:<14.0f} {mean_blocks:<14.2f}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("INTERPRETATION:\n")
            f.write("=" * 80 + "\n")
            f.write("Feasibility Conditions (Theorem 3.5):\n")
            f.write("  (i)   Σ Π A_k = +∞  →  Λ[0] is a.s. finite\n")
            f.write("  (ii)  Σ(1-A_k) < +∞  →  Λ[0] has summable tail\n")
            f.write("  (iii) Exponential decay  →  Λ[0] has exponential tail\n\n")
            f.write("Key Property:\n")
            f.write("  {L_k}_{k≥0} are i.i.d. (unlike Gallo's L_n)\n")
            f.write("  This enables renewal-like analysis and tight tail bounds\n")
    
    print(f"Saved: {results_file}")


# ============================================================
# COMPARISON WITH GALLO (if both available)
# ============================================================

def compare_garcia_vs_gallo(
    garcia_results: dict,
    gallo_results: dict,
    results_dir: str,
    filename_suffix: str
):
    """
    Generate comparison plots between Garcia and Gallo approaches.
    
    Args:
        garcia_results: Results from Garcia simulation
        gallo_results: Results from Gallo simulation  
        results_dir: Directory to save plots
        filename_suffix: Suffix for filename
    """
    
    print("\n" + "=" * 80)
    print("GENERATING GARCIA vs GALLO COMPARISON")
    print("=" * 80)
    
    # Extract coalescence data
    garcia_coal = garcia_results.get('coalescence', {})
    gallo_trunc = gallo_results.get('truncated', {})
    
    if not garcia_coal or not gallo_trunc:
        print("Insufficient data for comparison")
        return
    
    # Get common regime (kappa = 0.0)
    garcia_data = garcia_coal.get(0.0, {})
    gallo_data = gallo_trunc.get(0.0, {})
    
    if not garcia_data or not gallo_data:
        print("No common regime found")
        return
    
    times_dct = {0.0: garcia_data['times']}
    
    # Comparison 1: Coalescence/Lookback Depth
    filename = os.path.join(results_dir, f"Comparison_Garcia_vs_Gallo_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=times_dct,
        y={0.0: garcia_data['mean_coalescence']},
        z={0.0: gallo_data['mu_S_analytical']},
        xlabel=r"$\alpha$ parameter",
        ylabel="Expected Depth",
        title="Garcia E[|Λ[0]|] vs Gallo E[L̃_n]",
        label_1="Garcia: E[|Λ[0]|] (unbiased)",
        label_2="Gallo: μ_S (truncated)",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Create comparison text file
    comparison_file = os.path.join(results_dir, f"comparison_garcia_gallo_{filename_suffix}.txt")
    
    with open(comparison_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GARCIA vs GALLO COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("KEY DIFFERENCES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Property':<30} {'Gallo (2009)':<25} {'Garcia (2011)':<25}\n")
        f.write("-" * 80 + "\n")
        
        comparisons = [
            ("Coalescence mechanism", "Distance to ref. string", "Good coalescence times"),
            ("Context detection", "Lag function l_w(k)", "Skeleton τ membership"),
            ("Block structure", "No blocks", "Blocks B_k via θ^k"),
            ("Independence", "N/A", "{L_k} are i.i.d."),
            ("Truncation", "User limit S", "No truncation"),
            ("Bias", "Biased if S < E[L_n]", "Unbiased (exact)"),
            ("Complexity", "O(S)", "O(Λ[0])"),
        ]
        
        for prop, gallo, garcia in comparisons:
            f.write(f"{prop:<30} {gallo:<25} {garcia:<25}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("THEORETICAL ADVANTAGES OF GARCIA:\n")
        f.write("=" * 80 + "\n")
        f.write("1. True perfect simulation (no truncation bias)\n")
        f.write("2. Block independence: {L_k} are i.i.d.\n")
        f.write("3. General feasibility conditions (Theorem 3.5)\n")
        f.write("4. Works for arbitrary locally continuous kernels\n")
        f.write("5. Renewal-like structure enables tight tail bounds\n")
    
    print(f"Saved: {comparison_file}")


# ============================================================
# EXAMPLE USAGE
# ============================================================

def create_simple_skeleton():
    """
    Create a simple skeleton for testing.
    """
    alphabet = [0, 1]
    
    # Simple skeleton: contexts of length 0, 1, 2
    skeleton = {
        tuple(),  # Empty context
        (0,), (1,),  # Length 1
        (0, 0), (0, 1), (1, 0), (1, 1)  # Length 2
    }
    
    return skeleton, alphabet


if __name__ == "__main__":
    # Example usage
    skeleton, alphabet = create_simple_skeleton()
    
    args = {
        'skeleton': skeleton,
        'alphabet': alphabet,
        'alpha_sequence': {},  # Will be generated in the function
        'epsilon': 0.1,
        'alpha_params': [0.1, 0.15, 0.2, 0.25, 0.3],
    }
    
    results = run_garcia_simulation(
        window=(0, 0),
        args=args,
        max_blocks=100,
        max_block_size=50,
        num_validation_samples=1000,
        validate_with_simulation=True,
        run_coalescence_analysis=True,
    )
    
    print("\nExperiment completed successfully!")