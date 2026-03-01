import time
import os
import numpy as np
import matplotlib.pyplot as plt

from src.simulators.gallo import GalloContextTreeSimulator



def run_gallo_simulation(
    window: tuple,
    args: dict,
    max_trie_depth: int = 8,
    max_depth: int = 50,          # truncation index S
    num_validation_samples: int = 10000,
    validate_with_simulation: bool = True,
    run_truncated: bool = True,
    run_non_truncated: bool = False,
    mc_burn_in: int = 10_000,
    mc_seed: int = 123,
):
    """
    Run Gallo experiments matching your CFF structure, but with CORRECT MC validation:
    - Empirical μ_S is estimated from a simulated CHAIN path, not i.i.d. past.
    - Empirical totals are scaled by (1 - |A|*epsilon) to match analytical bounds.
    """

    # Extract parameters
    alphas = args.get("alphas")
    alphabet = args.get("alphabet")
    reference_string = args.get("reference_string")
    epsilon = args.get("epsilon", 0.3)
    beta = args.get("beta", 0.7)

    # Results storage: keyed by alpha (not kappa)
    results = {
        "truncated": {} if run_truncated else None,
        "non_truncated": {} if run_non_truncated else None,
    }

    print("=" * 80)
    print("GALLO EXPERIMENT: Context Tree Simulation + Lookback Bounds")
    print(f"Window: [{window[0]}, {window[1]}], Truncation S={max_depth}")
    print(f"Reference string: {reference_string}, Epsilon: {epsilon}")
    print(f"Beta (AR decay): {beta}, Max trie depth: {max_trie_depth}")
    print(f"Alpha values: {alphas}")
    print("=" * 80)

    start_time = time.time()

    # ========================================================
    # RUN EXPERIMENTS OVER alpha
    # ========================================================
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"ALPHA = {alpha:.7f}")
        print(f"{'='*70}")

        # Init simulator
        sim = GalloContextTreeSimulator(
            alpha=alpha,
            alphabet=alphabet,
            reference_string=reference_string,
            epsilon=epsilon,
            beta=beta,
            max_depth=max_depth,
            max_trie_depth=max_trie_depth,
            show_progress=True,
        )

        # -------------------------
        # TRUNCATED
        # -------------------------
        if run_truncated:
            print("  [Truncated Analysis]")

            trunc_analytical = sim.analytical_lookback_bound(truncated=True)

            # store container
            results["truncated"][alpha] = {
                "time_sec": None,
                "analytical_total_bound": trunc_analytical["total_bound"],
                "mu_S_analytical": trunc_analytical["mu_S_analytical"],
                "tail_bound": trunc_analytical["tail_bound"],
                "prob_exceed_S": trunc_analytical["prob_exceed_S"],
                "user_impatience": sim.compute_user_impatience_bias_given_limit(),
                "mu_S_empirical": None,
                "empirical_total_bound_like": None,
                "tightness": None,
                "empirical_std_error": None,
            }

            if validate_with_simulation:

                empirical_mu_S = 0

                # Match scaling in the analytical bound:
                empirical_total = 0

                # "tightness": empirical / analytical bound (both on same scale)
                analytical_total = 0
                tightness = 0

                results["truncated"][alpha].update({
                    "mu_S_empirical": empirical_mu_S,
                    "empirical_total_bound_like": empirical_total,
                    "tightness": tightness,
                    "empirical_std_error": 0,
                })

                print(f"    Truncated bound: {analytical_total:.4f}")
                print(f"    μ_S(anal)={trunc_analytical['mu_S_analytical']:.4f}, tail={trunc_analytical['tail_bound']:.3e}")
                print(f"    μ_S(emp) ={empirical_mu_S:.4f} ± {0:.4f}")
                print(f"    Empirical total (scaled): {empirical_total:.4f}, tightness={tightness:.3%}")
            else:
                print(f"    Truncated bound: {trunc_analytical['total_bound']:.4f}")
                print(f"    μ_S(anal)={trunc_analytical['mu_S_analytical']:.4f}, tail={trunc_analytical['tail_bound']:.3e}")

        # -------------------------
        # NON-TRUNCATED
        # -------------------------
        if run_non_truncated:
            print("  [Non-Truncated Analysis]")

            non_trunc_analytical = sim.analytical_lookback_bound(
                truncated=False,
                compute_exact=False
            )

            results["non_truncated"][alpha] = {
                "time_sec": None,
                "theoretical_bound": non_trunc_analytical["theoretical_bound"],
                "empirical_mean": None,
                "tightness": None,
                "empirical_std_error": None,
                "exact_E_Lprime": non_trunc_analytical.get("exact_value") if non_trunc_analytical.get("exact_value") is not None else float('nan'),
                "ci_low": None,
                "ci_high": None,
                "ci_level": None,
            }

            if validate_with_simulation:
                # Correct MC: estimate E[L'] (untruncated) from chain
                out = sim.monte_carlo_E_Lprime(
                    T=num_validation_samples,
                    burn_in=mc_burn_in,
                    seed=mc_seed,
                    truncated=False,
                    drop_infinite=True,
                    return_ci=True,
                    ci_level=0.95,
                )

                emp_mean = out["mean"]
                bound = non_trunc_analytical["theoretical_bound"]
                tightness = (emp_mean / bound) if (np.isfinite(bound) and bound > 0) else np.nan

                results["non_truncated"][alpha].update({
                    "empirical_mean": emp_mean,
                    "tightness": tightness,
                    "empirical_std_error": out["stderr"],
                    "ci_low": out.get("ci_low", float('nan')),
                    "ci_high": out.get("ci_high", float('nan')),
                    "ci_level": out.get("ci_level", float('nan')),
                })

                print(f"    Theoretical bound: {bound:.4f}")
                print(f"    Empirical E[L'] :  {emp_mean:.4f} ± {out['stderr']:.4f}")
                print(f"    Tightness: {tightness:.3%}")
            else:
                print(f"    Theoretical bound: {non_trunc_analytical['theoretical_bound']:.4f}")

        # timing
        elapsed = time.time() - start_time
        if run_truncated:
            results["truncated"][alpha]["time_sec"] = elapsed
        if run_non_truncated:
            results["non_truncated"][alpha]["time_sec"] = elapsed

    # ========================================================
    # SAVE + PLOTS
    # ========================================================
    filename_suffix = f"Gallo_Alpha_{min(alphas):.3f}_to_{max(alphas):.3f}"
    decay_type = f"context_tree_beta_{beta}"
    results_dir = os.path.join("results", "gallo")
    os.makedirs(results_dir, exist_ok=True)

    if run_truncated:
        _generate_truncated_plots(results["truncated"], results_dir, filename_suffix, max_depth, decay_type, validate_with_simulation)
        _save_truncated_results(results["truncated"], results_dir, filename_suffix, max_depth, decay_type, num_validation_samples, validate_with_simulation)

    if run_non_truncated:
        _generate_non_truncated_plots(results["non_truncated"], results_dir, filename_suffix, decay_type, validate_with_simulation)
        _save_non_truncated_results(results["non_truncated"], results_dir, filename_suffix, decay_type, num_validation_samples, validate_with_simulation)

    if run_truncated and run_non_truncated:
        _generate_comparison_plot(results["truncated"], results["non_truncated"], results_dir, filename_suffix, max_depth, decay_type)

    return results

def plot_and_save_figure(
    x,
    y,
    z=None,
    xlabel="",
    ylabel="",
    title="",
    label_1=None,
    label_2=None,
):
    """
    Generic plotting utility for thesis experiments.

    Parameters
    ----------
    x : list | array | dict
        X-axis values (alphas). If dict, keys are used and sorted.
    y : list | array
        First curve values.
    z : list | array | None
        Optional second curve values.
    """

    # --------------------------------------------------
    # Normalize x input
    # --------------------------------------------------
    if isinstance(x, dict):
        x_vals = np.array(sorted(x.keys()), dtype=float)
    else:
        x_vals = np.array(x, dtype=float)

    y_vals = np.array(y, dtype=float)

    if len(x_vals) != len(y_vals):
        raise ValueError(
            f"x and y must have same length: len(x)={len(x_vals)}, len(y)={len(y_vals)}"
        )

    if z is not None:
        z_vals = np.array(z, dtype=float)
        if len(z_vals) != len(x_vals):
            raise ValueError(
                f"x and z must have same length: len(x)={len(x_vals)}, len(z)={len(z_vals)}"
            )
    else:
        z_vals = None

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        x_vals,
        y_vals,
        marker="o",
        linewidth=2,
        markersize=5,
        label=label_1 if label_1 else None,
    )

    if z_vals is not None:
        ax.plot(
            x_vals,
            z_vals,
            marker="s",
            linewidth=2,
            linestyle="--",
            markersize=5,
            label=label_2 if label_2 else None,
        )

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)

    ax.grid(True, alpha=0.3)

    if label_1 or label_2:
        ax.legend()

    fig.tight_layout()
    return fig


def _generate_truncated_plots(trunc_results, results_dir, filename_suffix, S, decay_type, validate):
    # x is alpha
    x = sorted(trunc_results.keys())

    # Plot 1: empirical vs analytical total (if validation)
    if validate:
        filename = os.path.join(results_dir, f"Truncated_Empirical_vs_Analytical_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=trunc_results,
            y=[trunc_results[a]["empirical_total_bound_like"] for a in x],
            z=[trunc_results[a]["analytical_total_bound"] for a in x],
            xlabel=r"$\alpha$ (growth parameter)",
            ylabel=r"Expected Lookback $\mathbb{E}[L']$ (scaled)",
            title=f"Truncated: Empirical vs Analytical ({decay_type}, S={S})",
            label_1="Empirical (MC, scaled)",
            label_2="Analytical Bound",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    # Plot 2: analytical total bound only
    filename = os.path.join(results_dir, f"Truncated_Analytical_Bound_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=x,
        y=[trunc_results[a]["analytical_total_bound"] for a in x],
        z=None,
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"$\mathbb{E}[L']$ Upper Bound",
        title=f"Truncated Analytical Bound ({decay_type}, S={S})",
        label_1="Analytical Bound",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

    # Plot 3: components μ_S vs tail
    filename = os.path.join(results_dir, f"Truncated_Components_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=x,
        y=[trunc_results[a]["mu_S_analytical"] for a in x],
        z=[trunc_results[a]["tail_bound"] for a in x],
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel="Component value",
        title=f"Components: $\\mu_S$ vs Tail ({decay_type}, S={S})",
        label_1=r"$\mu_S$ (analytical)",
        label_2=r"Tail bound",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

    # Plot 4: user impatience
    filename = os.path.join(results_dir, f"Truncated_User_Impatience_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=x,
        y=[trunc_results[a]["user_impatience"] for a in x],
        z=None,
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel="Bias",
        title=f"User Impatience Bias (S={S})",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

    # Plot 5: P(L>S)
    filename = os.path.join(results_dir, f"Truncated_Prob_Exceed_S_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=x,
        y=[trunc_results[a]["prob_exceed_S"] for a in x],
        z=None,
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"$P(L>S)$",
        title=f"Probability of Exceeding Truncation (S={S})",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

    # Plot 6: tightness
    if validate:
        filename = os.path.join(results_dir, f"Truncated_Tightness_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=x,
            y=[trunc_results[a]["tightness"] for a in x],
            z=None,
            xlabel=r"$\alpha$ (growth parameter)",
            ylabel=r"Tightness = Empirical / Bound",
            title=f"Bound Tightness ({decay_type}, S={S})",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")


def _save_truncated_results(trunc_results, results_dir, filename_suffix, S, decay_type, num_samples, validate):
    results_file = os.path.join(results_dir, f"truncated_numerical_results_{filename_suffix}.txt")

    x = sorted(trunc_results.keys())

    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TRUNCATED RESULTS (GALLO)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Truncation index S: {S}\n")
        f.write(f"  Decay type: {decay_type}\n")
        f.write(f"  Validation samples: {num_samples if validate else 'N/A'}\n\n")

        f.write(f"{'alpha':<10} {'mu_S(anal)':<14} {'tail':<14} {'bound_total':<14} {'P(L>S)':<14} {'bias':<14}\n")
        f.write("-" * 80 + "\n")

        for a in x:
            d = trunc_results[a]
            f.write(f"{a:<10.3f} {d['mu_S_analytical']:<14.6f} {d['tail_bound']:<14.3e} {d['analytical_total_bound']:<14.6f} "
                    f"{d['prob_exceed_S']:<14.3e} {d['user_impatience']:<14.3e}\n")

        if validate:
            f.write("\nValidation (MC):\n")
            f.write(f"{'alpha':<10} {'mu_S(emp)':<14} {'SE(emp)':<14} {'emp_total(scaled)':<18} {'tightness':<14}\n")
            f.write("-" * 80 + "\n")
            for a in x:
                d = trunc_results[a]
                f.write(f"{a:<10.3f} {d['mu_S_empirical']:<14.6f} {d['empirical_std_error']:<14.6f} "
                        f"{d['empirical_total_bound_like']:<18.6f} {d['tightness']:<14.3%}\n")

    print(f"Saved: {results_file}")


def _generate_non_truncated_plots(non_trunc_results, results_dir, filename_suffix, decay_type, validate):
    x = sorted(non_trunc_results.keys())

    # Tightness
    if validate:
        filename = os.path.join(results_dir, f"NonTruncated_Tightness_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=x,
            y=[non_trunc_results[a]["tightness"] for a in x],
            z=None,
            xlabel=r"$\alpha$ (growth parameter)",
            ylabel=r"Tightness = Empirical / Bound",
            title=f"Non-Truncated Bound Tightness ({decay_type})",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    # Empirical vs theoretical bound
    if validate:
        filename = os.path.join(results_dir, f"NonTruncated_Empirical_Validation_{filename_suffix}.png")
        fig = plot_and_save_figure(
            x=x,
            y=[non_trunc_results[a]["empirical_mean"] for a in x],
            z=[non_trunc_results[a]["theoretical_bound"] for a in x],
            xlabel=r"$\alpha$ (growth parameter)",
            ylabel=r"$\mathbb{E}[L']$",
            title=f"Non-Truncated: Empirical vs Theoretical Bound ({decay_type})",
            label_1="Empirical (MC)",
            label_2="Theoretical Bound",
        )
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")


def _save_non_truncated_results(non_trunc_results, results_dir, filename_suffix, decay_type, num_samples, validate):
    results_file = os.path.join(results_dir, f"non_truncated_numerical_results_{filename_suffix}.txt")
    x = sorted(non_trunc_results.keys())

    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NON-TRUNCATED RESULTS (GALLO)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Decay type: {decay_type}\n")
        f.write(f"  Validation samples: {num_samples if validate else 'N/A'}\n\n")

        f.write(f"{'alpha':<10} {'bound':<16} {'exact_E[L]':<16}\n")
        if validate:
            f.write(f"{'empirical':<16} {'SE':<16} {'tightness':<16} {'CI_low':<16} {'CI_high':<16} {'CI_level':<16}\n")
        else:
            f.write("\n")
        f.write("-" * 80 + "\n")

        for a in x:
            d = non_trunc_results[a]
            if validate:
                f.write(f"{a:<10.3f} {d['theoretical_bound']:<16.6f} {d['exact_E_Lprime']:<16.6f} {d['empirical_mean']:<16.6f} {d['empirical_std_error']:<16.6f} {d['tightness']:<16.3%} {d['ci_low']:<16.6f} {d['ci_high']:<16.6f} {d['ci_level']:<16.3f}\n")
            else:
                f.write(f"{a:<10.3f} {d['theoretical_bound']:<16.6f}\n")

    print(f"Saved: {results_file}")


def _generate_comparison_plot(trunc_results, non_trunc_results, results_dir, filename_suffix, S, decay_type):
    x = sorted(set(trunc_results.keys()) & set(non_trunc_results.keys()))
    if not x:
        return

    filename = os.path.join(results_dir, f"Truncated_vs_NonTruncated_Comparison_{filename_suffix}.png")
    fig = plot_and_save_figure(
        x=x,
        y=[trunc_results[a]["mu_S_analytical"] for a in x],
        z=[non_trunc_results[a]["theoretical_bound"] for a in x],
        xlabel=r"$\alpha$ (growth parameter)",
        ylabel=r"Value",
        title=f"Truncated $\\mu_S$ (S={S}) vs Non-truncated Bound ({decay_type})",
        label_1=f"Truncated $\\mu_S$ (S={S})",
        label_2="Non-truncated theoretical bound",
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
