"""
Example: Using the Visualization Module with Pipeline Simulations

This script demonstrates how to use the visualization functions with the 
pipeline simulations to automatically generate analysis plots.
"""

import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.visualize import (
    plot_tightness_comparison,
    plot_bound_validation,
    plot_empirical_with_ci,
    plot_tightness_and_tail,
    plot_tail_probability,
    plot_single_tightness,
    plot_bias,
    plot_global_tightness,
    plot_log_scale,
)


def example_tightness_comparison():
    """Example: Compare tightness across decay types (from notebook cell 1)"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Tightness Comparison")
    print("="*70)
    
    # Data from test_plots.ipynb
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tight_exp = np.array([94.69, 97.33, 102.36, 100.34, 100.44,
                         98.70, 100.52, 99.32, 99.55])
    
    p = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
    tight_poly = np.array([98.34, 99.01, 101.47, 105.29,
                          98.18, 98.67, 103.01, 97.74, 100.01])
    
    fig = plot_tightness_comparison(
        rho, tight_exp, p, tight_poly,
        label1='Exponential decay',
        label2='Polynomial decay',
        filename='results/example_tightness_comparison.pdf'
    )
    print("✓ Generated: results/example_tightness_comparison.pdf")


def example_bound_validation():
    """Example: Validate theoretical bounds against MC estimates (notebook cell 4)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Bound Validation - Exponential Decay")
    print("="*70)
    
    # Truncated exponential data
    theo_te = np.array([0.1166, 0.2845, 0.5544, 0.9894, 1.7932, 
                       3.0319, 5.3819, 10.7224, 29.9261])
    mc_te = np.array([0.1104, 0.2769, 0.5544, 0.9928, 1.7932, 
                     2.9923, 5.3819, 10.6491, 29.7907])
    
    fig = plot_bound_validation(
        theo_te, mc_te,
        title='Truncation Induced Bound Validation — Exponential Decay',
        filename='results/example_validation_exponential.pdf'
    )
    print("✓ Generated: results/example_validation_exponential.pdf")


def example_empirical_with_ci():
    """Example: Plot empirical values with confidence intervals (notebook cell 7)"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Empirical Estimates with Confidence Intervals")
    print("="*70)
    
    alpha = np.array([0.005, 0.008, 0.011, 0.013, 0.016,
                     0.019, 0.022, 0.024, 0.027, 0.030])
    
    empirical = np.array([1.021573, 1.021573, 1.021573, 1.021573, 1.021573,
                         1.021573, 1.021573, 1.021573, 1.021573, 1.021574])
    
    ci_low = np.array([1.020690, 1.020690, 1.020690, 1.020690, 1.020690,
                      1.020690, 1.020690, 1.020691, 1.020691, 1.020691])
    
    ci_high = np.array([1.022456, 1.022456, 1.022456, 1.022456, 1.022456,
                       1.022456, 1.022456, 1.022456, 1.022456, 1.022456])
    
    fig = plot_empirical_with_ci(
        alpha, empirical, ci_low, ci_high,
        xlabel=r'$\alpha$',
        ylabel=r'Empirical value',
        title='Empirical Estimate with 95% Confidence Interval',
        filename='results/example_ci_plot.pdf'
    )
    print("✓ Generated: results/example_ci_plot.pdf")


def example_tightness_and_tail():
    """Example: Dual-axis plot of tightness and tail probability (notebook cell 5)"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Tightness and Tail Probability (Dual Axes)")
    print("="*70)
    
    # Exponential
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tight_exp = np.array([94.69, 97.33, 100, 100, 100, 98.70, 100, 99.32, 99.55])
    tail_exp = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 7.34e-228])
    
    fig = plot_tightness_and_tail(
        rho, tight_exp, tail_exp,
        xlabel=r'Load $\rho$',
        title='Truncated Case — Exponential Decay',
        filename='results/example_dual_axis_exponential.pdf'
    )
    print("✓ Generated: results/example_dual_axis_exponential.pdf")
    
    # Polynomial
    p = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    tight_poly = np.array([109.42, 98.34, 99.01, 101.47, 105.29, 
                          98.18, 98.67, 103.01, 97.74])
    tail_poly = np.array([1.80e-8, 2.40e-12, 3.33e-16, 1e-20, 1e-20,
                         1e-20, 1e-20, 1e-20, 1e-20])
    
    fig = plot_tightness_and_tail(
        p, tight_poly, tail_poly,
        xlabel=r'Tail parameter $p$',
        title='Truncated Case — Polynomial Decay',
        filename='results/example_dual_axis_polynomial.pdf'
    )
    print("✓ Generated: results/example_dual_axis_polynomial.pdf")


def example_tail_probability():
    """Example: Compare tail probabilities (notebook cell 6)"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Tail Probability Comparison")
    print("="*70)
    
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tail_exp = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 7.34e-228])
    
    p = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
    tail_poly = np.array([1.80e-8, 2.40e-12, 3.33e-16, 1e-20, 1e-20,
                         1e-20, 1e-20, 1e-20, 1e-20])
    
    fig = plot_tail_probability(
        rho, tail_exp, p, tail_poly,
        label1='Exponential decay',
        label2='Polynomial decay',
        filename='results/example_tail_probability.pdf'
    )
    print("✓ Generated: results/example_tail_probability.pdf")


def example_single_tightness():
    """Example: Single tightness plot (notebook cell 2)"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Single Tightness Plots")
    print("="*70)
    
    # Exponential
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tight_exp = np.array([94.69, 97.33, 100, 100, 100, 98.70, 100, 99.32, 99.55])
    
    fig = plot_single_tightness(
        rho, tight_exp,
        xlabel=r'Load $\rho$',
        title='Bound Tightness — Exponential Decay',
        filename='results/example_tightness_exponential.pdf',
        marker='o',
        linestyle='-'
    )
    print("✓ Generated: results/example_tightness_exponential.pdf")
    
    # Polynomial
    p = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
    tight_poly = np.array([98.34, 99.01, 101.47, 105.29, 98.18, 98.67, 103.01, 97.74, 100.01])
    
    fig = plot_single_tightness(
        p, tight_poly,
        xlabel=r'Tail parameter $p$',
        title='Bound Tightness — Polynomial Decay',
        filename='results/example_tightness_polynomial.pdf',
        marker='s',
        linestyle='--'
    )
    print("✓ Generated: results/example_tightness_polynomial.pdf")


def example_bias_analysis():
    """Example: Bias analysis (notebook cell 12)"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Bias Analysis")
    print("="*70)
    
    alpha = np.array([0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 
                     0.013, 0.016, 0.018, 0.020])
    
    bias = np.array([5.72e-1, 5.72e-1, 5.72e-1, 5.72e-1, 5.72e-1,
                    5.72e-1, 5.72e-1, 5.72e-1, 5.72e-1, 5.72e-1])
    
    fig = plot_bias(
        alpha, bias,
        xlabel=r'Growth parameter $\alpha$',
        title='Bias vs Growth Parameter (Context Tree)',
        filename='results/example_bias.pdf'
    )
    print("✓ Generated: results/example_bias.pdf")


def example_global_tightness():
    """Example: Global (non-truncated) tightness (notebook cell 3)"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Global Bound Tightness")
    print("="*70)
    
    # Non-truncated exponential
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tight_exp_nt = np.array([2.00, 4.43, 7.95, 11.91, 18.00, 23.94, 32.46, 42.60, 59.58])
    
    fig = plot_global_tightness(
        rho, tight_exp_nt,
        xlabel=r'Load $\rho$',
        title='Global Bound Tightness — Exponential Decay',
        filename='results/example_global_tightness_exponential.pdf'
    )
    print("✓ Generated: results/example_global_tightness_exponential.pdf")
    
    # Non-truncated polynomial
    p = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
    tight_poly_nt = np.array([10.81, 6.34, 4.60, 3.63, 3.01, 2.58, 2.25, 2.00, 1.80])
    
    fig = plot_global_tightness(
        p, tight_poly_nt,
        xlabel=r'Tail parameter $p$',
        title='Global Bound Tightness — Polynomial Decay',
        filename='results/example_global_tightness_polynomial.pdf'
    )
    print("✓ Generated: results/example_global_tightness_polynomial.pdf")


def example_log_scale():
    """Example: Log-scale plot (notebook cell 11)"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Log-Scale Analysis")
    print("="*70)
    
    alpha = np.array([0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 
                     0.013, 0.016, 0.018, 0.020])
    
    E_tail = np.array([9.210, 9.251, 9.296, 9.346, 9.401, 9.463, 
                      9.533, 9.613, 9.705, 9.811])
    
    fig = plot_log_scale(
        alpha, E_tail,
        xlabel=r'$\alpha$',
        ylabel=r'$\mathbb{E}[(L_1 - S)_+]$',
        title=r'$\mathbb{E}[(L_1 - S)_+]$ vs $\alpha$ (Context Tree $\beta=0.7$)',
        filename='results/example_log_scale.pdf',
        yscale='log'
    )
    print("✓ Generated: results/example_log_scale.pdf")


def main():
    """Run all visualization examples"""
    print("\n" + "="*70)
    print("VISUALIZATION MODULE EXAMPLES")
    print("All plots from notebooks/test_plots.ipynb now in src/utils/visualize.py")
    print("="*70)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run examples
    example_tightness_comparison()
    example_bound_validation()
    example_empirical_with_ci()
    example_tightness_and_tail()
    example_tail_probability()
    example_single_tightness()
    example_bias_analysis()
    example_global_tightness()
    example_log_scale()
    
    print("\n" + "="*70)
    print("✓ ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nGenerated plots saved to: results/")
    print("\nThese visualization functions are now available in:")
    print("  - src/utils/visualize.py")
    print("\nAnd automatically used in:")
    print("  - src/pipelines/context_tree_kernel_pipeline.py")
    print("  - src/pipelines/continuous_kernel_pipelines.py")
    print("  - src/pipelines/locally_continuous_kernel_pipeline.py")
    print("\nSee VISUALIZATION_GUIDE.md for detailed documentation.\n")


if __name__ == '__main__':
    main()
