import os
import math
import numpy as np
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

def _mc_thread_job(sim, T, burn_in, seed):
    mc = sim.monte_carlo_E_Lprime(
        T=T,
        burn_in=burn_in,
        seed=seed,
        truncated=False,
        drop_infinite=True,
    )
    # assumes mc has keys "mean" and "stderr"
    return {"T": T, "mean": mc["mean"], "stderr": mc["stderr"]}

def parallel_monte_carlo_E_Lprime(
    sim,
    T_total: int,
    burn_in: int,
    base_seed: int,
    n_threads: int | None = None,
):
    if n_threads is None:
        # For i5-1135G7: try 4 first; 8 sometimes helps
        n_threads = min(8, os.cpu_count() or 4)

    q, r = divmod(T_total, n_threads)
    chunks = [q + (1 if i < r else 0) for i in range(n_threads)]
    seeds = [base_seed + i * 10_000_019 for i in range(n_threads)]

    results = []
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futs = [ex.submit(_mc_thread_job, sim, chunks[i], burn_in, seeds[i]) for i in range(n_threads)]
        for f in as_completed(futs):
            results.append(f.result())

    # ---- Combine batch means and batch standard errors correctly ----
    Ts = [d["T"] for d in results]
    mus = [d["mean"] for d in results]
    ses = [d["stderr"] for d in results]

    T_tot = sum(Ts)
    mu = sum(Ti * mi for Ti, mi in zip(Ts, mus)) / T_tot

    # Reconstruct variance from within-batch variance + between-batch variance
    num = 0.0
    for Ti, mi, sei in zip(Ts, mus, ses):
        var_i = (sei ** 2) * Ti          # since sei^2 ≈ Var/Ti
        num += (Ti - 1) * var_i + Ti * (mi - mu) ** 2

    var = num / (T_tot - 1) if T_tot > 1 else float("nan")
    se = math.sqrt(var / T_tot) if T_tot > 0 else float("nan")

    return {
        "T_total": T_tot,
        "mean": mu,
        "stderr": se,
        "n_threads": n_threads,
        "batch_results": results,
    }
def batch_means_ci(x: np.ndarray, ci_level: float = 0.95, batch_size: int = 200) -> Tuple[float, float, float]:
    """
    Batch means CI for mean of correlated sequence.
    Returns (mean, ci_low, ci_high).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * batch_size:
        # fallback: fewer batches; increase variance rather than pretend iid
        batch_size = max(10, n // 10)

    b = n // batch_size
    x_use = x[:b * batch_size]
    batches = x_use.reshape(b, batch_size).mean(axis=1)

    mean = float(x_use.mean())
    # t-interval on batch means
    bm_std = float(np.std(batches, ddof=1)) if b > 1 else 0.0
    bm_stderr = bm_std / np.sqrt(b) if b > 1 else float("inf")

    # 95% t critical approx; for thesis usage fine; you can plug scipy if you want exact
    # For b>=30, 1.96 is fine.
    tcrit = 1.96 if abs(ci_level - 0.95) < 1e-12 or b >= 30 else 1.96
    half = tcrit * bm_stderr
    return mean, mean - half, mean + half, bm_std, bm_stderr