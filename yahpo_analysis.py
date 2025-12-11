import os
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. Load data and compute best values per optimizer & function
# =============================================================================

def load_best_values(base_dir="yahpo_results"):
    """
    Walks directory tree:
        yahpo_results/{optimizer_name}/{blackbox_function_id}.json

    Each JSON file is a list of:
        [ [param_1, param_2, ...], function_value, decision_time ]

    Returns:
        best_values: dict[optimizer][function_id] -> best_function_value
    """
    best_values = defaultdict(dict)

    if not os.path.isdir(base_dir):
        raise ValueError(f"Base dir '{base_dir}' does not exist")

    for optimizer_name in os.listdir(base_dir):
        opt_dir = os.path.join(base_dir, optimizer_name)
        if not os.path.isdir(opt_dir):
            continue

        for filename in os.listdir(opt_dir):
            if not filename.endswith(".json"):
                continue

            function_id = os.path.splitext(filename)[0]
            filepath = os.path.join(opt_dir, filename)

            with open(filepath, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Warning: could not read {filepath}: {e}")
                    continue

            if not data:
                continue

            # data is a list of [params, value, decision_time]
            values = [entry[1] for entry in data]
            best_val = float(min(values))
            best_values[optimizer_name][function_id] = best_val

    return best_values


# =============================================================================
# 2. Performance profiles (Dolan–Moré)
# =============================================================================

def compute_performance_profile(best_values, ratio_fail=1e3, num_points=200):
    """
    Computes performance profiles for all optimizers across all functions.

    best_values: dict[optimizer][function_id] -> best_function_value

    Returns:
        taus: 1D array of tau values
        perf_profiles: dict[optimizer] -> (taus, rho(tau)) arrays
    """
    optimizers = sorted(best_values.keys())
    # union of all function IDs
    all_functions = sorted(
        {fid for opt in optimizers for fid in best_values[opt].keys()}
    )

    if not optimizers or not all_functions:
        raise ValueError("No optimizers or functions found in best_values")

    # ratio[optimizer][function_id] = performance ratio
    ratio = {opt: {fid: ratio_fail for fid in all_functions} for opt in optimizers}

    finite_ratios = []

    # For each function, compute best value across optimizers that have it,
    # then assign ratio = f_opt / f_best for those optimizers.
    for fid in all_functions:
        # gather available values for this function
        vals = [(opt, best_values[opt][fid])
                for opt in optimizers if fid in best_values[opt]]

        if not vals:
            # no optimizer has this function, skip
            continue

        f_best = min(v for _, v in vals)

        for opt, v in vals:
            r = v / f_best if f_best != 0 else 1.0
            ratio[opt][fid] = r
            finite_ratios.append(r)

    if not finite_ratios:
        raise ValueError("No finite ratios computed; check your data")

    max_finite = max(finite_ratios)
    tau_max = min(max_finite * 1.1, ratio_fail)
    tau_min = 1.0

    taus = np.linspace(tau_min, tau_max, num_points)

    # Now compute performance profile for each optimizer
    perf_profiles = {}

    n_problems = len(all_functions)

    for opt in optimizers:
        r_list = np.array([ratio[opt][fid] for fid in all_functions], dtype=float)

        rhos = []
        for tau in taus:
            # fraction of problems where r <= tau
            success_fraction = np.mean(r_list <= tau)
            rhos.append(success_fraction)

        perf_profiles[opt] = (taus, np.array(rhos))

    return taus, perf_profiles


def plot_performance_profiles(perf_profiles, title="Performance Profiles", save_path="performance_profiles.png"):
    """
    perf_profiles: dict[optimizer] -> (taus, rho(tau))
    """
    plt.figure()
    for opt, (taus, rhos) in perf_profiles.items():
        plt.step(taus, rhos, where="post", label=opt)

    plt.xlabel(r"Performance ratio $\tau$")
    plt.ylabel(r"Fraction of problems with $f_{a,t} \leq \tau f^*_{t}$")
    plt.title(title)
    plt.xlim(1, None)
    plt.ylim(0, 1.01)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# 3. ECDF of normalized regret
# =============================================================================

def compute_normalized_regret(best_values, eps=1e-12):
    """
    Compute normalized regret per optimizer & function.

    For each function t:
        f_min_t = min_a f[a,t]
        f_max_t = max_a f[a,t]

    For each optimizer a, function t where it has data:
        norm_regret[a,t] = (f[a,t] - f_min_t) / (f_max_t - f_min_t + eps)

    Returns:
        norm_regret: dict[optimizer] -> list of normalized regrets over functions
    """
    optimizers = sorted(best_values.keys())
    all_functions = sorted(
        {fid for opt in optimizers for fid in best_values[opt].keys()}
    )

    # Compute per-function min and max over all optimizers that have data
    f_min = {}
    f_max = {}
    for fid in all_functions:
        vals = [best_values[opt][fid]
                for opt in optimizers if fid in best_values[opt]]
        if not vals:
            continue
        f_min[fid] = min(vals)
        f_max[fid] = max(vals)

    norm_regret = {opt: [] for opt in optimizers}

    for opt in optimizers:
        for fid, f_val in best_values[opt].items():
            if fid not in f_min or fid not in f_max:
                continue
            denom = f_max[fid] - f_min[fid]
            nr = (f_val - f_min[fid]) / (denom + eps)
            norm_regret[opt].append(nr)

    return norm_regret


def compute_ecdf(values):
    """
    Given a 1D list/array of values, return (x, y) for ECDF.
    """
    if len(values) == 0:
        return np.array([]), np.array([])

    x = np.sort(np.asarray(values))
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x, y


def plot_ecdf_normalized_regret(norm_regret, title="ECDF of Normalized Regret", save_path="ecdf_normalized_regret.png"):
    """
    norm_regret: dict[optimizer] -> list of normalized regrets
    """
    plt.figure()
    for opt, vals in norm_regret.items():
        x, y = compute_ecdf(vals)
        if len(x) == 0:
            continue
        plt.step(x, y, where="post", label=opt)

    plt.xlabel("Normalized regret (0 = best for each function)")
    plt.ylabel("ECDF")
    plt.title(title)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# 4. Main
# =============================================================================

def main():
    base_dir = "yahpo_results"

    # 1. Load best values
    best_values = load_best_values(base_dir=base_dir)
    print("Loaded best values for:")
    for opt, fs in best_values.items():
        print(f"  {opt}: {len(fs)} functions")

    # 2. Performance profiles
    taus, perf_profiles = compute_performance_profile(best_values)
    plot_performance_profiles(perf_profiles,
                              title="Performance Profiles (best f per function)")

    # 3. ECDF of normalized regret
    norm_regret = compute_normalized_regret(best_values)
    plot_ecdf_normalized_regret(norm_regret,
                                title="ECDF of Normalized Regret (per function)")


if __name__ == "__main__":
    main()
