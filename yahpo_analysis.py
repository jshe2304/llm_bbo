#!/usr/bin/env python3
"""
Compute Dolan–Moré performance profiles from experimental results stored as

    yahpo_results/{optimizer_name}/{blackbox_function_id}.json

Each JSON file is a list of evaluations:
[
  [ [param_1, param_2, ...], function_value_at_that_point, time_it_took_to_decide ],
  ...
]

This script:
  * treats the performance measure t_{p,a} for optimizer a on problem p
    as the **best (minimum) function value** seen in that JSON file.
  * assumes **smaller is better** (e.g. loss). If your metric is to be
    maximized (e.g. accuracy), either:
        - store -metric in the JSON, or
        - adapt the code where indicated below.

It then computes Dolan–Moré performance profiles:

    r_{p,a} = t_{p,a} / min_b t_{p,b}

and plots, for each optimizer a,

    rho_a(tau) = fraction of problems p with r_{p,a} <= tau
"""

import argparse
import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def load_best_values(results_root):
    """
    Scan results_root/yahpo_results/{optimizer}/{problem}.json and
    return a dict:

        best_values[optimizer][problem_id] = best_value (float)
    """
    best_values = {}

    # optimizers are subdirectories of results_root
    for opt_name in sorted(os.listdir(results_root)):
        opt_dir = os.path.join(results_root, opt_name)
        if not os.path.isdir(opt_dir):
            continue

        best_values[opt_name] = {}
        json_paths = glob.glob(os.path.join(opt_dir, "*.json"))

        for path in json_paths:
            problem_id = os.path.splitext(os.path.basename(path))[0]

            with open(path, "r") as f:
                data = json.load(f)

            if not data:
                # empty file, skip
                continue

            # data is a list of [params_list, value, decision_time]
            # extract all function values for this optimizer/problem
            values = [entry[1] for entry in data]

            # If your metric is to be maximized, you could do:
            # values = [-entry[1] for entry in data]
            # so that "smaller is better" still holds.
            best_val = min(values)

            best_values[opt_name][problem_id] = best_val

    return best_values


def compute_ratios(best_values):
    """
    Given best_values[optimizer][problem] = t_{p,a}, compute the
    performance ratios:

        r_{p,a} = t_{p,a} / min_b t_{p,b}

    Returns:
        optimizers: list of optimizer names
        problems:   list of problem IDs actually used
        ratios:     np.ndarray shape (n_problems, n_optimizers)
    """
    optimizers = sorted(best_values.keys())

    # union of all problem IDs across optimizers
    all_problems = sorted(
        {pid for opt in optimizers for pid in best_values[opt].keys()}
    )

    # Build matrix t[p, a] with Inf for missing combinations
    t = np.full((len(all_problems), len(optimizers)), np.inf, dtype=float)

    for j, opt in enumerate(optimizers):
        for i, problem in enumerate(all_problems):
            if problem in best_values[opt]:
                t[i, j] = best_values[opt][problem]

    # per-problem best across optimizers
    t_min = np.min(t, axis=1)

    # remove problems where all optimizers are missing (t_min == inf)
    valid_mask = ~np.isinf(t_min)
    t = t[valid_mask]
    t_min = t_min[valid_mask]
    used_problems = [p for k, p in enumerate(all_problems) if valid_mask[k]]

    if t.shape[0] == 0:
        raise RuntimeError("No valid (problem, optimizer) combinations found.")

    # small epsilon to avoid divide-by-zero if t_min is extremely small
    eps = 1e-12
    ratios = t / (t_min[:, None] + eps)

    return optimizers, used_problems, ratios


def compute_performance_profile(ratios, taus):
    """
    Given ratios[p, a] and a vector of tau values, compute:

        rho_a(tau) = fraction of problems p with ratios[p, a] <= tau

    Returns:
        profile: np.ndarray shape (len(taus), n_optimizers)
    """
    n_problems, n_opts = ratios.shape
    profile = np.zeros((len(taus), n_opts), dtype=float)

    for k, tau in enumerate(taus):
        profile[k, :] = np.sum(ratios <= tau, axis=0) / float(n_problems)

    return profile


def plot_performance_profiles(optimizers, taus, profile, output_path):
    """
    Plot performance profiles and save to output_path.
    """
    plt.figure()
    for j, opt in enumerate(optimizers):
        plt.plot(taus, profile[:, j], label=opt)

    plt.xlabel(r"$\tau$")
    plt.ylabel("Fraction of problems")
    plt.title("Performance Profiles")
    plt.ylim(0.0, 1.05)
    plt.xlim(1.0, taus[-1])
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute performance profiles from yahpo_results."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="yahpo_results",
        help="Root directory containing {optimizer}/{problem}.json",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=10.0,
        help="Maximum tau value for performance profiles.",
    )
    parser.add_argument(
        "--num-tau",
        type=int,
        default=200,
        help="Number of tau points between 1 and tau-max.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="performance_profiles.png",
        help="Path to save the performance profile plot.",
    )
    args = parser.parse_args()

    # 1. Load best values (performance measure per optimizer/problem)
    best_values = load_best_values(args.root)

    # 2. Compute performance ratios r_{p,a}
    optimizers, problems, ratios = compute_ratios(best_values)
    print(f"Loaded {len(optimizers)} optimizers and {len(problems)} problems.")

    # 3. Choose tau grid
    #    You can also set tau_max to the max finite ratio if you prefer:
    #    tau_max = min(args.tau_max, np.max(ratios[np.isfinite(ratios)]))
    tau_max = args.tau_max
    taus = np.linspace(1.0, tau_max, args.num_tau)

    # 4. Compute performance profiles rho_a(tau)
    profile = compute_performance_profile(ratios, taus)

    # 5. Plot and save
    plot_performance_profiles(optimizers, taus, profile, args.output)
    print(f"Saved performance profile plot to {args.output}")


if __name__ == "__main__":
    main()
