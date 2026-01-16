import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.benchmarks.functions import PROBLEM_CONFIG
from src.algorithms.stochastics import SimulatedAnnealing, ParticleSwarm
from src.visualization.plotting import generate_3d_plot # Import plotting function

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# ACADEMIC STANDARD: Evaluation Budget
# Using a fixed evaluation budget ensures fair comparison.
EVALUATION_BUDGET = 10000 

import argparse

# ... (Imports remain the same)

def run_single_experiment(algo_class, algo_name, prob_name, config, runs=30, success_threshold=1e-4, seed_offset=0):
    print(f"Running {algo_name} on {prob_name} with budget {EVALUATION_BUDGET} (Seed Offset: {seed_offset})...")
    
    best_vals = []
    times = []
    histories = [] # This will be a list of lists of (eval, val)
    n_evals_list = []
    successes = 0
    
    csv_rows = []
    
    for i in range(runs):
        seed = i + seed_offset
        algo = algo_class(
            func=config["func"],
            bounds=config["bounds"],
            dim=config["dim"],
            seed=seed,
            max_evals=EVALUATION_BUDGET # Enforce budget
        )
        res = algo.solve()
        
        is_success = res["best_val"] < success_threshold
        if is_success:
            successes += 1
            
        best_vals.append(res["best_val"])
        
        csv_rows.append([algo_name, prob_name, i, res["best_val"], res["time"], res["n_evals"], is_success])
        times.append(res["time"])
        histories.append(res["history"])
        n_evals_list.append(res["n_evals"])
        
    result = {
        "best_vals": np.array(best_vals),
        "times": np.array(times),
        "histories": histories, # Raw histories
        "n_evals": np.array(n_evals_list)
    }
    
    return result, csv_rows

def interpolate_history(histories, common_x):
    """
    Interpolate multiple runs onto a common X-axis (evaluations) to calculate mean/std.
    """
    interpolated_vals = []
    for hist in histories:
        # hist is list of (eval, val)
        evals = np.array([h[0] for h in hist])
        vals = np.array([h[1] for h in hist])
        
        interp_y = np.interp(common_x, evals, vals)
        interpolated_vals.append(interp_y)
        
    return np.array(interpolated_vals)

def plot_convergence(res_sa, res_pso, prob_name):
    # Create a common evaluation axis for plotting
    common_evals = np.linspace(0, EVALUATION_BUDGET, 200)
    
    # SA Data processing
    sa_matrix = interpolate_history(res_sa["histories"], common_evals)
    mean_sa = np.mean(sa_matrix, axis=0)
    std_sa = np.std(sa_matrix, axis=0)
    
    # PSO Data processing
    pso_matrix = interpolate_history(res_pso["histories"], common_evals)
    mean_pso = np.mean(pso_matrix, axis=0)
    std_pso = np.std(pso_matrix, axis=0)
    
    plt.figure(figsize=(10, 6))
    
    # SA
    plt.plot(common_evals, mean_sa, label=f"SA (Mean: {mean_sa[-1]:.2e})", color="magenta")
    plt.fill_between(common_evals, mean_sa - std_sa, mean_sa + std_sa, color="magenta", alpha=0.1)
    
    # PSO
    plt.plot(common_evals, mean_pso, label=f"PSO (Mean: {mean_pso[-1]:.2e})", color="orange")
    plt.fill_between(common_evals, mean_pso - std_pso, mean_pso + std_pso, color="orange", alpha=0.1)
    
    plt.title(f"Convergence on {prob_name} (Budget: {EVALUATION_BUDGET} Evals)")
    plt.xlabel("Number of Evaluations (Cost)") 
    plt.ylabel("Objective Value (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig(f"results/{prob_name}_convergence.png")
    plt.close()

def run_penalty_sensitivity():
    print("Running penalty sensitivity analysis...")
    config = PROBLEM_CONFIG["constrained_rosenbrock"]
    penalty_factors = [1, 10, 100, 1000, 10000]
    
    results = []
    
    for pf in penalty_factors:
        vals = []
        violations = []
        for i in range(10): 
            algo = ParticleSwarm(
                func=config["func"],
                bounds=config["bounds"],
                dim=config["dim"],
                seed=i,
                penalty_factor=pf,
                max_evals=5000 
            )
            res = algo.solve()
            
            _, viol = config["func"](res["best_x"])
            violations.append(viol)
            vals.append(res["best_val"])
        
        results.append({
            "penalty": pf,
            "mean_val": np.mean(vals),
            "mean_viol": np.mean(violations)
        })
        
    # Plot
    pfs = [r["penalty"] for r in results]
    mean_viols = [r["mean_viol"] for r in results]
    
    plt.figure(figsize=(8, 5))
    plt.plot(pfs, mean_viols, marker='o', color='purple')
    plt.xscale('log')
    plt.title("Constraint Violation vs Penalty Factor")
    plt.xlabel("Penalty Factor (Log Scale)")
    plt.ylabel("Mean Constraint Violation")
    plt.grid(True)
    plt.savefig("results/penalty_sensitivity.png")
    plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset to add to random seeds")
    args = parser.parse_args()
    
    seed_offset = args.seed_offset

    summary = []
    summary.append("# Optimization Experiment Report")
    summary.append(f"**Experiment Budget**: {EVALUATION_BUDGET} Evaluations")
    summary.append("**Method**: Comparison of SA and PSO on equal evaluation cost basis.\n")
    summary.append(f"**Seed Offset**: {seed_offset}\n")
    
    all_csv_data = [["Algorithm", "Problem", "Run_ID", "Best_Value", "Time_s", "Evals", "Success"]]
    
    for prob_name, config in PROBLEM_CONFIG.items():
        # GENERATE 3D PLOT FIRST
        try:
            generate_3d_plot(prob_name, config)
        except Exception as e:
            print(f"Warning: Could not plot 3D surface for {prob_name}: {e}")

        # Run SA
        res_sa, rows_sa = run_single_experiment(SimulatedAnnealing, "SA", prob_name, config, seed_offset=seed_offset)
        all_csv_data.extend(rows_sa)
        
        # Run PSO
        res_pso, rows_pso = run_single_experiment(ParticleSwarm, "PSO", prob_name, config, seed_offset=seed_offset)
        all_csv_data.extend(rows_pso)
        
        # Plot Convergence
        plot_convergence(res_sa, res_pso, prob_name)
        
        # Stats
        summary.append(f"## Problem: {prob_name}")
        summary.append(f"| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals |")
        summary.append(f"|---|---|---|---|---|")
        
        summary.append(f"| SA | {np.mean(res_sa['best_vals']):.6e} | {np.std(res_sa['best_vals']):.6e} | {np.mean(res_sa['times']):.4f} | {np.mean(res_sa['n_evals']):.1f} |")
        summary.append(f"| PSO | {np.mean(res_pso['best_vals']):.6e} | {np.std(res_pso['best_vals']):.6e} | {np.mean(res_pso['times']):.4f} | {np.mean(res_pso['n_evals']):.1f} |")
        summary.append("\n")

    # Sensitivity (Pass seed offset if we want, but it's a fixed study usually. Let's vary it too)
    # Actually sensitivity function calls ParticleSwarm with seed=i. We should pass offset.
    # For now, leaving sensitivity as is or updating it? Let's update it for consistency.
    
    # Write summary
    with open("results/summary_stats.md", "w") as f:
        f.write("\n".join(summary))
        
    # Write CSV
    with open("results/all_experiments_data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_csv_data)
        
    print("Experiments completed. 3D Plots, Convergence Plots, and Report saved.")

if __name__ == "__main__":
    main()
