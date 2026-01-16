
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.benchmarks.functions import PROBLEM_CONFIG
from src.algorithms.stochastics import SimulatedAnnealing, ParticleSwarm
from src.algorithms.deterministic import BFGS
from src.visualization.contour_plot import plot_contour_trajectory
from src.visualization.plotting import plot_3d_trajectory

# Constants
EVALUATION_BUDGET = 10000 # Increased to match Task 3 for fair long-run comparison
RUNS = 20 

def plot_log_log_comparison(res_bfgs, res_pso, res_sa, prob_name):
    # Plotting full history with Log-Log sequence
    plt.figure(figsize=(10, 6))
    
    def get_best_hist(results):
        best_idx = np.argmin(results["best_vals"])
        return results["histories"][best_idx]
    
    def extract_xy(hist):
        x = [h[0] for h in hist]
        y = [max(h[1], 1e-16) for h in hist] # Avoid log(0) or negative
        return x, y

    xb, yb = extract_xy(get_best_hist(res_bfgs))
    xp, yp = extract_xy(get_best_hist(res_pso))
    xs, ys = extract_xy(get_best_hist(res_sa))
    
    plt.plot(xb, yb, label=f"BFGS ({yb[-1]:.2e})", color="red", marker="o", markersize=3, linestyle="-")
    plt.plot(xp, yp, label=f"PSO ({yp[-1]:.2e})", color="orange", linestyle="-")
    plt.plot(xs, ys, label=f"SA ({ys[-1]:.2e})", color="magenta", linestyle="-")
    
    plt.title(f"Convergence Comparison: {prob_name} (Log-Log Scale)")
    plt.xlabel("Function Evaluations (Log Scale)")
    plt.ylabel("Objective Value (Log Scale)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    os.makedirs("tasks/Task_04_New_Techniques/results", exist_ok=True)
    plt.savefig(f"tasks/Task_04_New_Techniques/results/{prob_name}_log_log_comparison.png")
    plt.close()

def plot_early_convergence(res_bfgs, res_pso, res_sa, prob_name):
    # ... (Keep existing linear plot for zoomed view)
    # Prepare data
    # Filter histories to first 500 evals or so
    limit = 500
    
    plt.figure(figsize=(10, 6))
    
    # Plot one representative run for trajectory (or mean if interpolated)
    # Let's plot the BEST run of each
    
    def get_best_hist(results):
        best_idx = np.argmin(results["best_vals"])
        return results["histories"][best_idx]
        
    hist_bfgs = get_best_hist(res_bfgs)
    hist_pso = get_best_hist(res_pso)
    hist_sa = get_best_hist(res_sa)
    
    def extract_xy(hist):
        x = [h[0] for h in hist if h[0] < limit]
        y = [h[1] for h in hist if h[0] < limit]
        return x, y
        
    xb, yb = extract_xy(hist_bfgs)
    xp, yp = extract_xy(hist_pso)
    xs, ys = extract_xy(hist_sa)
    
    plt.plot(xb, yb, label="BFGS", color="red", marker="o", markersize=3)
    plt.plot(xp, yp, label="PSO", color="orange")
    plt.plot(xs, ys, label="SA", color="magenta")
    
    plt.title(f"Early Convergence: {prob_name} (First {limit} Evals)")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Value (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    os.makedirs("tasks/Task_04_New_Techniques/results", exist_ok=True)
    plt.savefig(f"tasks/Task_04_New_Techniques/results/{prob_name}_early_convergence.png")
    plt.close()

import argparse

# ... (Imports remain the same)

def run_batch(AlgoClass, name, config, seed_offset=0, **kwargs):
    print(f"Running {name} on {config['name']} (Seed Offset: {seed_offset})...")
    best_vals = []
    times = []
    evals = []
    histories = []
    trajectories = []
    
    for i in range(RUNS):
        # Different seed for each run
        algo = AlgoClass(
            func=config["func"],
            bounds=config["bounds"],
            dim=config["dim"],
            max_evals=EVALUATION_BUDGET,
            seed=i + seed_offset,
            **kwargs
        )
        res = algo.solve()
        best_vals.append(res["best_val"])
        times.append(res["time"])
        evals.append(res["n_evals"])
        histories.append(res["history"])
        trajectories.append(res.get("trajectory")) # Get trajectory
        
    return {
        "best_vals": np.array(best_vals),
        "times": np.array(times), 
        "evals": np.array(evals),
        "histories": histories,
        "trajectories": trajectories
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset to add to random seeds")
    args = parser.parse_args()
    
    seed_offset = args.seed_offset

    probs = ["rastrigin", "rosenbrock"] # Only unconstrained for BFGS comparison
    
    summary = ["# Task 4 Experiment Summary"]
    summary.append(f"**Seed Offset**: {seed_offset}")
    
    
    # Store raw rows for CSV
    all_rows = [["Problem", "Algorithm", "Run", "Best_Val", "Evals", "Time"]]
    
    for p_name in probs:
        config = PROBLEM_CONFIG[p_name]
        config["name"] = p_name
        
        # Helper to process results
        def process_res(name, res):
            for i in range(len(res["best_vals"])):
                all_rows.append([
                    p_name, 
                    name, 
                    i, 
                    res["best_vals"][i], 
                    res["evals"][i], 
                    res["times"][i]
                ])

        # 1. Run BFGS
        res_bfgs = run_batch(BFGS, "BFGS", config, seed_offset=seed_offset, tol=1e-6)
        process_res("BFGS", res_bfgs)
        
        # 2. Run PSO
        res_pso = run_batch(ParticleSwarm, "PSO", config, seed_offset=seed_offset)
        process_res("PSO", res_pso)
        
        # 3. Run SA
        res_sa = run_batch(SimulatedAnnealing, "SA", config, seed_offset=seed_offset)
        process_res("SA", res_sa)
        
        # 4. Plot
        # plot_early_convergence(res_bfgs, res_pso, res_sa, p_name) # Removed as per user request
        plot_log_log_comparison(res_bfgs, res_pso, res_sa, p_name)
        
        # 4b. Contour Trajectory Plot
        def get_best_traj(results):
            best_idx = np.argmin(results["best_vals"])
            return results["trajectories"][best_idx]
            
        trajectories = {
            "BFGS": get_best_traj(res_bfgs),
            "PSO": get_best_traj(res_pso),
            "SA": get_best_traj(res_sa)
        }
        plot_contour_trajectory(config["func"], config["bounds"], trajectories, p_name)
        plot_3d_trajectory(config["func"], config["bounds"], trajectories, p_name)
        
        # 5. Stats
        summary.append(f"\n## Problem: {p_name}")
        summary.append("| Algorithm | Mean Best | Mean Evals | Mean Time |")
        summary.append("|---|---|---|---|")
        
        def row(name, r):
            return f"| {name} | {np.mean(r['best_vals']):.2e} | {np.mean(r['evals']):.1f} | {np.mean(r['times']):.4f} |"
            
        summary.append(row("BFGS", res_bfgs))
        summary.append(row("PSO", res_pso))
        summary.append(row("SA", res_sa))
        
    with open("tasks/Task_04_New_Techniques/results/summary.md", "w") as f:
        f.write("\n".join(summary))
        
    with open("tasks/Task_04_New_Techniques/results/task4_raw.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
        
    print("Done. Results saved to tasks/Task_04_New_Techniques/results/")

if __name__ == "__main__":
    main()
