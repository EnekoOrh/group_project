
import os
import numpy as np
import matplotlib.pyplot as plt
from test_functions import PROBLEM_CONFIG
from optimization_algorithms import SimulatedAnnealing, ParticleSwarm

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def run_single_experiment(algo_class, algo_name, prob_name, config, runs=10):
    print(f"Running {algo_name} on {prob_name}...")
    
    best_vals = []
    times = []
    histories = []
    n_evals = []
    
    for i in range(runs):
        # Different seed for each run to evaluate stochastic performance
        seed = i 
        algo = algo_class(
            func=config["func"],
            bounds=config["bounds"],
            dim=config["dim"],
            seed=seed,
            max_iter=500
        )
        res = algo.solve()
        best_vals.append(res["best_val"])
        times.append(res["time"])
        histories.append(res["history"])
        n_evals.append(res["n_evals"])
        
    return {
        "best_vals": np.array(best_vals),
        "times": np.array(times),
        "histories": np.array(histories), # Shape: (runs, iter)
        "n_evals": np.array(n_evals)
    }

def plot_convergence(histories_sa, histories_pso, prob_name):
    # Calculate mean and std
    mean_sa = np.mean(histories_sa, axis=0)
    std_sa = np.std(histories_sa, axis=0)
    
    mean_pso = np.mean(histories_pso, axis=0)
    std_pso = np.std(histories_pso, axis=0)
    
    iterations = np.arange(len(mean_sa))
    
    plt.figure(figsize=(10, 6))
    
    # SA
    plt.plot(iterations, mean_sa, label="SA (Mean)", color="blue")
    plt.fill_between(iterations, mean_sa - std_sa, mean_sa + std_sa, color="blue", alpha=0.1)
    
    # PSO
    plt.plot(iterations, mean_pso, label="PSO (Mean)", color="orange")
    plt.fill_between(iterations, mean_pso - std_pso, mean_pso + std_pso, color="orange", alpha=0.1)
    
    plt.title(f"Convergence on {prob_name}")
    plt.xlabel("Iterations")
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
    
    # We will use PSO for this as it was more effective
    results = []
    
    for pf in penalty_factors:
        # Run 5 times each to get a stable estimate
        vals = []
        violations = []
        for i in range(5):
            algo = ParticleSwarm(
                func=config["func"],
                bounds=config["bounds"],
                dim=config["dim"],
                seed=i,
                penalty_factor=pf,
                max_iter=200
            )
            res = algo.solve()
            # Decompose value back to obj + penalty to find violation
            # val = obj + pf * viol^2 -> this is hard to invert blindly if we don't return violating
            # simpler to re-evaluate the violation of the best_x
            
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
    plt.plot(pfs, mean_viols, marker='o')
    plt.xscale('log')
    plt.title("Effect of Penalty Factor on Constraint Violation (PSO)")
    plt.xlabel("Penalty Factor")
    plt.ylabel("Mean Constraint Violation")
    plt.grid(True)
    plt.savefig("results/penalty_sensitivity.png")
    plt.close()
    
    return results

def main():
    summary = []
    
    for prob_name, config in PROBLEM_CONFIG.items():
        # Run SA
        res_sa = run_single_experiment(SimulatedAnnealing, "SA", prob_name, config)
        
        # Run PSO
        res_pso = run_single_experiment(ParticleSwarm, "PSO", prob_name, config)
        
        # Plot
        plot_convergence(res_sa["histories"], res_pso["histories"], prob_name)
        
        # Stats
        summary.append(f"## Problem: {prob_name}")
        summary.append(f"| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals |")
        summary.append(f"|---|---|---|---|---|")
        
        summary.append(f"| SA | {np.mean(res_sa['best_vals']):.6e} | {np.std(res_sa['best_vals']):.6e} | {np.mean(res_sa['times']):.4f} | {np.mean(res_sa['n_evals']):.1f} |")
        summary.append(f"| PSO | {np.mean(res_pso['best_vals']):.6e} | {np.std(res_pso['best_vals']):.6e} | {np.mean(res_pso['times']):.4f} | {np.mean(res_pso['n_evals']):.1f} |")
        summary.append("\n")

    # Sensitivity
    sens_results = run_penalty_sensitivity()
    summary.append("## Penalty Sensitivity (Constrained Rosenbrock - PSO)")
    summary.append("| Penalty Factor | Mean Best Val | Mean Violation |")
    summary.append("|---|---|---|")
    for r in sens_results:
        summary.append(f"| {r['penalty']} | {r['mean_val']:.6e} | {r['mean_viol']:.6e} |")
        
    # Write summary
    with open("results/summary_stats.md", "w") as f:
        f.write("\n".join(summary))
        
    print("Experiments completed. Check 'results' directory.")

if __name__ == "__main__":
    main()
