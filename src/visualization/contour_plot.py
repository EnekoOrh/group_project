
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_contour_trajectory(func, bounds, trajectories, prob_name):
    """
    Generates a 2D filled contour plot of the function and overlays the search trajectories.
    
    Args:
        func: The function to plot.
        bounds: List of [min, max] for each dimension (assumes 2D).
        trajectories: Dict of {AlgoName: trajectory_array}.
        prob_name: Name of the problem (for title/filename).
    """
    if len(bounds) != 2:
        print("Contour plot only supported for 2D problems.")
        return

    # 1. Create Grid
    # Handle different bound formats
    # Case A: bounds = [-5, 5] (Scalar bounds for all dims)
    if isinstance(bounds[0], (int, float)):
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[0], bounds[1]
    # Case B: bounds = [[-5, 5], [-5, 5]] (Per-dimension bounds)
    else:
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
    
    # Add some padding for better visualization
    pad = (x_max - x_min) * 0.1
    x = np.linspace(x_min - pad, x_max + pad, 200)
    y = np.linspace(y_min - pad, y_max + pad, 200)
    X, Y = np.meshgrid(x, y)
    
    # Vectorized evaluation
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])
            
    # Apply Log Scale for Z if range is huge (like Rosenbrock)
    if "rosenbrock" in prob_name.lower():
        Z = np.log10(Z + 1e-10)
        z_label = "Log10(Cost)"
    else:
        z_label = "Cost"

    plt.figure(figsize=(10, 8))
    
    # 2. Contour Plot
    # Use 'viridis' or 'plasma' for premium look
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label=z_label)
    
    # 3. Trajectories
    colors = {"BFGS": "red", "PSO": "orange", "SA": "white"}
    styles = {"BFGS": "-", "PSO": "--", "SA": ":"}
    
    for name, traj in trajectories.items():
        if traj is None or len(traj) == 0:
            continue
            
        # Plot Path
        plt.plot(traj[:, 0], traj[:, 1], 
                 label=name, 
                 color=colors.get(name, "black"), 
                 linestyle=styles.get(name, "-"),
                 linewidth=1.5,
                 alpha=0.9)
        
        # Plot Start Point
        plt.plot(traj[0, 0], traj[0, 1], 'x', color=colors.get(name, "black"))
        
        # Plot End Point
        plt.plot(traj[-1, 0], traj[-1, 1], 'o', color=colors.get(name, "black"), markersize=5)

    plt.title(f"Search Trajectory: {prob_name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    
    os.makedirs("tasks/Task_04_New_Techniques/results", exist_ok=True)
    plt.savefig(f"tasks/Task_04_New_Techniques/results/{prob_name}_trajectory_contour.png", dpi=300)
    plt.close()
