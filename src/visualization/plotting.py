import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.benchmarks.functions import PROBLEM_CONFIG

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def generate_3d_plot(prob_name, config):
    print(f"Generating 3D plot for {prob_name}...")
    
    # Extract config
    func = config["func"]
    bounds = config["bounds"]
    
    # Create meshgrid
    # If bounds is single list [-5, 5], assume same for both dims
    if len(bounds) == 2 and isinstance(bounds[0], (int, float)):
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[0], bounds[1]
    else:
        # Assume [[x_min, x_max], [y_min, y_max]]
        x_min, x_max = bounds[0][0], bounds[0][1]
        y_min, y_max = bounds[1][0], bounds[1][1]
        
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Vectorize evaluation
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # func expects 1D array
            val = func([X[i, j], Y[i, j]])
            if isinstance(val, tuple):
                # For constrained, just plot the objective value for now, 
                # or maybe the penalized value? Let's stick to objective value 
                # to visualize the landscape.
                Z[i, j] = val[0] 
            else:
                Z[i, j] = val
                
    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add contour for better visualization of minima
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)

    ax.set_title(f"3D Surface of {prob_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Objective Value")
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    output_file = f"results/{prob_name}_3d_surface.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved to {output_file}")

def main():
    for prob_name, config in PROBLEM_CONFIG.items():
        try:
            generate_3d_plot(prob_name, config)
        except Exception as e:
            print(f"Failed to plot {prob_name}: {e}")

if __name__ == "__main__":
    main()

def plot_3d_trajectory(func, bounds, trajectories, prob_name):
    """
    Generates a 3D surface plot with optimization trajectories overlaid.
    """
    print(f"Generating 3D trajectory plot for {prob_name}...")
    
    # 1. Setup Grid (Same as generate_3d_plot)
    if len(bounds) == 2 and isinstance(bounds[0], (int, float)):
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[0], bounds[1]
    else:
        x_min, x_max = bounds[0][0], bounds[0][1]
        y_min, y_max = bounds[1][0], bounds[1][1]
        
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            val = func([X[i, j], Y[i, j]])
            if isinstance(val, tuple):
                Z[i, j] = val[0]
            else:
                Z[i, j] = val

    # 2. Plot Surface
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with transparency
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)
    
    # Add contour on floor
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.3)
    
    # 3. Plot Trajectories onto Surface
    colors = {"BFGS": "red", "PSO": "orange", "SA": "magenta"}

    for name, traj in trajectories.items():
        if traj is None or len(traj) == 0:
            continue
            
        # We need Z values for the trajectory path
        # Re-evaluate function along trajectory
        traj_z = []
        for point in traj:
            val = func(point)
            if isinstance(val, tuple):
                traj_z.append(val[0])
            else:
                traj_z.append(val)
        traj_z = np.array(traj_z)
        
        # Lift trajectory slightly above surface to avoid clipping
        lift = (np.max(Z) - np.min(Z)) * 0.01
        
        ax.plot(traj[:, 0], traj[:, 1], traj_z + lift, 
                label=f"{name} ({traj_z[-1]:.2e})", 
                color=colors.get(name, "black"), 
                linewidth=2,
                alpha=0.9,
                zorder=10) # High zorder
                
        # Start/End
        ax.scatter(traj[0, 0], traj[0, 1], traj_z[0] + lift, color=colors.get(name, "black"), marker='x', s=50)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj_z[-1] + lift, color=colors.get(name, "black"), marker='o', s=50)

    ax.set_title(f"3D Search Trajectory: {prob_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Objective Value")
    ax.legend()
    # Adjust view
    ax.view_init(elev=30, azim=45)
    
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    output_dir = "tasks/Task_04_New_Techniques/results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{prob_name}_trajectory_3d.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved 3D trajectory to {output_file}")
