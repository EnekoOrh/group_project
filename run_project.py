
import os
import sys
import shutil
import argparse
import subprocess
import time

def clean_results(task_dir):
    """
    Removes the 'results' directory within a specific task directory.
    """
    results_dir = os.path.join(task_dir, "results")
    if os.path.exists(results_dir):
        print(f"Cleaning results in {task_dir}...")
        try:
            shutil.rmtree(results_dir)
            print(f"  - Deleted {results_dir}")
        except Exception as e:
            print(f"  - Error deleting {results_dir}: {e}")
    else:
        print(f"No results to clean in {task_dir}")

def run_script(script_path):
    """
    Runs a python script using the current interpreter.
    """
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Running {script_path}...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        # Run using the same python executable
        subprocess.check_call([sys.executable, script_path])
        duration = time.time() - start_time
        print(f"\n{'-'*60}")
        print(f"Finished {script_path} in {duration:.2f}s")
        print(f"{'-'*60}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Group Project Experiments and Generate Results")
    parser.add_argument("--task", type=str, choices=["3", "4", "all"], default="all",
                        help="Specify which task to run (3, 4, or all)")
    parser.add_argument("--no-clean", action="store_true", 
                        help="Skip cleaning the results directory before running")
    
    parser.add_argument("--random", action="store_true", 
                        help="Use a random seed offset for fresh data generation")
                        
    parser.add_argument("--seed", type=int, default=None,
                        help="Specify a seed offset (integer). Overrides --random.")

    args = parser.parse_args()
    
    # Generate random seed offset if requested
    seed_offset = 0
    if args.seed is not None:
         seed_offset = args.seed
         print(f"Using Specified Seed Offset: {seed_offset}")
    elif args.random:
        import random
        seed_offset = random.randint(1000, 99999)
        print(f"Using Random Seed Offset: {seed_offset}")
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define tasks configuration
    tasks = {
        "3": {
            "dir": os.path.join(base_dir, "tasks", "Task_03_Comparison"),
            "script": "run_experiments.py"
        },
        "4": {
            "dir": os.path.join(base_dir, "tasks", "Task_04_New_Techniques"),
            "script": "run_experiments.py"
        }
    }
    
    # Determine which tasks to run
    tasks_to_run = []
    if args.task == "all":
        tasks_to_run = ["3", "4"]
    else:
        tasks_to_run = [args.task]
        
    print(f"Starting Project Execution for Task(s): {', '.join(tasks_to_run)}")
    
    for task_id in tasks_to_run:
        task_config = tasks[task_id]
        task_dir = task_config["dir"]
        script_name = task_config["script"]
        script_path = os.path.join(task_dir, script_name)
        
        # 1. Clean (unless skipped)
        if not args.no_clean:
            clean_results(task_dir)
            
        # 2. Run
        cmd = [sys.executable, script_path, "--seed-offset", str(seed_offset)]
        print(f"\n{'='*60}")
        print(f"Running {script_path} with seed offset {seed_offset}...")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        try:
            subprocess.check_call(cmd)
            duration = time.time() - start_time
            print(f"\n{'-'*60}")
            print(f"Finished {script_path} in {duration:.2f}s")
            print(f"{'-'*60}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_path}: {e}")
            sys.exit(1)
            
    print("\nAll requested tasks completed successfully.")

if __name__ == "__main__":
    main()
