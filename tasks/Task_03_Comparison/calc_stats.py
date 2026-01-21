
import csv
import os
import statistics

def main():
    data = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    csv_file = os.path.join(results_dir, 'all_experiments_data.csv')
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['Problem'], row['Algorithm'])
            if key not in data:
                data[key] = {'val': [], 'time': [], 'success': []}
            data[key]['val'].append(float(row['Best_Value']))
            data[key]['time'].append(float(row['Time_s']))
            data[key]['success'].append(1 if row['Success'] == 'True' else 0)

    output_file = os.path.join(results_dir, 'stats_output_clean.txt')
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("| Problem | Algorithm | Mean Best | Std Dev | Mean Time | Success Rate |\n")
        outfile.write("|---|---|---|---|---|---|\n")
        
        # Sort keys for consistent output
        sorted_keys = sorted(data.keys())
        
        for k in sorted_keys:
            vals = data[k]['val']
            times = data[k]['time']
            successes = data[k]['success']
            
            mean_val = statistics.mean(vals)
            std_val = statistics.stdev(vals) if len(vals) > 1 else 0
            mean_time = statistics.mean(times)
            success_rate = statistics.mean(successes) * 100
            
            outfile.write(f"| {k[0]} | {k[1]} | {mean_val:.4e} | {std_val:.4e} | {mean_time:.4f} | {success_rate:.1f}% |\n")

if __name__ == "__main__":
    main()
