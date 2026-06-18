import os
import argparse
from benchmarks.runner import run_benchmark
from benchmarks.visualize import generate_all_plots

def main():
    parser = argparse.ArgumentParser(description="Run clustering benchmark suite.")
    parser.add_argument("--test-mode", action="store_true", help="Run only on toy datasets for testing.")
    args = parser.parse_args()

    filter_datasets = ["toy_"] if args.test_mode else None
    
    print("==================================================")
    print(" Algorithm Stress Test & Analysis Benchmark Suite")
    print("==================================================")
    
    # 1. Run Execution Loop
    df_results = run_benchmark(datasets_filter=filter_datasets)
    
    # 2. Generate Plots
    csv_path = os.path.join(os.path.dirname(__file__), 'results', 'benchmark_results.csv')
    out_dir = os.path.join(os.path.dirname(__file__), 'visualize')
    
    if os.path.exists(csv_path):
        generate_all_plots(csv_path, out_dir)
        
    print("\nBenchmark Suite execution completed.")
if __name__ == "__main__":
    main()
