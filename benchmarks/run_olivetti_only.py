import argparse
from evaluate import run_olivetti_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda", type=str)
    args = parser.parse_args()
    
    run_olivetti_benchmark(backend=args.backend)
