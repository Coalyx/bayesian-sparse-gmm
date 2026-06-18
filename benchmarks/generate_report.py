import sys
import logging
from benchmarks.data_pipeline import DataPipelineOrchestrator, compute_statistics, generate_markdown_report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    o = DataPipelineOrchestrator(data_dir="./data")
    datasets = o.load_all()
    stats = compute_statistics(datasets)
    generate_markdown_report(stats, "DATA_REPORT.md")
    print("DONE")
