import json
import os

def display_results():
    results_path = "results/evaluation_results.json"
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}. Run your evaluation script first!")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    print("\n================ SAVED EVALUATION METRICS ================")
    print(f"  Device Used:         {data.get('device', 'N/A')}")
    print(f"  Average BM25 nDCG:   {data.get('avg_bm25_ndcg', 0.0):.4f}")
    print(f"  Average Gaggle nDCG: {data.get('avg_gaggle_ndcg', 0.0):.4f}")
    print(f"  Average Latency:     {data.get('avg_latency_ms', 0.0):.2f} ms")
    print("==========================================================\n")

if __name__ == "__main__":
    display_results()
    