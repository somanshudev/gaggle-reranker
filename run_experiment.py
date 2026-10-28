import time
import json
import torch
from sklearn.metrics import ndcg_score
from transformers import AutoModel, AutoTokenizer

def run_benchmark():
    # Detect Apple Silicon M2 MPS GPU or CPU fallback
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("MPS not available, falling back to CPU")

    model_name = "castorini/gaggle-reranker-20261005"
    print(f"Loading {model_name}...")

    # Use AutoModel instead of AutoModelForCausalLM for custom reranker architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        dtype=dtype
    ).to(device)
    model.eval()

    with open("data/sample_dataset.json", "r") as f:
        data = json.load(f)

    all_bm25_ndcg = []
    all_gaggle_ndcg = []
    latencies = []

    for idx, item in enumerate(data):
        query = item["query"]
        passages = item["passages"]
        ground_truth = [item["relevance"]]

        # Simulated BM25 order (passages in initial dataset order)
        bm25_simulated_scores = [[len(passages) - i for i in range(len(passages))]]
        bm25_ndcg = ndcg_score(ground_truth, bm25_simulated_scores, k=len(passages))
        all_bm25_ndcg.append(bm25_ndcg)

        # Evaluate with Gaggle Reranker
        start = time.time()
        pairs = [(query, p) for p in passages]
        
        with torch.no_grad():
            # Call compute_score or compute logit difference directly
            if hasattr(model, "compute_score"):
                scores = model.compute_score(pairs, tokenizer=tokenizer)
            else:
                # Fallback: compute scores via query-passage input encoding
                inputs = tokenizer([f"Query: {q} Document: {p}" for q, p in pairs], return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model(**inputs)
                scores = outputs.logits[:, -1].cpu().tolist() if hasattr(outputs, "logits") else [0.0] * len(pairs)
        
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        gaggle_ndcg = ndcg_score(ground_truth, [scores], k=len(passages))
        all_gaggle_ndcg.append(gaggle_ndcg)

        print(f"\nQuery {idx+1}: '{query}'")
        print(f"  BM25 nDCG:   {bm25_ndcg:.4f}")
        print(f"  Gaggle nDCG: {gaggle_ndcg:.4f}")
        print(f"  Latency:     {latency:.2f} ms")

    results = {
        "avg_bm25_ndcg": float(sum(all_bm25_ndcg) / len(all_bm25_ndcg)),
        "avg_gaggle_ndcg": float(sum(all_gaggle_ndcg) / len(all_gaggle_ndcg)),
        "avg_latency_ms": float(sum(latencies) / len(latencies)),
        "device": str(device)
    }

    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n================ SUMMARY RESULTS ================")
    print(f"Average BM25 nDCG:   {results['avg_bm25_ndcg']:.4f}")
    print(f"Average Gaggle nDCG: {results['avg_gaggle_ndcg']:.4f}")
    print(f"Average Latency:     {results['avg_latency_ms']:.2f} ms")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()