import time
import json
import torch
from transformers import AutoModel, AutoTokenizer

def run_overflow_test():
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        dtype=dtype
    ).to(device)
    model.eval()

    # Load dataset
    with open("data/sample_dataset_fail.json", "r") as f:
        data = json.load(f)

    # Programmatically inject a massive out-of-bounds padding (>2048 tokens)
    massive_padding = "Project Greenhouse aims to build sovereign open-source search agents. " + ("Here is extra padding text to intentionally overflow the 2048 token limit. " * 350)
    data[0]["passages"][1] = massive_padding

    for idx, item in enumerate(data):
        query = item["query"]
        passages = item["passages"]
        
        print(f"\n--- Testing Query {idx+1}: '{query}' ---")
        
        for p_idx, p in enumerate(passages):
            tokens = tokenizer(p, truncation=False)
            token_count = len(tokens['input_ids'])
            print(f"  Passage {p_idx+1} token count: {token_count} tokens (Limit: 2048)")
            if token_count > 2048:
                print(f"  -> WARNING: Passage {p_idx+1} exceeds the 2048-token limit!")

        pairs = [(query, p) for p in passages]
        
        start = time.time()
        with torch.no_grad():
            if hasattr(model, "compute_score"):
                scores = model.compute_score(pairs, tokenizer=tokenizer)
            else:
                scores = [0.0] * len(passages)
        latency = (time.time() - start) * 1000

        print(f"  Gaggle Scores: {scores}")
        print(f"  Execution Latency: {latency:.2f} ms")

if __name__ == "__main__":
    run_overflow_test()