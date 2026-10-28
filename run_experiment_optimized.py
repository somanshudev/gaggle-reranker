import time
import json
import torch
from transformers import AutoModel, AutoTokenizer

def chunk_text(text, tokenizer, max_tokens=512, overlap=64):
    """Splits long text into small, safe token windows under the 2048 limit."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

def run_optimized_experiment():
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

    query = "What is the primary objective of Project Greenhouse?"
    
    # Normal short passage
    short_passage = "Project Greenhouse aims to build sovereign open-source search agents and rerankers."
    
    # Oversized passage (without measuring token length raw to prevent model warning)
    massive_passage = "Project Greenhouse aims to build sovereign open-source search agents. " + ("Here is extra padding text to intentionally overflow the 2048 token limit. " * 150)
    
    print("\n--- Running Safe Sliding-Window Chunking Strategy ---")
    
    # Chunk into safe 512-token blocks
    chunks = chunk_text(massive_passage, tokenizer, max_tokens=512, overlap=64)
    print(f"  Massive passage split into {len(chunks)} safe chunks (<512 tokens each).")

    start = time.time()
    with torch.no_grad():
        def evaluate_pair(q, p):
            inputs = tokenizer(q, p, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # Handle direct Tensor outputs vs dict/ModelOutput structures
            if isinstance(outputs, torch.Tensor):
                score = outputs.mean().item()
            elif hasattr(outputs, "logits") and outputs.logits is not None:
                score = outputs.logits.squeeze().item()
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                score = outputs.last_hidden_state.mean().item()
            else:
                score = 0.0
            return float(score)

        # Score the short passage
        short_score = evaluate_pair(query, short_passage)

        # Score each chunk independently
        chunk_scores = [evaluate_pair(query, c) for c in chunks]
        
        # Max-pooling across windows
        final_massive_score = max(chunk_scores) if chunk_scores else 0.0

    latency = (time.time() - start) * 1000

    print(f"\n  RESULTS:")
    print(f"  Short Passage Score:          {short_score:.4f}")
    print(f"  Chunk Scores for Long Doc:   {[round(s, 4) for s in chunk_scores]}")
    print(f"  Final Aggregated Score:       {final_massive_score:.4f}")
    print(f"  Total Inference Latency:      {latency:.2f} ms")
    print("\nSUCCESS: Sliding-window chunking executed successfully!")

if __name__ == "__main__":
    run_optimized_experiment()