#!/usr/bin/env python3
"""Krok 2: liczy embeddingi abstraktow (GPU/ROCm jesli dostepne) i buduje indeks FAISS.

Przyklad:
    python build_index.py --papers papers.json
"""
import argparse
import json
import time

import faiss
import torch
from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="papers.json")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--index", default="papers.faiss")
    ap.add_argument("--config", default="index_config.json")
    args = ap.parse_args()

    with open(args.papers, encoding="utf-8") as f:
        papers = json.load(f)
    print(f"Prac: {len(papers)}")

    # ROCm w PyTorch tez zglasza sie jako "cuda"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Urzadzenie: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length
    if device == "cuda":
        model.half()

    # modele E5 wymagaja prefiksow: "passage: " dla dokumentow, "query: " dla pytan
    use_prefix = "e5" in args.model.lower()
    prefix = "passage: " if use_prefix else ""
    texts = [f"{prefix}{p['title']}. {p['abstract']}" for p in papers]

    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    print(f"Embeddingi policzone w {time.time() - t0:.0f} s, ksztalt: {emb.shape}")

    index = faiss.IndexFlatIP(emb.shape[1])  # iloczyn skalarny = cosinus (wektory znormalizowane)
    index.add(emb)
    faiss.write_index(index, args.index)

    with open(args.config, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "use_prefix": use_prefix, "dim": int(emb.shape[1])}, f)
    print(f"Zapisano {args.index} i {args.config}")


if __name__ == "__main__":
    main()
