#!/usr/bin/env python3
"""Krok 3: wyszukiwarka publikacji (CPU). Zwraca prace, ktore odpowiadaja na pytanie.

Przyklady:
    python search_app.py            # interfejs webowy (Gradio)
    python search_app.py --cli      # tryb terminalowy
"""
import argparse
import json

import faiss
from sentence_transformers import SentenceTransformer


def make_search(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(args.papers, encoding="utf-8") as f:
        papers = json.load(f)
    index = faiss.read_index(args.index)
    model = SentenceTransformer(cfg["model"], device="cpu")  # celowo CPU
    prefix = "query: " if cfg["use_prefix"] else ""

    def search(question, k=5):
        q = model.encode([prefix + question], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")
        scores, ids = index.search(q, int(k))
        return [(float(s), papers[i]) for s, i in zip(scores[0], ids[0]) if i >= 0]

    return search


def to_markdown(results):
    if not results:
        return "Brak wynikow."
    out = []
    for n, (score, p) in enumerate(results, 1):
        snippet = p["abstract"][:300].rstrip() + "..."
        out.append(
            f"**{n}. [{p['title']}]({p['url']})** ({p['year']})  \n"
            f"{p['authors']} · `{p['categories']}` · podobienstwo: `{score:.3f}`\n\n"
            f"> {snippet}\n"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="papers.json")
    ap.add_argument("--index", default="papers.faiss")
    ap.add_argument("--config", default="index_config.json")
    ap.add_argument("--cli", action="store_true")
    args = ap.parse_args()

    search = make_search(args)

    if args.cli:
        while True:
            q = input("\nPytanie (puste = wyjscie): ").strip()
            if not q:
                break
            for score, p in search(q, 5):
                print(f"{score:.3f}  {p['title']} ({p['year']})\n       {p['url']}")
        return

    import gradio as gr

    def run(question, k):
        if not question.strip():
            return "Wpisz pytanie."
        return to_markdown(search(question, k))

    with gr.Blocks(title="Wyszukiwarka publikacji naukowych") as demo:
        gr.Markdown("# Wyszukiwarka publikacji naukowych\n"
                    "Zadaj pytanie (po polsku lub angielsku) - dostaniesz prace, "
                    "ktore na nie odpowiadaja.")
        question = gr.Textbox(label="Pytanie", placeholder="np. jak sterowac ramieniem robota za pomoca jezyka naturalnego?")
        k = gr.Slider(1, 10, value=5, step=1, label="Liczba wynikow")
        btn = gr.Button("Szukaj")
        output = gr.Markdown()
        btn.click(run, [question, k], output)
        question.submit(run, [question, k], output)
    demo.launch()


if __name__ == "__main__":
    main()
