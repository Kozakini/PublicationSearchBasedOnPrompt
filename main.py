#!/usr/bin/env python3
"""Wyszukiwarka publikacji (CPU) + podpowiedzi zapytan do Google Scholar (maly LLM przez Ollame).

Przyklady:
    uv run main.py            # interfejs webowy (Gradio)
    uv run main.py --cli      # tryb terminalowy (tylko wyszukiwanie)

Wymaga suggest.py w tym samym katalogu. Podpowiedzi wymagaja dzialajacej Ollamy
i pobranego modelu (np. ollama pull qwen2.5:3b).
"""
import argparse
import json

import faiss
from sentence_transformers import SentenceTransformer

from suggest import DEFAULT_MODEL, suggest_queries
from suggest import to_markdown as hints_markdown


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


def build_ui(search):
    import gradio as gr

    def run(question, k):
        if not question.strip():
            return "Wpisz pytanie."
        return to_markdown(search(question, k))

    def hints(question, since, model_name):
        if not question.strip():
            yield "Wpisz pytanie."
            return
        yield "Model generuje zapytania (na CPU moze to potrwac do minuty)..."
        try:
            # tytuly z lokalnego indeksu daja modelowi prawdziwe slownictwo dziedziny
            titles = [p["title"] for _, p in search(question, 10)]
            result = suggest_queries(question, titles=titles,
                                     model=(model_name or "").strip() or DEFAULT_MODEL)
            yield hints_markdown(result, since=int(since) if since else None)
        except Exception as e:
            yield f"Nie udalo sie wygenerowac zapytan: {e}"

    with gr.Blocks(title="Wyszukiwarka publikacji naukowych") as demo:
        gr.Markdown("# Wyszukiwarka publikacji naukowych\n"
                    "Zadaj pytanie (po polsku lub angielsku) - dostaniesz prace, "
                    "ktore na nie odpowiadaja. Mozesz tez poprosic o podpowiedz, "
                    "jak szukac dalej w Google Scholar.")
        question = gr.Textbox(
            label="Pytanie",
            placeholder="np. jak sterowac ramieniem robota za pomoca jezyka naturalnego?")
        k = gr.Slider(1, 10, value=5, step=1, label="Liczba wynikow")
        with gr.Row():
            btn = gr.Button("Szukaj", variant="primary")
            hint_btn = gr.Button("Podpowiedz zapytania do Google Scholar")
        with gr.Accordion("Ustawienia podpowiedzi", open=False):
            since = gr.Number(value=2020, precision=0,
                              label="Szukaj prac od roku (0 = bez filtra)")
            model_name = gr.Textbox(value=DEFAULT_MODEL, label="Model Ollama")
        hints_out = gr.Markdown()
        output = gr.Markdown()

        btn.click(run, [question, k], output)
        question.submit(run, [question, k], output)
        hint_btn.click(hints, [question, since, model_name], hints_out)
    return demo


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

    build_ui(search).launch()


if __name__ == "__main__":
    main()
