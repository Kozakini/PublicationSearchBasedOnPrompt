#!/usr/bin/env python3
"""Generator zapytan do Google Scholar (maly LLM przez Ollame, CPU) - wersja 2.

Podzial pracy:
  * LLM robi to, w czym jest dobry: rozbija temat na pojecia i dobiera slownictwo
    naukowe (angielskie terminy, synonimy), uziemione w tytulach z lokalnego indeksu.
  * Skladnie zapytan do Google Scholar buduje zwykly kod wedlug dobrych praktyk
    wyszukiwania (rdzen tematu, synonimy, przeglady, intitle, wykluczenia).
    Dzieki temu maly model nie psuje skladni, a zapytania sa przewidywalne.

Uzycie z terminala:
    uv run suggest.py "jak gry uzalezniaja graczy"
    uv run suggest.py "sterowanie robotem jezykiem naturalnym" --since 2022 --model qwen2.5:3b
    uv run suggest.py "temat" --debug          # pokaz surowa odpowiedz modelu

Uzycie z kodu (np. w Gradio):
    from suggest import suggest_queries, to_markdown
    result = suggest_queries(pytanie, titles=lista_tytulow_z_indeksu)
    markdown = to_markdown(result, since=2020)

Wymaga uruchomionej Ollamy i pobranego modelu, np.:  ollama pull qwen2.5:3b
"""
import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = """You are a research librarian. You turn a student's topic (in any language) into the building blocks of a good literature search. You do NOT write search queries: you output concepts and vocabulary, and software assembles the queries.

Output ONLY a JSON object:
{"topic_en": "...", "concepts": [{"name": "...", "terms": ["preferred term", "synonym", "related term"]}], "exclude": ["..."]}

Best practices you must follow:
1. Translate the topic to English. Replace everyday words with the academic terminology that authors actually use in titles and abstracts.
2. Split the topic into 2-3 distinct, non-overlapping concepts (what is studied; in what context or with what method; optionally a third aspect).
3. For each concept give 2-4 terms. The FIRST term is the most common standard term in the literature (a 1-3 word noun phrase). The others are real synonyms, alternative names, or acronyms.
4. Prefer established terminology over invented phrases. Use noun phrases, never sentences, questions or verbs.
5. "exclude": 0-2 single words that typically bring irrelevant results for this topic (e.g. another meaning of an ambiguous term). Use [] if unsure.
6. If RELATED TITLES are provided, reuse their terminology when it matches the topic; ignore them when they are unrelated.
7. Never invent paper titles, authors or numbers."""

EXAMPLES = [
    ("jak gry uzależniają graczy", {
        "topic_en": "how video games addict players",
        "concepts": [
            {"name": "gaming addiction",
             "terms": ["gaming disorder", "internet gaming disorder", "problematic video game use"]},
            {"name": "game design mechanics",
             "terms": ["reward schedule", "variable ratio reinforcement", "loot boxes", "compulsion loop"]},
        ],
        "exclude": [],
    }),
    ("sterowanie ramieniem robota za pomocą języka naturalnego", {
        "topic_en": "controlling a robot arm with natural language",
        "concepts": [
            {"name": "robot arm control",
             "terms": ["robot manipulation", "robotic arm", "manipulator control"]},
            {"name": "natural language interface",
             "terms": ["natural language instructions", "large language models", "language grounding"]},
        ],
        "exclude": [],
    }),
    ("diagnostyka łożysk silników elektrycznych sieciami neuronowymi", {
        "topic_en": "bearing fault diagnosis in electric motors using neural networks",
        "concepts": [
            {"name": "bearing fault diagnosis",
             "terms": ["bearing fault diagnosis", "condition monitoring", "fault detection", "predictive maintenance"]},
            {"name": "neural networks",
             "terms": ["deep learning", "convolutional neural network", "neural network"]},
            {"name": "electric motors",
             "terms": ["induction motor", "electric machine"]},
        ],
        "exclude": ["biological"],
    }),
]

# etykieta strategii -> krotkie uzasadnienie (po polsku, bez udzialu LLM)
WHY = {
    "Rdzeń tematu": "Szeroki start: dwa główne pojęcia w standardowym brzmieniu.",
    "Synonimy": "Łapie prace, które nazywają to samo inaczej (OR między synonimami).",
    "Alternatywna terminologia": "Te same pojęcia opisane innymi terminami - inny nurt słownictwa.",
    "Zawężenie": "Dokłada trzecie pojęcie, żeby zawęzić wyniki.",
    "Tytuł zawiera pojęcie": "Wymusza główne pojęcie w tytule - mniej przypadkowych trafień.",
    "Przeglądy": "Prace przeglądowe dają mapę tematu i listę kolejnych źródeł.",
    "Bez szumu": "Odcina typowe nietrafne znaczenia (operator minus).",
}

PLAN_STEPS = [
    "Zacznij od zapytania „Przeglądy”: wybierz 1-2 świeże prace typu survey/review - dają mapę tematu.",
    "Zbierz słownictwo: z tytułów, abstraktów i słów kluczowych trafnych prac wypisz nowe terminy i wróć do zapytań z innymi synonimami.",
    "Snowballing: w Google Scholar użyj „Cytowane przez” (do przodu), bibliografii pracy (wstecz) i „Podobne artykuły”.",
    "Zawęź lata i sprawdź, czy źródła są recenzowane, a nie tylko preprintami.",
    "Notuj użyte zapytania i daty - ułatwia to powtórzenie wyszukiwania i opisanie metody w pracy.",
]


def ollama_chat(messages, model):
    body = json.dumps({
        "model": model, "messages": messages, "stream": False, "format": "json",
        # num_ctx: domyslne okno Ollamy bywa za male na prompt z przykladami
        "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 700},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(f"Nie mozna polaczyc sie z Ollama pod {OLLAMA_URL} "
                           f"(czy dziala i czy model '{model}' jest pobrany?): {e}") from e


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def clean_term(term):
    """Czysci pojedynczy termin: bez cudzyslowow, nawiasow, wiodacych +/- i zbednych spacji."""
    term = re.sub(r"[\"()]", " ", str(term))
    term = re.sub(r"\s+", " ", term).strip(" '+-")
    return term


def quote(term):
    return f'"{term}"' if " " in term else term


def sanitize_query(q):
    """Zabezpieczenie skladni Google Scholar: bez nawiasow, wiodacego +, slowa AND i niedomknietych cudzyslowow."""
    q = q.replace("(", " ").replace(")", " ")
    q = re.sub(r"(^|\s)\+", " ", q)
    q = re.sub(r"\bAND\b", " ", q)
    if q.count('"') % 2:
        q = q.replace('"', "")
    return re.sub(r"\s+", " ", q).strip()


def normalize(data):
    """Sprowadza odpowiedz modelu do czystej struktury pojec; odporne na typowe bledy malych modeli."""
    concepts = []
    for c in data.get("concepts", []):
        if not isinstance(c, dict):
            continue
        raw_terms = c.get("terms") or c.get("synonyms") or []
        if isinstance(raw_terms, str):
            raw_terms = [raw_terms]
        terms, seen = [], set()
        for t in raw_terms:
            t = clean_term(t)
            if t and t.lower() not in seen:
                seen.add(t.lower())
                terms.append(t)
        if terms:
            concepts.append({"name": str(c.get("name") or terms[0]), "terms": terms[:4]})
    exclude = []
    for e in data.get("exclude", []) or []:
        e = clean_term(e)
        if e and e.lower() not in {x.lower() for x in exclude}:
            exclude.append(e)
    return {
        "topic_en": str(data.get("topic_en") or "").strip(),
        "concepts": concepts[:3],
        "exclude": exclude[:2],
    }


def build_queries(concepts, exclude=None):
    """Deterministycznie sklada zapytania do Google Scholar z pojec i synonimow."""
    cs = [c for c in concepts if c["terms"]]
    if not cs:
        return []
    c1 = cs[0]["terms"]
    c2 = cs[1]["terms"] if len(cs) > 1 else []
    c3 = cs[2]["terms"] if len(cs) > 2 else []

    core_parts = [quote(c1[0])] + ([quote(c2[0])] if c2 else [])
    core = " ".join(core_parts)

    queries = []

    def add(strategy, query):
        query = sanitize_query(query)
        if query and all(x["query"].lower() != query.lower() for x in queries):
            queries.append({"strategy": strategy, "query": query, "why": WHY[strategy]})

    add("Rdzeń tematu", core)
    if len(c1) > 1:
        add("Synonimy", " ".join(([quote(c2[0])] if c2 else []) + [f"{quote(c1[0])} OR {quote(c1[1])}"]))
    alt = [quote(c1[1] if len(c1) > 1 else c1[0])]
    if c2:
        alt.append(quote(c2[1] if len(c2) > 1 else c2[0]))
    add("Alternatywna terminologia", " ".join(alt))
    if c3:
        add("Zawężenie", f"{core} {quote(c3[0])}")
    add("Tytuł zawiera pojęcie", f"intitle:{quote(c1[0])} " + (quote(c2[0]) if c2 else ""))
    add("Przeglądy", f"{core} review OR survey")
    if exclude:
        add("Bez szumu", core + " " + " ".join("-" + quote(e) for e in exclude[:2]))
    return queries


def suggest_queries(question, titles=None, model=DEFAULT_MODEL, debug=False):
    user = f"Topic: {question}"
    if titles:
        user += "\n\nRELATED TITLES (may be unrelated):\n" + "\n".join(f"- {t}" for t in titles)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_q, ex_a in EXAMPLES:
        messages.append({"role": "user", "content": f"Topic: {ex_q}"})
        messages.append({"role": "assistant", "content": json.dumps(ex_a, ensure_ascii=False)})
    messages.append({"role": "user", "content": user})

    raw = ollama_chat(messages, model)
    if debug:
        print("--- surowa odpowiedz modelu ---\n" + raw + "\n-------------------------------")
    result = normalize(parse_json(raw))
    if not result["concepts"]:
        raise ValueError("Model nie zwrocil uzytecznych pojec - sprobuj inaczej sformulowac temat "
                         "albo uzyc wiekszego modelu.")
    result["queries"] = build_queries(result["concepts"], result["exclude"])
    return result


def scholar_url(query, since=None):
    params = {"q": query, "hl": "en"}
    if since:
        params["as_ylo"] = str(since)
    return "https://scholar.google.com/scholar?" + urllib.parse.urlencode(params)


def to_markdown(result, since=None):
    lines = []
    if result.get("topic_en"):
        lines.append(f"**Temat (EN):** {result['topic_en']}\n")
    lines.append("### Pojęcia i słownictwo")
    for c in result["concepts"]:
        lines.append(f"- **{c['name']}**: {', '.join(c['terms'])}")
    if result.get("exclude"):
        lines.append(f"- *Wykluczenia:* {', '.join(result['exclude'])}")
    lines.append("\n### Zapytania do Google Scholar (kliknij, żeby wyszukać)")
    for n, item in enumerate(result["queries"], 1):
        lines.append(f"{n}. **{item['strategy']}**: [`{item['query']}`]({scholar_url(item['query'], since)})  \n"
                     f"   {item['why']}")
    lines.append("\n### Plan dalszego szukania")
    lines += [f"{n}. {s}" for n, s in enumerate(PLAN_STEPS, 1)]
    return "\n".join(lines)


def is_lfs_pointer(path):
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return False


def load_related_titles(question, k=10, config="index_config.json",
                        papers="papers.json", index="papers.faiss"):
    """Tytuly najblizszych prac z lokalnego indeksu (uziemienie slownictwa). Przy problemach zwraca []."""
    if not all(os.path.exists(p) for p in (config, papers, index)):
        return []
    for p in (config, papers, index):
        if is_lfs_pointer(p):
            print(f"Uwaga: {p} to wskaznik Git LFS, a nie dane (uruchom: git lfs pull). "
                  "Pomijam uziemienie.")
            return []
    try:
        with open(config, encoding="utf-8") as f:
            cfg = json.load(f)
        with open(papers, encoding="utf-8") as f:
            data = json.load(f)
        import faiss
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg["model"], device="cpu")
        prefix = "query: " if cfg.get("use_prefix") else ""
        q = model.encode([prefix + question], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")
        _, ids = faiss.read_index(index).search(q, k)
        return [data[i]["title"] for i in ids[0] if i >= 0]
    except Exception as e:
        print(f"Uwaga: nie udalo sie wczytac indeksu ({e}). Pomijam uziemienie.")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--since", type=int, default=None, help="tylko prace od tego roku")
    ap.add_argument("--no-ground", action="store_true",
                    help="nie uzywaj tytulow z lokalnego indeksu")
    ap.add_argument("--debug", action="store_true", help="pokaz surowa odpowiedz modelu")
    args = ap.parse_args()

    titles = [] if args.no_ground else load_related_titles(args.question)
    if titles:
        print(f"(uziemienie: {len(titles)} tytulow z indeksu)\n")
    result = suggest_queries(args.question, titles=titles, model=args.model, debug=args.debug)

    if result["topic_en"]:
        print(f"Temat (EN): {result['topic_en']}")
    for c in result["concepts"]:
        print(f"* {c['name']}: {', '.join(c['terms'])}")
    if result["exclude"]:
        print(f"* wykluczenia: {', '.join(result['exclude'])}")
    print()
    for n, item in enumerate(result["queries"], 1):
        print(f"{n}. [{item['strategy']}] {item['query']}\n   {item['why']}\n"
              f"   {scholar_url(item['query'], args.since)}\n")
    print("Plan dalszego szukania:")
    for n, s in enumerate(PLAN_STEPS, 1):
        print(f"  {n}. {s}")


if __name__ == "__main__":
    main()
