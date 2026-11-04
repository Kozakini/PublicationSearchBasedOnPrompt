#!/usr/bin/env python3
"""Generator zapytan do Google Scholar (maly LLM przez Ollame, CPU) - wersja 3.

Podzial pracy:
  * LLM robi to, w czym jest dobry: rozbija temat (albo twierdzenie do weryfikacji)
    na pojecia i dobiera neutralne slownictwo naukowe (angielskie terminy, synonimy).
  * Skladnie zapytan do Google Scholar buduje zwykly kod wedlug dobrych praktyk
    wyszukiwania. Dzieki temu maly model nie psuje skladni, a zapytania sa przewidywalne.

Tryby:
  * temat       - rdzen, synonimy, alternatywna terminologia, intitle, przeglady
  * twierdzenie - (pytanie tak/nie albo cos, co ktos twierdzi w sieci) zapytania neutralne,
                  szukajace dowodow za i przeciw: meta-analizy, mieszane dowody, metodologia

Uzycie z terminala:
    uv run suggest.py "czy praca zdalna zmniejsza produktywnosc"
    uv run suggest.py "sterowanie robotem jezykiem naturalnym" --since 2022 --ground
    uv run suggest.py "temat" --debug          # pokaz surowa odpowiedz modelu

Uzycie z kodu (np. w Gradio):
    from suggest import suggest_queries, to_markdown
    result = suggest_queries(pytanie)                       # ogolnie
    result = suggest_queries(pytanie, titles=tytuly)        # z uziemieniem w korpusie
    markdown = to_markdown(result, since=2020)

Uziemienie w tytulach z lokalnego korpusu (--ground) ma sens TYLKO dla tematow z dziedziny
korpusu (robotyka, automatyka, wizja). Dla innych tematow podsuwa modelowi nietrafne slownictwo.

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

SYSTEM_PROMPT = """You are a research librarian. You turn a student's topic, question or claim (in any language) into the building blocks of a good literature search. You do NOT write search queries: you output concepts and vocabulary, and software assembles the queries.

Output ONLY a JSON object:
{"kind": "topic" or "claim", "topic_en": "...", "concepts": [{"name": "...", "terms": ["preferred term", "synonym", "related term"]}], "exclude": ["..."]}

Best practices you must follow:
1. Translate to English. Replace everyday words with the academic terminology that authors actually use in titles and abstracts.
2. Set "kind" to "claim" when the input is a yes/no question or a statement someone wants to verify (something seen online, a popular belief); otherwise "topic".
3. For a claim, restate it as a neutral research question in "topic_en". Use neutral, non-loaded terminology that would find evidence both for and against the claim. Never presuppose the answer and never use wording that already assumes the claim is true or false.
4. Split the input into 2-3 distinct, non-overlapping concepts. For a claim: (a) the population or phenomenon, (b) the outcome or measure.
5. Only include concepts that are present in the input. Do NOT add methods or techniques (machine learning, neural networks, statistics, etc.) unless the input itself mentions them.
6. For each concept give 2-4 terms. The FIRST term is the most common standard term in the literature (a 1-3 word noun phrase). The others are real synonyms, alternative names or acronyms.
7. Use noun phrases, never sentences or questions. Prefer established terminology over invented phrases.
8. "exclude": 0-2 single words that typically bring irrelevant results for this topic (another meaning of an ambiguous term). Use [] if unsure.
9. If RELATED TITLES are provided, reuse their terminology only when it clearly matches the input; ignore them otherwise.
10. Never invent paper titles, authors or numbers."""

EXAMPLES = [
    ("jak gry uzależniają graczy", {
        "kind": "topic",
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
        "kind": "topic",
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
        "kind": "topic",
        "topic_en": "bearing fault diagnosis in electric motors using neural networks",
        "concepts": [
            {"name": "bearing fault diagnosis",
             "terms": ["bearing fault diagnosis", "condition monitoring", "fault detection"]},
            {"name": "neural networks",
             "terms": ["deep learning", "convolutional neural network"]},
            {"name": "electric motors",
             "terms": ["induction motor", "electric machine"]},
        ],
        "exclude": ["biological"],
    }),
    ("czy praca zdalna zmniejsza produktywność", {
        "kind": "claim",
        "topic_en": "does remote work affect employee productivity",
        "concepts": [
            {"name": "remote work",
             "terms": ["remote work", "telework", "working from home"]},
            {"name": "productivity",
             "terms": ["employee productivity", "job performance", "labor productivity"]},
        ],
        "exclude": [],
    }),
    ("czy kawa skraca życie", {
        "kind": "claim",
        "topic_en": "is coffee consumption associated with mortality",
        "concepts": [
            {"name": "coffee consumption",
             "terms": ["coffee consumption", "coffee intake", "caffeine"]},
            {"name": "mortality",
             "terms": ["all-cause mortality", "longevity", "life expectancy"]},
        ],
        "exclude": [],
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
    "Meta-analizy": "Najpierw podsumowania wielu badań - pojedyncze prace potrafią sobie przeczyć.",
    "Mieszane dowody": "Szuka źródeł, które omawiają sprzeczne lub niejednoznaczne wyniki.",
    "Metodologia i błędy": "Prace o ograniczeniach pomiaru i zakłócających czynnikach - do oceny wiarygodności.",
}

PLAN_STEPS_TOPIC = [
    "Zacznij od zapytania „Przeglądy”: wybierz 1-2 świeże prace typu survey/review - dają mapę tematu.",
    "Zbierz słownictwo: z tytułów, abstraktów i słów kluczowych trafnych prac wypisz nowe terminy i wróć do zapytań z innymi synonimami.",
    "Snowballing: w Google Scholar użyj „Cytowane przez” (do przodu), bibliografii pracy (wstecz) i „Podobne artykuły”.",
    "Zawęź lata i sprawdź, czy źródła są recenzowane, a nie tylko preprintami.",
    "Notuj użyte zapytania i daty - ułatwia to powtórzenie wyszukiwania i opisanie metody w pracy.",
]

PLAN_STEPS_CLAIM = [
    "Doprecyzuj twierdzenie: kto lub co jest porównywane, jaką miarą, w jakim kraju i okresie. Ogólne zdanie z sieci zwykle ukrywa te szczegóły.",
    "Zacznij od meta-analiz i przeglądów systematycznych - pojedyncze badania często sobie przeczą, a podsumowania ważą dowody.",
    "Szukaj dowodów za i przeciw. Nie zatrzymuj się na pierwszej pracy, która potwierdza Twoje przypuszczenie.",
    "Oceń metodologię: grupa porównawcza, sposób pomiaru, kontrola czynników zakłócających, źródło danych, wielkość próby.",
    "Odróżnij korelację od związku przyczynowego: czy projekt badania w ogóle pozwala na wniosek przyczynowy?",
    "Dojdź do źródła twierdzenia: dane urzędowe lub oryginalne badanie, a nie streszczenie w mediach. Sprawdź daty i kontekst.",
    "Wynik „dowody są mieszane” albo „nie wiadomo” też jest poprawnym wnioskiem. Zapisz użyte zapytania i daty.",
]

CLAIM_START = re.compile(
    r"^\s*(czy|is|are|do|does|did|can|will|should|has|have)\b|prawd[aęy] (jest )?(że|ze)\b",
    re.IGNORECASE)


def looks_like_claim(question):
    return bool(CLAIM_START.search(question))


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
    return re.sub(r"\s+", " ", term).strip(" '+-")


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


def normalize(data, question=""):
    """Sprowadza odpowiedz modelu do czystej struktury; odporne na typowe bledy malych modeli."""
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
    model_kind = str(data.get("kind") or "").strip().lower()
    kind = "claim" if (model_kind == "claim" or looks_like_claim(question)) else "topic"
    return {
        "kind": kind,
        "topic_en": str(data.get("topic_en") or "").strip(),
        "concepts": concepts[:3],
        "exclude": exclude[:2],
    }


def build_queries(concepts, exclude=None, kind="topic"):
    """Deterministycznie sklada zapytania do Google Scholar z pojec i synonimow."""
    cs = [c for c in concepts if c["terms"]]
    if not cs:
        return []
    c1 = cs[0]["terms"]
    c2 = cs[1]["terms"] if len(cs) > 1 else []
    c3 = cs[2]["terms"] if len(cs) > 2 else []

    core = " ".join([quote(c1[0])] + ([quote(c2[0])] if c2 else []))
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

    if kind == "claim":
        add("Meta-analizy", f'{core} "meta-analysis" OR "systematic review"')
        add("Mieszane dowody", f'{core} "mixed evidence" OR "inconsistent findings"')
        add("Metodologia i błędy", f'{core} confounding OR "selection bias"')
    else:
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
    result = normalize(parse_json(raw), question)
    if not result["concepts"]:
        raise ValueError("Model nie zwrocil uzytecznych pojec - sprobuj inaczej sformulowac temat "
                         "albo uzyc wiekszego modelu.")
    result["queries"] = build_queries(result["concepts"], result["exclude"], result["kind"])
    return result


def scholar_url(query, since=None):
    params = {"q": query, "hl": "en"}
    if since:
        params["as_ylo"] = str(since)
    return "https://scholar.google.com/scholar?" + urllib.parse.urlencode(params)


def plan_steps(result):
    return PLAN_STEPS_CLAIM if result.get("kind") == "claim" else PLAN_STEPS_TOPIC


def to_markdown(result, since=None):
    lines = []
    if result.get("kind") == "claim":
        lines.append("**Tryb: weryfikacja twierdzenia** (zapytania neutralne - szukają dowodów za i przeciw)\n")
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
    lines += [f"{n}. {s}" for n, s in enumerate(plan_steps(result), 1)]
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
    ap.add_argument("--ground", action="store_true",
                    help="uziemij slownictwo w tytulach z lokalnego korpusu "
                         "(tylko dla tematow z robotyki/automatyki/wizji)")
    ap.add_argument("--debug", action="store_true", help="pokaz surowa odpowiedz modelu")
    args = ap.parse_args()

    titles = load_related_titles(args.question) if args.ground else []
    if titles:
        print(f"(uziemienie: {len(titles)} tytulow z indeksu)\n")
    result = suggest_queries(args.question, titles=titles, model=args.model, debug=args.debug)

    print(f"Tryb: {'weryfikacja twierdzenia' if result['kind'] == 'claim' else 'temat'}")
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
    for n, s in enumerate(plan_steps(result), 1):
        print(f"  {n}. {s}")


if __name__ == "__main__":
    main()
