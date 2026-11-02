#!/usr/bin/env python3
"""Krok 1: filtruje zrzut arXiv z Kaggle (JSONL) do kompaktowego papers.json.

Przyklad:
    python prepare_corpus.py --input arxiv-metadata-oai-snapshot.json \
        --categories cs.RO,eess.SY,cs.SY,cs.CV,eess.IV --min-year 2018 --per-category 15000
"""
import argparse
import json
import random
import re
from collections import defaultdict
from email.utils import parsedate_to_datetime


def get_year(rec):
    try:
        return parsedate_to_datetime(rec["versions"][0]["created"]).year
    except Exception:
        return None


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def format_authors(rec):
    parsed = rec.get("authors_parsed") or []
    names = [f"{a[1]} {a[0]}".strip() for a in parsed[:3] if len(a) >= 2]
    if not names:
        return clean(rec.get("authors"))[:120]
    return ", ".join(names) + (" et al." if len(parsed) > 3 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="arxiv-metadata-oai-snapshot.json")
    ap.add_argument("--output", default="papers.json")
    ap.add_argument("--categories", default="cs.RO,eess.SY,cs.SY,cs.CV,eess.IV",
                    help="kategorie arXiv oddzielone przecinkami")
    ap.add_argument("--min-year", type=int, default=2018)
    ap.add_argument("--per-category", type=int, default=15000,
                    help="limit prac na kategorie glowna (zeby cs.CV nie zdominowalo reszty)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
    groups = defaultdict(list)

    with open(args.input, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if n % 500000 == 0:
                print(f"  przetworzono {n} rekordow...")
            rec = json.loads(line)
            hits = [c for c in rec.get("categories", "").split() if c in wanted]
            if not hits:
                continue
            year = get_year(rec)
            if year is None or year < args.min_year:
                continue
            abstract = clean(rec.get("abstract"))
            if len(abstract) < 200 or "withdrawn" in abstract[:200].lower():
                continue
            groups[hits[0]].append({
                "id": rec["id"],
                "title": clean(rec.get("title")),
                "abstract": abstract,
                "year": year,
                "authors": format_authors(rec),
                "categories": rec.get("categories", ""),
                "url": f"https://arxiv.org/abs/{rec['id']}",
            })

    rng = random.Random(args.seed)
    papers, seen = [], set()
    for cat, items in sorted(groups.items()):
        rng.shuffle(items)
        taken = items[:args.per_category]
        print(f"{cat}: {len(items)} pasujacych, bierze {len(taken)}")
        for p in taken:
            if p["id"] not in seen:
                seen.add(p["id"])
                papers.append(p)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False)
    print(f"Zapisano {len(papers)} prac do {args.output}")


if __name__ == "__main__":
    main()
