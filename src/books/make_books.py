"""Build src/books/books.csv: 48 famous Goodreads books with 8 hidden genre tags.

Reproduces the list used by `alternatives.books`. Run from the repo root:

    python src/books/make_books.py

Source: goodbooks-10k (https://github.com/zygmuntz/goodbooks-10k), the 10,000 most-rated books on
Goodreads with per-book shelf counts. Downloaded on demand to external/goodbooks-10k/ (gitignored).
Design and checks: Notebooks/books_domain_design.ipynb, docs/progress.md 2026-10-12.

Decisions (Lotan, 2026-10-12): real books, as famous as possible; genre tags only; the prompt shows
title + author + year and the tags stay hidden (features for the analysis and for contracts).
"""
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(ROOT, "external", "goodbooks-10k")
OUT = os.path.join(HERE, "books.csv")
URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"

# Goodreads shelves are user-made, so one genre has many spellings. These are merged into 8 genres.
# `children` and `horror` were dropped: only 3 famous books each.
GENRES = {
    "fantasy":            ["fantasy", "urban-fantasy", "magic", "high-fantasy", "epic-fantasy"],
    "science-fiction":    ["science-fiction", "sci-fi", "scifi", "dystopian", "dystopia"],
    "romance":            ["romance", "paranormal-romance", "contemporary-romance", "chick-lit", "love"],
    "mystery-thriller":   ["mystery", "thriller", "crime", "suspense", "mystery-thriller", "detective"],
    "young-adult":        ["young-adult", "ya", "teen", "ya-fiction", "young-adult-fiction"],
    "classics":           ["classics", "classic", "literature", "classic-literature"],
    "historical-fiction": ["historical-fiction", "historical", "historical-fiction-romance"],
    "nonfiction":         ["non-fiction", "nonfiction", "memoir", "biography", "history", "science",
                           "philosophy", "self-help", "psychology", "business"],
}
GENRE_LIST = list(GENRES)

THR = 0.12          # a book has a genre if it is >= 12% of the book's genre shelvings
N, TOP, MIN, CAP = 48, 32, 8, 18


def load_raw():
    os.makedirs(RAW, exist_ok=True)
    for name in ["books.csv", "book_tags.csv", "tags.csv"]:
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(URL + name, path)
    books = pd.read_csv(os.path.join(RAW, "books.csv"))
    bt = pd.read_csv(os.path.join(RAW, "book_tags.csv")).merge(
        pd.read_csv(os.path.join(RAW, "tags.csv")), on="tag_id")
    return books, bt


def genre_shares(bt):
    """Per book, the share of its genre shelvings that goes to each of the 8 genres (rows sum to 1)."""
    syn2g = {s: g for g, ss in GENRES.items() for s in ss}
    bt = bt.assign(genre=bt.tag_name.map(syn2g)).dropna(subset=["genre"])
    g = bt.groupby(["goodreads_book_id", "genre"])["count"].sum().unstack(fill_value=0)
    return g.div(g.sum(axis=1), axis=0)


def candidates(books, share, thr=THR):
    """English books with a year and at least one genre, most-rated first, with clean title/author."""
    b = books.set_index("goodreads_book_id")
    b["title_clean"] = b.title.str.replace(r"\s*\(.*\)\s*$", "", regex=True).str.strip()  # drop "(Series, #1)"
    b["author"] = b.authors.str.split(",").str[0].str.strip()                               # first author only
    b["year"] = b.original_publication_year
    b["series"] = b.title.str.extract(r"\(([^,#]+)[,#]")[0].str.strip()
    b["series_no"] = b.title.str.extract(r"#(\d+)")[0].astype(float)
    b = b.join((share >= thr).astype(int), how="inner")
    b = b[b.language_code.fillna("eng").str.startswith("en") & b.year.notna()]
    b["n_tags"] = b[GENRE_LIST].sum(axis=1)
    return b[b.n_tags > 0].sort_values("ratings_count", ascending=False)


def pick(b, n=N, top=TOP, min_per_tag=MIN, cap=CAP):
    """Fame first, then balance. 1: the `top` most-rated books. 2: every tag up to `min_per_tag`
    books. 3: fill to `n` with a cap per tag. Always one book per series (its first), max 2 per author."""
    picked, cnt, series_seen, author_cnt = [], {t: 0 for t in GENRE_LIST}, set(), {}

    def ok(gid, r, need=None, cap=None):
        tl = [t for t in GENRE_LIST if r[t]]
        if gid in picked: return False
        if need is not None and need not in tl: return False
        if cap is not None and any(cnt[t] >= cap for t in tl): return False
        if pd.notna(r.series) and (r.series in series_seen or r.series_no > 1): return False
        if author_cnt.get(r.author, 0) >= 2: return False
        return True

    def take(gid, r):
        picked.append(gid); series_seen.add(r.series)
        author_cnt[r.author] = author_cnt.get(r.author, 0) + 1
        for t in GENRE_LIST:
            if r[t]: cnt[t] += 1

    for gid, r in b.iterrows():                       # pass 1
        if len(picked) >= top: break
        if ok(gid, r): take(gid, r)
    for t in GENRE_LIST:                              # pass 2
        for gid, r in b.iterrows():
            if cnt[t] >= min_per_tag: break
            if ok(gid, r, need=t): take(gid, r)
    for gid, r in b.iterrows():                       # pass 3
        if len(picked) >= n: break
        if ok(gid, r, cap=cap): take(gid, r)
    while len(picked) > n:                            # pass 2 can overshoot: drop the last-added book no tag needs
        for gid in reversed(picked):
            tl = [t for t in GENRE_LIST if b.loc[gid, t]]
            if all(cnt[t] > min_per_tag for t in tl):
                picked.remove(gid)
                for t in tl: cnt[t] -= 1
                break

    sel = b.loc[picked].sort_values("ratings_count", ascending=False).copy()
    sel["year"] = sel.year.astype(int)
    sel["tags"] = sel[GENRE_LIST].apply(lambda r: ", ".join(t for t in GENRE_LIST if r[t]), axis=1)
    return sel, cnt


def build():
    books, bt = load_raw()
    share = genre_shares(bt)
    b = candidates(books, share)
    sel, cnt = pick(b)
    sel["pop_rank_all"] = [int((books.ratings_count > rc).sum()) + 1 for rc in sel.ratings_count]
    return books, share, b, sel, cnt


def main():
    books, share, b, sel, cnt = build()
    out = sel[["title_clean", "author", "year", "average_rating", "ratings_count", "pop_rank_all"] + GENRE_LIST]
    out = out.rename(columns={"title_clean": "title"})
    out.index.name = "goodreads_book_id"
    out.to_csv(OUT)
    print(f"wrote {OUT}: {len(out)} books; per tag: {cnt}; least famous is #{out.pop_rank_all.max()} by ratings count")


if __name__ == "__main__":
    main()
