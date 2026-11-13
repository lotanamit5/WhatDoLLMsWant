"""Collect pairwise preference scores for one (model, item set, frame, constraints) run.

Merged 2026-08-10 from the old `data_collection.py` and `data_collection_robustness.py`,
which had drifted apart (one had constraints, the other had all templates and all item sets).

Writes `<exp_dir>/config.json` and `<exp_dir>/scores.csv`. See
`docs/config_schema.md` for what goes in the config and why.

The prompt sent to the model is:

    <frame text> <constraints text>
    <template with {A} and {B} filled in>

and `prompt_prefix` in the config records that first line verbatim.
"""
import os
import sys
import json
import time
import argparse
import itertools
import contextlib
import subprocess

import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

# Add project root to sys.path to allow imports from src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'src'))
import src.alternatives
import src.prompts
# src.agent is imported inside collect_data: it pulls in torch, which --dry_run and the notebooks
# that borrow the wording functions from this file do not need.


ALTERNATIVES_ALIASES = {
    'colors': src.alternatives.colors,
    'foods': src.alternatives.foods,
    'cars': src.alternatives.cars,
    'stocks': src.alternatives.stocks,
    'laptops': src.alternatives.laptops,
    'laptop_brands': src.alternatives.laptop_brands,
    'laptops_num_vs_txt': src.alternatives.laptops_num_vs_txt,
    'laptops_txt_ram_screen': src.alternatives.laptops_txt_ram_screen,
    'laptops_num_ram_screen': src.alternatives.laptops_num_ram_screen,
    'laptops_robustness': src.alternatives.laptops_robustness,
    # 2026-10-12, feature-less domains added for the simple_domains batch
    'literature_genres': src.alternatives.literature_genres,
    'ai_companies': src.alternatives.ai_companies,
    'travel_destinations': src.alternatives.travel_destinations,
    'beverages': src.alternatives.beverages,
    # 2026-10-12: second feature domain. Dicts with title/author/year + 8 hidden genre tags.
    'books': src.alternatives.books,
    'books_small': src.alternatives.books_small,
}


def real_laptops():
    """60 real laptops (2026-11-12), from the design file written by scripts/real_laptops_prepare.py.
    `text` is the one line shown in the prompt. `id` (the dataset's laptop_ID) and the three contract
    features go to scores.csv; every other spec of an item stays in the design file."""
    df = pd.read_csv(os.path.join(ROOT, "data", "real_laptops", "design", "items.csv"))
    return df[['id', 'brand', 'screen', 'ram', 'text']].to_dict('records')


# A function, not a list: the file is read only when a job asks for these items.
ALTERNATIVES_ALIASES['real_laptops'] = real_laptops


def synthetic_cars():
    """120 synthetic cars (2026-11-13), from the design file written by scripts/synthetic_cars_prepare.py.
    `text` is the one line shown in the prompt (the six features, nothing else). `id` and the six
    features go to scores.csv; `part` / `track` (which cars are targets) stay in the design file."""
    df = pd.read_csv(os.path.join(ROOT, "data", "synthetic_cars", "design", "items.csv"))
    return df[['id', 'company', 'color', 'fuel', 'transmission', 'ownership', 'horsepower',
               'text']].to_dict('records')


ALTERNATIVES_ALIASES['synthetic_cars'] = synthetic_cars


def load_items(alias):
    items = ALTERNATIVES_ALIASES[alias]
    return items() if callable(items) else items

# Which prompt templates to use, and the answer strings whose last token gets scored.
# The two are bound together because they have to agree: a template ending
# "...I prefer Option " needs ["1", "2"] or ["A", "B"], and "Answer: " needs whatever the
# agent's default is. `labels: None` keeps the agent class default.
#
#   options       what every run up to 2026-08-16 used; asks a question, ends "Answer: ".
#   pretrained    for base models: ends mid-sentence so the model completes rather than
#                 answers. Digits as the answer.
#   pretrained_ab same, but the options are lettered and the scored tokens are A/B. Exists
#                 to test whether the answer-token prior is what slot-locks base models -
#                 with "1"/"2" that prior is inseparable from the positional bias, because
#                 option 1 is always in slot A. See docs/progress.md 2026-08-18.
TEMPLATE_SETS = {
    'options':       {'templates': src.prompts.options_comparisons,
                      'labels': None},
    'pretrained':    {'templates': src.prompts.pretrained_options_comparisons,
                      'labels': ['1', '2']},
    'pretrained_ab': {'templates': src.prompts.pretrained_options_comparisons_ab,
                      'labels': ['A', 'B']},
}

# The sentence that sets the scene, before any spec is named. "shopping" is what every
# run up to 2026-08-10 used. The others exist to measure how much the frame itself moves
# the answer - we have never run without one.
FRAMES = {
    'bare': "",
    'shopping': "I am looking to buy a laptop.",
    'self': "You are choosing a laptop for yourself.",
    'third_person': "A person is looking to buy a laptop.",
    # 2026-10-12: the books domain's counterpart of `shopping`. Same voice, same shape.
    'reading': "I am looking for a book to read.",
    # 2026-11-13: the cars domain's counterpart of `shopping`. Same sentence, "car" for "laptop".
    'car_shopping': "I am looking to buy a car.",
    # Personas (2026-10-08): a sentence about who the user is, then the `shopping` sentence
    # unchanged - so against `shopping` the persona is the only difference. Neither names a
    # spec: the model has to infer one (student -> cheap -> 4GB? editor -> 16GB?).
    'student': "I am a student and I do not have much money. I am looking to buy a laptop.",
    'editor': "I am a professional video editor. I am looking to buy a laptop.",
    # 2026-10-08, second persona batch. The two halves of `student`, to see which one carries its
    # effect (identity or money), and two brand stereotypes with no price in them: one should
    # push away from Apple (`windows`), one toward it (`designer`).
    'student_only': "I am a student. I am looking to buy a laptop.",
    'no_money': "I do not have much money. I am looking to buy a laptop.",
    'windows': "My company only supports Windows. I am looking to buy a laptop.",
    'designer': "I am a graphic designer. I am looking to buy a laptop.",
    # 2026-10-08, ROLE voice: the same content, but the model is told to BE the person instead
    # of the user describing themselves. The template still ends "Which do you prefer?", so the
    # model answers as that person. `role_none` is the role-voice baseline, so a role-vs-user
    # difference can be split into "the Act-as wording itself" and "the persona inside it":
    #     user voice:  shopping   vs  student        (both exist)
    #     role voice:  role_none  vs  role_student
    'role_none': "Act as a person who is looking to buy a laptop.",
    'role_student': "Act as a student who does not have much money and is looking to buy a laptop.",
}

# Who the frame sentence speaks as - recorded in config.json as frame.voice, so runs can be
# filtered by phrasing without parsing the text. The frame NAME stays the unique key.
#   user          the user describes themselves ("I am a student ...")
#   role          the model is told to act as someone ("Act as a student ...")
#   model         the model chooses for itself ("You are choosing a laptop for yourself.")
#   third_person  someone else is buying
#   none          no frame sentence at all
FRAME_VOICE = {'bare': 'none', 'self': 'model', 'third_person': 'third_person',
               'role_none': 'role', 'role_student': 'role'}           # anything else: 'user'


def frame_voice(frame):
    return FRAME_VOICE.get(frame, 'user')

# level string (exactly as it appears in alternatives.py / scores.csv) -> prose
CONSTRAINT_PHRASE = {
    'screen': lambda v: f"a {v} screen",      # "14-inch" -> "a 14-inch screen"
    'ram': lambda v: f"{v[:-2]} GB ram",      # "8GB"     -> "8 GB ram"
    'brand': lambda v: f"a {v} laptop",
}

# Books (2026-10-12): each tag is a yes/no feature; a contract names the tag, level "yes" only
# ("I prefer fantasy." / "I prefer fantasy and classics."). The noun is what a reader would say.
BOOK_TAG_NOUN = {
    'fantasy': "fantasy",
    'science-fiction': "science fiction",
    'romance': "romance",
    'mystery-thriller': "mysteries and thrillers",
    'young-adult': "young adult books",
    'classics': "classics",
    'historical-fiction': "historical fiction",
    'nonfiction': "nonfiction",
}


def _book_tag_phrase(tag):
    def phrase(v):
        assert v == "yes", f"books: only {tag}=yes has a sentence, got {tag}={v!r}"
        return BOOK_TAG_NOUN[tag]
    return phrase


CONSTRAINT_PHRASE.update({tag: _book_tag_phrase(tag) for tag in BOOK_TAG_NOUN})

# Word order of the generated sentence. Fixed here rather than taken from the dict's
# insertion order, because json.dump(sort_keys=True) reorders the dict on the way to disk -
# so insertion order does not survive a round trip and must not be load-bearing.
# This order reproduces the historical strings ("a 14-inch screen and 8 GB ram").
FEATURE_ORDER = ['brand', 'screen', 'ram'] + list(BOOK_TAG_NOUN)


# Synthetic cars (2026-11-13). Same shape as the laptop phrases: one noun phrase per requirement,
# joined by "and": "I prefer a BMW car and an automatic transmission and 200 hp."
def _a(phrase):
    return ("an " if phrase[0].lower() in "aeiou" else "a ") + phrase


CONSTRAINT_PHRASE.update({
    'company': lambda v: _a(f"{v} car"),                    # "a BMW car", "an Audi car"
    'color': lambda v: _a(f"{v} color"),                    # "a black color"
    'fuel': lambda v: _a(f"{v} engine"),                    # "a diesel engine"
    'transmission': lambda v: _a(f"{v} transmission"),      # "an automatic transmission"
    'horsepower': lambda v: v,                              # "200 hp"
    # "a private first owner"; the two non-private levels need the noun: "an ex-lease car"
    'ownership': lambda v: _a(v if v.startswith("private") else f"{v} car"),
})
FEATURE_ORDER += ['company', 'color', 'fuel', 'transmission', 'horsepower', 'ownership']


def parse_constraints(spec):
    """'screen=14-inch,ram=8GB' -> {'screen': '14-inch', 'ram': '8GB'}.

    An empty/missing spec means no constraint at all, which is `{}` - never None.
    The order you write them in does not matter; FEATURE_ORDER decides the sentence.
    """
    if not spec:
        return {}
    out = {}
    for part in spec.split(','):
        feature, _, level = part.partition('=')
        if not level:
            raise ValueError(f"constraint {part!r} must look like feature=level, "
                             f"e.g. screen=14-inch")
        out[feature.strip()] = level.strip()
    return out


def parse_contracts(spec):
    """Several contracts for ONE job, separated by ';' - one run each, one model load:
    'none;ram=8GB;brand=Dell,ram=8GB' -> [{}, {'ram': '8GB'}, {'brand': 'Dell', 'ram': '8GB'}].
    'none' (or an empty part) is the no-contract run. A plain single spec gives a list of one."""
    return [parse_constraints("" if part.strip() == "none" else part.strip()) for part in spec.split(";")]


def constraints_text(constraints):
    if not constraints:
        return ""
    ordered = sorted(constraints, key=FEATURE_ORDER.index)
    parts = [CONSTRAINT_PHRASE[f](constraints[f]) for f in ordered]
    return "I prefer " + " and ".join(parts) + "."


def constraints_id(constraints):
    """Unambiguous slug for filtering and folder names. Derived, never typed by hand -
    which is the whole point: `screen=16-inch` and `ram=16GB` used to both be "16"."""
    return "+".join(f"{f}={v}" for f, v in sorted(constraints.items())) or "none"


def validate_constraints(constraints, items):
    """A constraint naming a level no item has would silently produce a run where nothing
    satisfies the contract. That has happened; it is worth the six lines."""
    for feature, level in constraints.items():
        if feature not in CONSTRAINT_PHRASE:
            raise ValueError(f"no prose template for feature {feature!r}; "
                             f"add one to CONSTRAINT_PHRASE")
        levels = {item[feature] for item in items if feature in item}
        if not levels:
            raise ValueError(f"items have no feature {feature!r}")
        if level not in levels:
            raise ValueError(f"{feature}={level!r} is not a level of {feature}; "
                             f"choose one of {sorted(levels)}")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def format_features_to_text(item_dict):
    return (f"{item_dict.get('screen', '')} {item_dict.get('brand', '')} "
            f"laptop with {item_dict.get('ram', '')} RAM").strip()


def format_book_to_text(item_dict):
    """Title, author, year - and nothing else. The tags stay out of the prompt on purpose:
    the choice should come from what the model knows about the book (Lotan, 2026-10-12)."""
    return f"{item_dict['title']} by {item_dict['author']} ({item_dict['year']})"


def item_text(item):
    """What goes into the prompt: a plain string item (colors, foods, ...) as is, a book dict
    through format_book_to_text, a laptop dict through format_features_to_text."""
    if isinstance(item, str):
        return item
    if 'text' in item:                     # the item brings its own line (real_laptops)
        return item['text']
    return format_book_to_text(item) if 'title' in item else format_features_to_text(item)


def item_columns(prefix, item):
    """scores.csv columns for one option: a string item is one feature called `item`
    (a_item / b_item); a dict gives one column per feature (a_brand, a_screen, ...)."""
    if isinstance(item, str):
        return {f"{prefix}item": item}
    return {f"{prefix}{k}": v for k, v in item.items() if k != 'text'}      # `text` is prompt-only


def ordered_pairs(items, pairs=''):
    """What gets scored. Default: every ordered pair of items. With `pairs` (a CSV of two item-id
    columns, path from the repo root): only those pairs, each in both orders. Needs items with `id`."""
    if not pairs:
        return list(itertools.permutations(items, 2))
    by_id = {item['id']: item for item in items}
    listed = pd.read_csv(os.path.join(ROOT, pairs)).itertuples(index=False, name=None)
    return [(by_id[a], by_id[b]) for x, y in listed for a, b in ((x, y), (y, x))]


_DEVNULL = open(os.devnull, "w")


def query(agent, prompt, show=False):
    """agent.query prints the whole prompt, its tokens and the scores on EVERY call: 114 MB of log
    for one all-templates run, and the lines that matter are lost in it. Show the first prompt of a
    run (to check the wording), mute the rest. The scores are untouched."""
    if show:
        return agent.query(prompt)
    with contextlib.redirect_stdout(_DEVNULL):
        return agent.query(prompt)


class Progress:
    """One log line a minute while scoring: rows done, speed, time left. Call tick() once per row."""

    def __init__(self, total, every=60):
        self.total, self.every, self.done = total, every, 0
        self.t0 = self.last = time.time()

    def tick(self):
        self.done += 1
        now = time.time()
        if now - self.last >= self.every or self.done == self.total:
            rate = self.done / max(now - self.t0, 1e-9)
            print(f"[{datetime.now():%H:%M:%S}] {self.done}/{self.total} rows "
                  f"({100 * self.done // self.total}%) | {rate:.1f} rows/s | "
                  f"{(self.total - self.done) / rate / 60:.0f} min left", flush=True)
            self.last = now


def collect_data(model_family, model_size, alternatives, exp_dir,
                 constraints=None, frame='shopping', n_templates=5,
                 template_set='options', pairs='', dry_run=False):
    """One job = ONE model load = one run per (item set, contract).

    `alternatives` is one alias, or several separated by commas (2026-10-12, simple_domains).
    `constraints` is one contract dict, or a list of them (2026-11-13, real laptops).
    A single run is written to `exp_dir`. Several get the part that varies appended:
    `<exp_dir>_<alias>`, `<exp_dir>_<constraints_id>`, or `<exp_dir>_<alias>_<constraints_id>`.
    `dry_run` prints the first prompt of every run and stops: no model, no files."""
    aliases = alternatives.split(",")
    contracts = constraints if isinstance(constraints, list) else [constraints or {}]
    cids = [constraints_id(c) for c in contracts]
    assert len(set(cids)) == len(cids), f"the same contract twice: {cids}"
    for alias in aliases:                                   # fail before loading a model, not after
        assert alias in ALTERNATIVES_ALIASES, f"unknown alternatives alias {alias!r}"
        for con in contracts:
            validate_constraints(con, load_items(alias))
    if dry_run and not exp_dir:
        exp_dir = "data/<exp_name>/<job_id>"

    n_tmpl = len(TEMPLATE_SETS[template_set]['templates'][:n_templates or None])
    progress = Progress(n_tmpl * len(contracts) *                       # the whole job, all its runs
                        sum(len(ordered_pairs(load_items(a), pairs)) for a in aliases))
    print(f"{model_family}-{model_size} | {len(aliases) * len(contracts)} runs | {progress.total} prompts",
          flush=True)

    agent = None
    if not dry_run:
        from src.agent import agent_factory
        agent = agent_factory(model_family, model_size, TEMPLATE_SETS[template_set]['labels'])
    for alias in aliases:
        for i, (con, cid) in enumerate(zip(contracts, cids)):
            parts = ([alias] if len(aliases) > 1 else []) + ([cid] if len(contracts) > 1 else [])
            collect_one(agent, model_family, model_size, alias, "_".join([exp_dir] + parts), con, frame,
                        n_templates, template_set, pairs, dry_run, progress, show_items=(i == 0))


def collect_one(agent, model_family, model_size, alternatives_alias, exp_dir,
                constraints=None, frame='shopping', n_templates=5, template_set='options',
                pairs='', dry_run=False, progress=None, show_items=True):
    items = load_items(alternatives_alias)
    constraints = constraints or {}
    validate_constraints(constraints, items)
    ordered = ordered_pairs(items, pairs)

    templates = TEMPLATE_SETS[template_set]['templates']
    labels = TEMPLATE_SETS[template_set]['labels']
    if n_templates:
        templates = templates[:n_templates]

    frame_text = FRAMES[frame]
    con_text = constraints_text(constraints)
    prompt_prefix = " ".join(p for p in (frame_text, con_text) if p)

    if dry_run:
        first = f"{prompt_prefix}\n{templates[0]}" if prompt_prefix else templates[0]
        print(f"\n===== {alternatives_alias} | {constraints_id(constraints)} | "
              f"{len(templates) * len(ordered)} prompts -> {exp_dir}")
        print(first.format(A=item_text(ordered[0][0]), B=item_text(ordered[0][1])))
        return

    print('=== alternatives:', alternatives_alias, '| constraints:', constraints_id(constraints),
          '|', len(templates) * len(ordered), 'prompts ->', exp_dir)
    print('model_id:', agent.tokenizer.name_or_path)
    print('template_set:', template_set, '| labels:', labels or agent.labels)
    print('prompt_prefix:', repr(prompt_prefix), '| voice:', frame_voice(frame))
    print('constraints:', constraints)
    if show_items:                         # once per item set, not once per contract
        print('items:')
        for item in items:
            print('-', item)
    print('templates:')
    for template in templates:
        print('-', template)

    config = {
        "model_family": model_family,
        "model_size": model_size,
        "alternatives": alternatives_alias,

        "frame": {"name": frame, "text": frame_text, "voice": frame_voice(frame)},
        "constraints": constraints,
        "constraints_text": con_text,
        "constraints_id": constraints_id(constraints),
        "prompt_prefix": prompt_prefix,
        # "all" = every ordered pair of the items; otherwise the pairs file (each pair in both orders)
        "pairs": pairs or "all",
        "n_rows": len(templates) * len(ordered),     # what a COMPLETE scores.csv holds

        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": git_commit(),
        "collection_script": "scripts/data_collection.py",
        "template_set": template_set,
        "labels": labels if labels is not None else agent.labels,
        "templates": {i: t for i, t in enumerate(templates)},
    }
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    records = []
    progress = progress or Progress(len(templates) * len(ordered))
    for idx, template in tqdm(enumerate(templates)):
        full_template = f"{prompt_prefix}\n{template}" if prompt_prefix else template
        for option_a, option_b in ordered:
            prompt = full_template.format(A=item_text(option_a), B=item_text(option_b))
            score_a, score_b = query(agent, prompt, show=not records)
            progress.tick()
            record = {'template_idx': idx, 'score_a': score_a, 'score_b': score_b}
            record.update(item_columns("a_", option_a))
            record.update(item_columns("b_", option_b))
            records.append(record)

    pd.DataFrame(records).to_csv(os.path.join(exp_dir, "scores.csv"), index=False)
    print(f"Finished! {len(records)} rows -> {exp_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pairwise preference scores")
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--alternatives", type=str, required=True,
                        help="one alias, or several separated by commas (one run each, one model load)")
    parser.add_argument("--exp_dir", type=str, required=False)
    parser.add_argument("--constraints", type=str, default="",
                        help='"screen=14-inch,ram=8GB". Empty means no constraint. Several contracts '
                             'in one job: separate them with ";", and write "none" for no constraint.')
    parser.add_argument("--pairs", type=str, default="",
                        help="CSV of item-id pairs (path from the repo root), each scored in both "
                             "orders. Empty means every ordered pair.")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the first prompt of every run and stop. No model, no files.")
    parser.add_argument("--frame", type=str, default="shopping", choices=sorted(FRAMES),
                        help="the scene-setting sentence before any spec")
    parser.add_argument("--n_templates", type=int, default=5,
                        help="use the first N prompt templates; 0 means all")
    parser.add_argument("--template_set", type=str, default="options",
                        choices=sorted(TEMPLATE_SETS),
                        help="'options' for instruct models, 'pretrained' for base models")

    args = parser.parse_args()
    collect_data(
        args.model_family,
        args.model_size,
        args.alternatives,
        args.exp_dir,
        constraints=parse_contracts(args.constraints),
        frame=args.frame,
        n_templates=args.n_templates,
        template_set=args.template_set,
        pairs=args.pairs,
        dry_run=args.dry_run,
    )
