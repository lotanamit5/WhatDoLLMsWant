import os
import sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

exp_name = "laptops_robustness_pt"  # CHANGE WHEN RUNNING (a set may override with 'n')
nodes = [
    'plato2',
    'plotinus1',
    'plotinus2',
]
# Model sizes pinned to node pools (round-robin inside a pool). qwen-72B needs a plotinus node:
# the 72B editor run of the first persona batch was sent to plato2 and never landed.
# Sizes not listed here use `nodes` above.
NODES_BY_SIZE = {
    '32': ['plato1', 'plato2'],
    '72': ['plotinus1', 'plotinus2'],
}

QWEN = ['0.5', '7', '32', '72']
GEMMA = ['1', '4', '12', '27']
OLMO = ['1', '7', '13', '32']

# Constraints are "feature=level" using the exact level strings from alternatives.py
# ('13-inch'/'14-inch'/'16-inch', '4GB'/'8GB'/'16GB', and the brand names).
# "" means no constraint.
#
# Already collected on laptops_robustness (all 8 models):
#     '', 'screen=14-inch', 'ram=8GB', 'screen=14-inch,ram=8GB'
# Every one of those asks for a MID or LOW level - that is the hole this batch fills.
PHASE_A = [
    'screen=13-inch',
    'screen=16-inch',
    'ram=4GB',
    'ram=16GB',
]

# The four contract conditions every instruct model already has on laptops_robustness.
# Repeating exactly these on the base models is what makes the comparison a clean one.
BASELINE_FOUR = [
    '',
    'screen=14-inch',
    'ram=8GB',
    'screen=14-inch,ram=8GB',
]

# One parameter dict per batch. Kept as a list because model family and size do not
# cross (qwen has no 1B, gemma has no 0.5B), and because the bare-frame batch varies
# a different flag. Everything is a plain product inside each dict.
parameter_sets = [
    # --- 2026-10-08, second persona batch: qwen 32B and 72B, no contract. 8 jobs.
    # The first batch showed `student` ("...and I do not have much money.") moves brand and screen
    # far more than RAM, and drops Apple by up to 22 log-odds (progress.md 2026-10-08). This asks
    # (1) which half of that sentence does it: `student_only` vs `no_money`, and (2) whether brand
    # moves from a stereotype with no price in it, in both directions: `windows` (away from
    # Apple), `designer` (toward Apple). Same folder as the first batch; frame names are new, so
    # the run key (family, size, frame, constraints_id) cannot collide.
    {'m': ['qwen'], 's': ['32', '72'], 'a': ['laptops_robustness'], 'c': [''],
     'f': ['student_only', 'no_money', 'windows', 'designer'], 'n': ['laptops_persona']},

    # --- 2026-10-08: persona VOICE - "I am a poor student" vs "Act as a poor student". 4 jobs.
    # Role voice for the baseline and for the student, so the comparison is a 2x2:
    #     user voice: shopping (exists) vs student (exists)
    #     role voice: role_none         vs role_student      <- these 4 jobs
    # Does telling the model to BE the person move it as much as the user describing
    # themselves? config.json records frame.voice = "role".
    {'m': ['qwen'], 's': ['32', '72'], 'a': ['laptops_robustness'], 'c': [''],
     'f': ['role_none', 'role_student'], 'n': ['laptops_persona']},

    # --- DONE 2026-10-08: first persona batch. Landed: qwen-32B student (1442267), qwen-32B
    # editor (1442268), qwen-72B student (1442269). NOT landed: qwen-72B editor - it was sent to
    # plato2. To re-run it, add {'m': ['qwen'], 's': ['72'], 'a': ['laptops_robustness'],
    # 'c': [''], 'f': ['editor'], 'n': ['laptops_persona']} - but first check on the cluster that
    # the old job is not still running, or two runs land under the same key.
    # --- 2026-10-08: personas, qwen 32B and 72B, no contract. 4 jobs.
    # Does saying who the user is move the model to a spec it was never asked for? `student`
    # ("...I do not have much money.") should pull toward 4GB, `editor` (video editor) toward
    # 16GB. Compare against the `shopping` none run and the `ram=4GB` run in
    # laptops_robustness. Own folder, so these can never collide with those runs.
    # {'m': ['qwen'], 's': ['32', '72'], 'a': ['laptops_robustness'], 'c': [''],
    #  'f': ['student', 'editor'], 'n': ['laptops_persona']},

    # --- 2026-10-07: the double contract with the LOW RAM level, qwen 32B and 72B. 2 jobs.
    # We have `screen=14-inch,ram=8GB` (asks for the MIDDLE RAM level) for every model, and
    # `ram=4GB` alone (Phase A) for every qwen. This adds the missing cell of that 2x2: does a
    # screen requirement next to a 4GB request change what the model does with RAM, the way
    # adding 14-inch to 8GB brought the 16GB twin back on top for qwen-32B (progress.md
    # 2026-10-07)? Writes constraints_id "ram=4GB+screen=14-inch", next to the 8GB run.
    #
    # NOT queued: `ram=4GB` alone. It already exists for both models in laptops_robustness
    # (qwen-32B job 1293546, qwen-72B job 1293550, 9900 rows each) - a second run would
    # collide on (family, size, frame, constraints_id).
    # DONE 2026-10-07: landed as jobs 1441002 (32B) and 1441003 (72B), audited 2026-10-08
    # (9900 rows, PMI off). Do not re-run; it would duplicate the key.
    # {'m': ['qwen'], 's': ['32', '72'], 'a': ['laptops_robustness'],
    #  'c': ['screen=14-inch,ram=4GB'], 'n': ['laptops_robustness']},

    # --- DONE 2026-10-07 check: the OLMo stage-2 batch and gap-fillers (27 jobs, ce0ef8f).
    # All landed EXCEPT two, which have no complete run on disk as of 2026-10-07:
    #     olmo-pt 32B  screen=14-inch   (laptops_olmo_pt)
    #     qwen-pt 72B  screen=14-inch   (laptops_robustness_pt)
    # They may still be running or may have died - check squeue / out/ before re-adding
    # them, or a live job and a re-queued one will write two runs under the same key.
    # --- OLMo 2 STAGE 2: the three contracts. 24 jobs. Gate passed 2026-08-18.
    # This is the batch that matters: it is the only route to adherence, kappa and a brand
    # decay for a third family. Sizes ascend so the cheap models finish first.
    #
    # NOTE olmo-pt 32B UNCONSTRAINED is running separately (launched by hand) and is
    # deliberately absent here - a second run under the same key is the collision the data
    # contract warns about. Its CONTRACT runs below are a different key and are fine.
    # {'m': ['olmo'],    's': OLMO, 'a': ['laptops_robustness'], 'c': BASELINE_FOUR[1:],
    #  'p': ['options'],    'n': ['laptops_olmo']},
    # {'m': ['olmo-pt'], 's': OLMO, 'a': ['laptops_robustness'], 'c': BASELINE_FOUR[1:],
    #  'p': ['pretrained'], 'n': ['laptops_olmo_pt']},

    # --- Known holes in the existing families, worth filling while the GPUs are busy. 3 jobs.
    # qwen-pt 72B never got its screen contract, so that model has no screen adherence number
    # and its base-vs-aligned correlation uses 33 points instead of 44.
    # {'m': ['qwen-pt'], 's': ['72'], 'a': ['laptops_robustness'], 'c': ['screen=14-inch'],
    #  'p': ['pretrained'], 'n': ['laptops_robustness_pt']},
    # qwen-72B instruct is the only model missing two of the eight Phase A conditions.
    # {'m': ['qwen'], 's': ['72'], 'a': ['laptops_robustness'],
    #  'c': ['screen=13-inch', 'ram=16GB'], 'n': ['laptops_robustness']},

    # --- SUPERSEDED 2026-08-18 by the OLMo runs above: the gemma format probe.
    # Kept because the question it asks is still open, just no longer the priority.
    # --- FORMAT PROBE for the base models. Unconstrained only, 10 jobs.
    #
    # Why: every gemma base model came out content-blind on the `pretrained` format - it
    # answers by slot (92 / 99.8 / 12 / 0.6 % go to one side) and its positional bias is
    # ~2x its whole preference range. Before spending a full 4-contract batch on a guess,
    # this asks which prompt format, if any, gets a usable signal out of them.
    #
    # Two candidates, one folder each so the runs can never collide on
    # (family, size, constraints_id) the way the frame runs did:
    #   options        the instruct format. Included as the NEGATIVE control: base qwen was
    #                  run this way in June and came out slot-locked at 91-99.6%, so this is
    #                  expected to fail - it closes the loop rather than testing a hope.
    #   pretrained_ab  the same continuation, but lettered options scored on A/B instead of
    #                  1/2. This is the real candidate: with digits the answer-token prior
    #                  and the positional bias are inseparable (option 1 is always slot A),
    #                  and that prior is the best explanation we have for the lock.
    #
    # qwen-pt 7B rides along in both as the POSITIVE control: it already works on the
    # digit format, so if a format breaks it too, the format is bad rather than gemma odd.
    #
    # Unconstrained is enough to gate: |gamma|/S needs only the `none` run, and the content
    # check becomes "does more RAM win?" instead of the contract's free-lunch pairs. Run
    # the full 4-contract batch only for a format that passes.
    # {'m': ['gemma-pt'], 's': GEMMA,   'a': ['laptops_robustness'], 'c': [''],
    #  'p': ['options'],       'n': ['laptops_pt_fmt_options']},
    # {'m': ['qwen-pt'],  's': ['7'],   'a': ['laptops_robustness'], 'c': [''],
    #  'p': ['options'],       'n': ['laptops_pt_fmt_options']},
    # {'m': ['gemma-pt'], 's': GEMMA,   'a': ['laptops_robustness'], 'c': [''],
    #  'p': ['pretrained_ab'], 'n': ['laptops_pt_fmt_ab']},
    # {'m': ['qwen-pt'],  's': ['7'],   'a': ['laptops_robustness'], 'c': [''],
    #  'p': ['pretrained_ab'], 'n': ['laptops_pt_fmt_ab']},

    # --- DONE 2026-08-18: the 28-job base-model batch (31 of 32 runs landed).
    # {'m': ['qwen-pt'],  's': ['0.5', '32', '72'], 'a': ['laptops_robustness'],
    #  'c': BASELINE_FOUR, 'p': ['pretrained']},
    # {'m': ['gemma-pt'], 's': GEMMA,              'a': ['laptops_robustness'],
    #  'c': BASELINE_FOUR, 'p': ['pretrained']},
    #
    # --- Still missing: qwen-pt 72B's screen=14-inch run.
    # {'m': ['qwen-pt'], 's': ['72'], 'a': ['laptops_robustness'],
    #  'c': ['screen=14-inch'], 'p': ['pretrained'], 'n': ['laptops_robustness_pt']},

    # --- DONE 2026-08-16, jobs 1305867-70. Do not re-run; it would duplicate the folder.
    # {'m': ['qwen-pt'], 's': ['7'], 'a': ['laptops_robustness'],
    #  'c': BASELINE_FOUR, 'p': ['pretrained']},
    #
    # --- Also still missing: the two qwen-72B INSTRUCT Phase A runs that never landed.
    # NOTE: exp_name must go back to "laptops_robustness" for these.
    # {'m': ['qwen'], 's': ['72'], 'a': ['laptops_robustness'],
    #  'c': ['screen=13-inch', 'ram=16GB']},
    #
    # --- GRUM Phase A: top and bottom level of each feature, one at a time. 32 runs.
    # Tests whether kappa is level-independent; the 18-parameter GRUM rests on this.
    # Also completes the 2x2 against the 14-inch / 8GB runs we already have.
    # NOTE: exp_name must go back to "laptops_robustness" for these.
    # {'m': ['qwen'],  's': QWEN,  'a': ['laptops_robustness'], 'c': PHASE_A},
    # {'m': ['gemma'], 's': GEMMA, 'a': ['laptops_robustness'], 'c': PHASE_A},
    #
    # --- The genuinely unframed prompt. 8 runs.
    # Every run we have ever done - including the ones we call "unconstrained" - carries
    # "I am looking to buy a laptop." We have never measured brand preference without it,
    # so we cannot say whether the Apple premium is the model's own or the frame's.
    # {'m': ['qwen'],  's': QWEN,  'a': ['laptops_robustness'], 'c': [''], 'f': ['bare']},
    # {'m': ['gemma'], 's': GEMMA, 'a': ['laptops_robustness'], 'c': [''], 'f': ['bare']},
    #
    # --- Later:
    # brand contract - does naming a brand leak into the specs?
    # {'m': ['qwen'],  's': QWEN,  'a': ['laptops_robustness'],
    #  'c': ['brand=Apple', 'brand=Dell', 'brand=ASUS']},
    # {'m': ['gemma'], 's': GEMMA, 'a': ['laptops_robustness'],
    #  'c': ['brand=Apple', 'brand=Dell', 'brand=ASUS']},
    #
    # finishes Nir's Experiment 1 - only qwen-7B was ever collected, the plan said 7B and 72B
    # {'m': ['qwen'], 's': ['72'],
    #  'a': ['laptops_num_ram_screen', 'laptops_txt_ram_screen'], 'c': ['']},
]

dst_path = "scripts/slurms.sh"
prefix = "sbatch -p bml -A bml"
script_path = "scripts/run_data_collection.sh"

lines = []
i = 0
pool_i = {}           # round-robin position inside each node pool
with open(dst_path, 'w') as f:
    f.write("#!/usr/bin/env bash\n")
    f.write("# GENERATED FILE - do not edit by hand.\n")
    f.write("# Edit scripts/create_slurms.py and re-run it, then commit BOTH files together.\n")
    f.write("# (No timestamp on purpose: identical params must produce an identical file,\n")
    f.write("#  so `git diff` shows a changed batch and nothing else.)\n")
    for parameters in parameter_sets:
        for combo in itertools.product(*parameters.values()):
            # Quote each value - an empty string (the no-constraint case) must
            # still produce a real, non-empty shell token ("" not nothing), or
            # the flag before it silently swallows the NEXT flag's value once
            # this line is word-split by the shell.
            kv = dict(zip(parameters.keys(), combo))
            # a batch may span several exp_names (e.g. one folder per prompt format), so a
            # parameter set can override the global with its own 'n'
            name = kv.pop('n', exp_name)
            flags = " ".join(f'-{k} "{v}"' for k, v in kv.items())
            pool = NODES_BY_SIZE.get(kv['s'], nodes)
            k = pool_i.get(tuple(pool), 0)
            node = pool[k % len(pool)]
            pool_i[tuple(pool)] = k + 1
            cmd = f"{prefix} -w {node} {script_path} {flags} -n {name}"
            f.write(cmd + "\n")
            lines.append((name, f"{kv['m']}-{kv['s']}B", kv.get('p', '-'), kv.get('f', 'shopping'),
                          kv.get('c', '') or 'none', node))
            i += 1

print(f"wrote {i} jobs to {dst_path}")
for name in sorted({l[0] for l in lines}):
    rows = [l for l in lines if l[0] == name]
    print(f"  data/{name}/  ({len(rows)} jobs)")
    for _, model, tmpl, frame, con, node in rows:
        print(f"      {model:14} [{tmpl}] f={frame:12} c={con:10} -> {node}")
