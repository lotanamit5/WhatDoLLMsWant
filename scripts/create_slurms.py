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
    # --- THIRD FAMILY: OLMo 2, base and instruct. Stage 1 = unconstrained only, 8 jobs.
    #
    # qwen replicated the "preferences are pretrained, alignment is ~9x louder" result;
    # gemma could not be measured at all. Rather than keep probing gemma's prompt format,
    # this runs the ordinary experiment on a third family.
    #
    # Why OLMo 2: the claim under test is that these preferences come from PRETRAINING, and
    # OLMo 2 is the only family we checked whose pretraining corpus and full training
    # pipeline are public - so a positive result is traceable to data rather than a black
    # box. It is also ungated (no license-approval risk) and has four sizes: 1/7/13/32B.
    #
    # STAGED on the gemma lesson: do not spend a 4-contract batch before knowing the base
    # model produces a usable signal. These 8 unconstrained runs are enough to gate
    # (|gamma|/S, "does more RAM win?") AND already give a complete base-vs-aligned
    # comparison - r on the scale-free weights, and the loudness ratio. Stage 2 below adds
    # the three contracts, and is only worth launching if the base models pass.
    #
    # One folder per side, matching the template split: instruct uses `options`, base uses
    # `pretrained`, exactly as for qwen.
    {'m': ['olmo'],    's': OLMO, 'a': ['laptops_robustness'], 'c': [''],
     'p': ['options'],    'n': ['laptops_olmo']},
    {'m': ['olmo-pt'], 's': OLMO, 'a': ['laptops_robustness'], 'c': [''],
     'p': ['pretrained'], 'n': ['laptops_olmo_pt']},

    # --- Stage 2, uncomment once stage 1 passes the gate: the other three contracts. 24 jobs.
    # {'m': ['olmo'],    's': OLMO, 'a': ['laptops_robustness'], 'c': BASELINE_FOUR[1:],
    #  'p': ['options'],    'n': ['laptops_olmo']},
    # {'m': ['olmo-pt'], 's': OLMO, 'a': ['laptops_robustness'], 'c': BASELINE_FOUR[1:],
    #  'p': ['pretrained'], 'n': ['laptops_olmo_pt']},

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

i = 0
with open(dst_path, 'w') as f:
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
            node = nodes[i % len(nodes)]
            cmd = f"{prefix} -w {node} {script_path} {flags} -n {name}"
            f.write(cmd + "\n")
            i += 1

print(f"wrote {i} jobs to {dst_path}")
