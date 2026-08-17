import os
import sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

exp_name = "laptops_robustness_pt"  # CHANGE WHEN RUNNING
nodes = [
    'plato2',
    'plotinus1',
    'plotinus2',
]

QWEN = ['0.5', '7', '32', '72']
GEMMA = ['1', '4', '12', '27']

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
    # --- The rest of the base (non-instruct) models: 3 qwen sizes + all 4 gemma. 28 runs.
    # Completes the base-vs-aligned comparison, which so far rests on qwen-7B alone
    # (r = 0.990 on scale-free weights, ~9x louder aligned - one point is not a trend).
    # Uses the `pretrained` template set: it ends mid-sentence ("...I prefer Option ")
    # rather than asking a question, which is what a base model is trained to continue.
    # Same exp_name as the qwen-7B base runs; the template set differs from
    # laptops_robustness, so base runs live in their own folder.
    {'m': ['qwen-pt'],  's': ['0.5', '32', '72'], 'a': ['laptops_robustness'],
     'c': BASELINE_FOUR, 'p': ['pretrained']},
    {'m': ['gemma-pt'], 's': GEMMA,              'a': ['laptops_robustness'],
     'c': BASELINE_FOUR, 'p': ['pretrained']},

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
            flags = " ".join(f'-{k} "{v}"' for k, v in zip(parameters.keys(), combo))
            node = nodes[i % len(nodes)]
            cmd = f"{prefix} -w {node} {script_path} {flags} -n {exp_name}"
            f.write(cmd + "\n")
            i += 1

print(f"wrote {i} jobs to {dst_path}")
