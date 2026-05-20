import os
import sys
import argparse
import itertools
import pandas as pd

from datetime import datetime
from tqdm.auto import tqdm

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
import src.alternatives
import src.prompts
from src.agent import InstructedHFAgent, load_gemma3_agent, load_qwen2_5_agent

MODEL_FAMILY_ALIASES = {
    'qwen': load_qwen2_5_agent,
    'gemma': load_gemma3_agent,
}

ALTERNATIVES_ALIASES = {
    'colors': src.alternatives.colors,
    'foods': src.alternatives.foods,
    'cars': src.alternatives.cars,
    'stocks': src.alternatives.stocks,
    # 'laptops': src.alternatives.laptops,
    'laptop_brands': src.alternatives.laptop_brands,
}

def validate_tokens():
    # Parameters
    agent_loader = MODEL_FAMILY_ALIASES['qwen']
    agent = agent_loader('0.5')
    for set_name, alts in ALTERNATIVES_ALIASES.items():
        print(f"Validating {set_name} alternatives:")
        labels_tokens = agent.tokenizer(alts, add_special_tokens=False)["input_ids"]
        for idx, tokens in enumerate(labels_tokens):
            decoded_tokens = [agent.tokenizer.decode([t]) for t in tokens]
            if len(tokens) > 1:
                print(f"WARNING: '{alts[idx]}' is tokenized into multiple tokens: {tokens}")
                print(f"  Decoded tokens: {decoded_tokens}")
            else:
                print(f"'{alts[idx]}' is tokenized into a single token: {tokens[0]}")
                print(f"  Decoded token: {decoded_tokens[0]}")
        print("\n")
    print("Finished!")

if __name__ == "__main__":
    validate_tokens()