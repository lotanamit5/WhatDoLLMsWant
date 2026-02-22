import os
from datetime import date
import pandas as pd
import numpy as np
import re

def config2title(config):
    return f"{config['model_name'].replace('Qwen/', '')}"

def get_experiments_with_prefix(prefix,
                                base_folder="experiments"):
    good_names = []
    for name in os.listdir(base_folder):
        if name.startswith(prefix):
            good_names.append(name)
    return good_names

def get_experiments_in_range(start_date, 
                             end_date=None,
                             base_folder="experiments"):
    # date format: "%Y-%m-%d_%H-%M-%S"
    good_names = []
    for name in os.listdir(base_folder):
        try:
            exp_date = date.fromisoformat(name.split("_")[0])
            if not end_date and exp_date == start_date:
                good_names.append(name)
            elif end_date and start_date <= exp_date <= end_date:
                good_names.append(name)
            
        except ValueError:
            continue
    return good_names

def get_experiments_with_regex(regex, base_folder="experiments"):
    good_names = []
    for name in os.listdir(base_folder):
        if re.search(regex, name):
            good_names.append(name)
    return good_names

def convert_bt_weights_to_probs(df_weights):
    bt = pd.pivot(data=df_weights, index="iteration", columns="Item", values="BT_Score")
    bt_values = np.exp(bt.values)
    bt = pd.DataFrame(bt_values, index=bt.index, columns=bt.columns)
    return bt