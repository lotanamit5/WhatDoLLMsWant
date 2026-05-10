import itertools

from src import prompts, alternatives
from src.agent import *

MODEL_ALIASES = {
    'qwen05I': "qwen/qwen2.5-0.5B-instruct",
    'qwen72I': "qwen/qwen2.5-72B-instruct",
}

AGENTS = {
    "itay_v0": ItayHFAgentV0,
    "itay_v1": ItayHFAgentV1,
    "lotan_v0": LotanHFAgentV0,
    "lotan_v1": LotanHFAgentV1,
    "lotan_v2": LotanHFAgentV2,
    "lotan_v3": LotanHFAgentV3,
}

ALTERNATIVES_ALIASES = {
    "colors": alternatives.colors,
    "foods": alternatives.foods,
    "cars": alternatives.cars,
    "stocks": alternatives.stocks,
    'laptops': alternatives.laptops,
    'laptop_brands': alternatives.laptop_brands,
}

# TEMPLATE_ALIASES = {
#     'oneliners': prompts.general_comparisons,
#     'options': prompts.options_comparisons,
# }

def collect_data_options(model_alias, alternatives_alias, exp_name):
    exp_dir = os.path.join("data", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"options-{alternatives_alias}-{model-alias}-{timestamp}"

    alternatives = ALTERNATIVES_ALIASES[alternatives_alias]
    model_id = MODEL_ALIASES[model_alias]

    # model, tokenizer = HFAgent._load_model_and_tokenizer(model_id)
    agent = LotanHFAgentV3(model_id)

    records = []
    labels = ['1', '2']
    for template in prompts.options_comparisons:
        print("Template: ", template)
        for option_a, option_b in itertools.permutations(alternatives, 2):
            print("Option_A: ", option_a)
            print("Option_B: ", option_b)
            prompt = template.format(A=option_a, B=option_b)
            print("Prompt: ", prompt)
            score_a, score_b = agent.query(prompt, labels, prefill="Option ")
            norm_score_a, norm_score_b = agent.query(prompt, labels, prefill="Option ", normalize="Answer: ")
            records.append({
                'template': template,
                'option_a': option_a,
                'option_b': option_b,
                'prompt': prompt,
                'score_a': score_a,
                'score_b': score_b,
                'norm_score_a': norm_score_a,
                'norm_score_b': norm_score_b,
            })
            
    df = pd.DataFrame(records)
    df.to_csv(file_path)
    print("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data")
    parser.add_argument("--cluster_job", type=str, default="", help="Job name")
    parser.add_argument("--model", type=str, required=True, help="Model alias")
    parser.add_argument("--alternatives", type=str, required=True, help="Alternatives alias")

    args = parser.parse_args()

# itay_v0 = ItayHFAgentV0.with_loaded_model(model, tokenizer)
# itay_v0_score_a, itay_v0_score_b = itay_v0.query([itay_v0.convert_to_chat_format(prompt)], labels)
# itay_v1 = ItayHFAgentV1.with_loaded_model(model, tokenizer)
# itay_v1_score_a, itay_v1_score_b = itay_v1.query([itay_v1.convert_to_chat_format(prompt)], labels)
# lotan_v0 = LotanHFAgentV0.with_loaded_model(model, tokenizer)
# lotan_v0_score_a, lotan_v0_score_b = lotan_v0.query(prompt, labels)
# lotan_v1 = LotanHFAgentV1.with_loaded_model(model, tokenizer)
# lotan_v1_score_a, lotan_v1_score_b = lotan_v1.query(prompt, labels)
# lotan_v2 = LotanHFAgentV2.with_loaded_model(model, tokenizer)
# lotan_v2_score_a, lotan_v2_score_b = lotan_v2.query(prompt, labels, prefill="Option ")
# lotan_v3 = LotanHFAgentV3.with_loaded_model(model, tokenizer)
# lotan_v3_score_a, lotan_v3_score_b = lotan_v3.query(prompt, labels, prefill="Option ")
# lotan_v3_norm_score_a, lotan_v3_norm_score_b = lotan_v3.query(prompt, labels, prefill="Option ", normalize="Answer: ")
# records.append({
#     'template': template,
#     'option_a': option_a,
#     'option_b': option_b,
#     'prompt': prompt,
#     'itay_v0_score_a': itay_v0_score_a,
#     'itay_v0_score_b': itay_v0_score_b,
#     'itay_v1_score_a': itay_v1_score_a,
#     'itay_v1_score_b': itay_v1_score_b,
#     'lotan_v0_score_a': lotan_v0_score_a,
#     'lotan_v0_score_b': lotan_v0_score_b,
#     'lotan_v1_score_a': lotan_v1_score_a,
#     'lotan_v1_score_b': lotan_v1_score_b,
#     'lotan_v2_score_a': lotan_v2_score_a,
#     'lotan_v2_score_b': lotan_v2_score_b,
#     'lotan_v3_score_a': lotan_v3_score_a,
#     'lotan_v3_score_b': lotan_v3_score_b,
#     'lotan_v3_norm_score_a': lotan_v3_norm_score_a,
#     'lotan_v3_norm_score_b': lotan_v3_norm_score_b,
# })