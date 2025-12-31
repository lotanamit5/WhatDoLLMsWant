import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id="Qwen/Qwen2.5-0.5B"):
    """Loads the model and tokenizer."""
    
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use float16 or bfloat16 for efficiency if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval() # Set to evaluation mode
    return model, tokenizer, device

def get_winner_logits(model, tokenizer, prompt_text, option_a, option_b, device):
    """
    Determines the winner by comparing the probability of the first token
    of option_a vs option_b given the prompt.
    """
    # Encode the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # We need the token IDs for the options (e.g., " Red" vs " Blue")
    # Note: We add a leading space often required by tokenizers for the start of a word
    # TODO: this assumes single-token options; extend for multi-token options if needed
    ids_a = tokenizer.encode(" " + option_a, add_special_tokens=False)
    ids_b = tokenizer.encode(" " + option_b, add_special_tokens=False)
    if len(ids_a) != 1 or len(ids_b) != 1:
        print(f"Warning: One of the options is not a single token. ids_a: {ids_a}, ids_b: {ids_b}") 
    id_a = ids_a[0]
    id_b = ids_b[0]

    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits of the last token in the prompt
    next_token_logits = outputs.logits[0, -1, :]

    score_a = next_token_logits[id_a].item()
    score_b = next_token_logits[id_b].item()
    winner = option_a if score_a > score_b else option_b
    
    return winner, score_a, score_b
