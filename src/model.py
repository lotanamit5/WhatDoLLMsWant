import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_id="Qwen/Qwen2.5-0.5B"):
    """Loads the model and tokenizer."""
    
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use float16 or bfloat16 for efficiency if GPU is available
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs.")
        # Distribute the model across all available GPUs
        # Reserve some memory for activations (KV cache) by limiting max_memory per GPU
        # L4 has 24GB. Setting limit to ~20GB forces distribution across more GPUs
        # max_memory = {i: "20GB" for i in range(num_gpus)}
        max_memory = None
        if num_gpus > 8:
             print("Warning: Using > 8 GPUs may cause peer mapping errors. Consider reducing GPU count.")
        device_map = "auto"
        device = "cuda"
    else:
        device_map = None
        max_memory = None
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device_map,
        max_memory=max_memory
    )
    model.eval() # Set to evaluation mode
    
    # If device_map is auto, the model is split. We can use model.device for the first layer,
    # but generally we just need to know we are on cuda.
    # Returning model.device is safer than hardcoded string if we want to be precise,
    # but for 'auto', model.device usually returns the device of the first parameter.
    return model, tokenizer

def get_top_k_tokens(model, tokenizer, prompt_text, k=5):
    """
    Returns the top-k tokens for the given prompt.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    next_token_logits = outputs.logits[0, -1, :]
    top_logits = torch.topk(next_token_logits, k)
    top_indices = top_logits.indices.tolist()
    
    # decode tokens
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    return top_tokens

def get_winner(model, tokenizer, prompt_text, option_a, option_b):
    """
    Determines the winner by comparing the cumulative probability of valid tokens
    for option_a vs option_b.
    Valid tokens for option 'red': ['red', ' red', 'Red', ' Red', 'A', ' A'] (and similarly for B)
    """
    # Encode the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # 1. Define candidate strings for each option
    # Note: 'A' and 'B' are fixed aliases for option_a and option_b respectively in our prompts
    candidates_a = [
        option_a, 
        " " + option_a, 
        option_a.capitalize(), 
        " " + option_a.capitalize(),
        "A", 
        " A"
    ]
    candidates_b = [
        option_b, 
        " " + option_b, 
        option_b.capitalize(), 
        " " + option_b.capitalize(),
        "B", 
        " B"
    ]
    
    # Remove duplicates
    candidates_a = list(set(candidates_a))
    candidates_b = list(set(candidates_b))

    # 2. Get token IDs for all valid candidates
    # We only care about single-token candidates. Multi-token ones are ignored for "next token" probability.
    
    def get_valid_token_ids(candidates):
        valid_ids = []
        for cand in candidates:
            # add_special_tokens=False is crucial
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                valid_ids.append(ids[0])
            else:
                pass 
        return list(set(valid_ids))

    ids_a_list = get_valid_token_ids(candidates_a)
    ids_b_list = get_valid_token_ids(candidates_b)
    
    # 3. Get logits and Probabilities
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits of the last token in the prompt
    next_token_logits = outputs.logits[0, -1, :]
    
    # Softmax to get probabilities
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    
    # 4. Sum probabilities
    prob_a = sum(probs[idx].item() for idx in ids_a_list)
    prob_b = sum(probs[idx].item() for idx in ids_b_list)
    
    winner = option_a if prob_a > prob_b else option_b
    
    return winner, prob_a, prob_b


def get_choice_via_scoring(model, tokenizer, prompt, choice_a, choice_b):
    """
    Determines winner by prefilling the answer and checking which one 
    has lower perplexity (higher likelihood).
    """
    options = [choice_a, choice_b]
    scores = []

    for option in options:
        # 1. Prepare Full Text
        # Note: Add a leading space if your prompt ends without one
        full_text = prompt + " " + option 
        
        # 2. Tokenize
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        
        # 3. Identify where the "Option" part starts
        # We only want to score the option tokens, not the prompt tokens
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        
        # 4. Forward Pass (No Generation!)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 5. Calculate Loss for the Option tokens only
        # Shift logits and labels so we predict the next token
        shift_logits = logits[..., prompt_len-1:-1, :].contiguous()
        shift_labels = input_ids[..., prompt_len:].contiguous()
        
        # Cross Entropy Loss (Average negative log likelihood)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1),
            reduction='mean' # Use 'sum' if you prefer total probability over average
        )
        
        scores.append(loss.item())

    # Lower loss means higher probability -> Winner
    winner = choice_a if scores[0] < scores[1] else choice_b
    
    return winner, scores[0], scores[1]
