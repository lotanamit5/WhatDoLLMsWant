import torch
import torch.nn.functional as F

from typing import List
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer


class Agent(ABC):
    @abstractmethod
    def query(self, prompt: str, item_a: str, item_b: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement this method")

class HFAgent(Agent):
    def __init__(self, model_id):
        self.model_id = model_id
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_id)
        
    def _load_model_and_tokenizer(self, model_id):
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
            print("No GPU found. Running on CPU (not recommended for large models).")
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
        
        return model, tokenizer

class PretrainedAgent(HFAgent):
    def query(self, prompt: str, alternative_pair: List[str]) -> List[float]:
        # Define neutral target tokens to evaluate
        target_tokens = ["1", "2"]
        
        # Grab the exact token IDs. Using [-1] handles tokenizers that prepend spaces/meta characters.
        target_ids = [self.tokenizer.encode(t, add_special_tokens=False)[-1] for t in target_tokens]
        
        item_a, item_b = alternative_pair
        
        # Format the user's template to reference the neutral identifiers
        question = prompt.format(A="Option 1", B="Option 2")
        
        def get_target_logprobs(prompt_text: str) -> tuple:
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Isolate the logits for the next predicted token
                next_token_logits = outputs.logits[0, -1, :]
                # Convert to log probabilities over the full vocabulary
                logprobs = F.log_softmax(next_token_logits, dim=-1)
                
            return logprobs[target_ids[0]].item(), logprobs[target_ids[1]].item()

        # --- Pass 1: Original Order ---
        prompt_1 = f"Option 1: {item_a}\nOption 2: {item_b}\n\n{question}\nAnswer with strictly '1' or '2':"
        p1_logprob_1, p1_logprob_2 = get_target_logprobs(prompt_1)
        
        # Map token probabilities to the respective items
        pass1_score_a = p1_logprob_1  # "1" corresponds to item_a
        pass1_score_b = p1_logprob_2  # "2" corresponds to item_b
        
        # --- Pass 2: Swapped Order (Positional Bias Mitigation) ---
        prompt_2 = f"Option 1: {item_b}\nOption 2: {item_a}\n\n{question}\nAnswer with strictly '1' or '2':"
        p2_logprob_1, p2_logprob_2 = get_target_logprobs(prompt_2)
        
        # Map token probabilities to the respective items (Swapped mapping)
        pass2_score_b = p2_logprob_1  # "1" corresponds to item_b
        pass2_score_a = p2_logprob_2  # "2" corresponds to item_a
        
        # Average the logprobs across both passes
        final_score_a = (pass1_score_a + pass2_score_a) / 2.0
        final_score_b = (pass1_score_b + pass2_score_b) / 2.0
        
        return [final_score_a, final_score_b]

class InstructedHFAgent(PretrainedAgent):
    SYSTEM_MESSAGE = "You are a helpful assistant. Answer shortly with only your choice with no explanation.\n\n"
    def __init__(self, model_id):
        super().__init__(model_id)
        self.system_prompt = self.SYSTEM_MESSAGE
    
    def _convert_to_chat_template(self, text):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def query(self, prompt: str, labels: List[str]) -> List[float]:
        prompt = self._convert_to_chat_template(prompt)
        return super().query(prompt, labels)

def load_qwen2_5_agent(model_size: int, instructed=False):
    optional_sizes = [0.5, 7, 32, 72]
    assert model_size in optional_sizes, f"Model size must be one of {optional_sizes}"
    
    model_id = f"Qwen/Qwen2.5-{model_size}B"
    if instructed:
        model_id += "-instruct"
        return InstructedHFAgent(model_id)
    
    return PretrainedAgent(model_id)

def load_gemma3_agent(model_size: int, instructed=False):
    optional_sizes = [1, 4, 12, 27]
    assert model_size in optional_sizes, f"Model size must be one of {optional_sizes}"
    
    model_type = "it" if instructed else "pt"
    
    model_id = f"google/gemma-3-{model_size}b-{model_type}"
    
    if instructed:
        return InstructedHFAgent(model_id)
    return PretrainedAgent(model_id)


def agent_factory(model: str, model_size, instructed=False) -> Agent:
    model = model.lower()
    if model == "qwen25":
        return load_qwen2_5_agent(model_size, instructed)
    elif model == "gemma3":
        return load_gemma3_agent(model_size, instructed)
    else:
        raise ValueError(f"Unsupported model: {model}")