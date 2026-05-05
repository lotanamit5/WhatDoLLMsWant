import torch
import torch.nn.functional as F

from typing import List
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer


class Agent(ABC):
    @abstractmethod
    def query(self, prompt: str, option_a: str, option_b: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement this method")

class HFAgent(Agent):
    def __init__(self, model_id):
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
    def query(self, prompt: str, labels: List[str]) -> List[float]:
        scores = []
        for label in labels:
            # 1. Construct the full sequence
            full_text = prompt + " " + label
            
            # 2. Tokenize inputs
            input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.model.device)
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            # 3. Find where the label starts
            prompt_len = prompt_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            # 4. Calculate Loss (Negative Log Likelihood) for the label part only
            label_logits = logits[0, prompt_len-1 : -1, :] 
            label_ids = input_ids[0, prompt_len:]
            
            loss = F.cross_entropy(label_logits, label_ids, reduction='sum')
            
            scores.append(-loss.item())
            
        return scores

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