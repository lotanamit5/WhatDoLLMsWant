import os
import torch
import huggingface_hub
import torch.nn.functional as F

from dotenv import load_dotenv
from typing import List
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer


class Agent(ABC):
    @abstractmethod
    def query(self, prompt: str, labels: List[str]) -> List[float]:
        raise NotImplementedError("Subclasses must implement this method")

class HFAgent(Agent):
    SYSTEM_MESSAGE = "You are a helpful assistant. Answer shortly with only your choice with no explanation.\n\n"
    
    def __init__(self, model_id,
                 normalize_pmi: bool = False,
                 pmi_base_context: str = None):
        load_dotenv()
        huggingface_hub.login(token=os.getenv("HF_TOKEN"))
        self.model_id = model_id
        self.normalize_pmi = normalize_pmi
        self.pmi_base_context = pmi_base_context
        self.system_prompt = self.SYSTEM_MESSAGE
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_id)
    
    @classmethod
    def with_model(cls, model, tokenizer):
        new_agent = cls.__new__(cls) 
        new_agent.model = model
        new_agent.tokenizer = tokenizer
        if hasattr(tokenizer, 'name_or_path'):
            new_agent.model_id = tokenizer.name_or_path
        new_agent.system_prompt = cls.SYSTEM_MESSAGE
        
        return new_agent

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

    @staticmethod
    def _load_model_and_tokenizer(model_id, cache_dir=None):
        """Loads the model and tokenizer."""
        cwd = os.getcwd()
        cache_dir = cwd + "/huggingface/.cache"
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Use float16 or bfloat16 for efficiency if GPU is available
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Found {num_gpus} GPUs.")
            if num_gpus > 8:
                print("Warning: Using > 8 GPUs may cause peer mapping errors. Consider reducing GPU count.")
            device_map = "auto"
            device = "cuda"
        else:
            print("No GPU found. Running on CPU (not recommended for large models).")
            device_map = None
            device = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        model.eval() # Set to evaluation mode
        
        return model, tokenizer

class InstructedHFAgent(HFAgent):
    def query(self, prompt: str, labels: List[str]):
            
        formatted_prompt = self._convert_to_chat_template(prompt)

        # Get raw conditional logprobs: log P(label | prompt)
        prompt_scores = self._get_logprobs(formatted_prompt, labels)
        
        if not self.normalize_pmi:
            return prompt_scores
        
        # PMI Normalization: get base logprobs: log P(label | empty_template)
        base_prompt = self.pmi_base_context if self.pmi_base_context else prompt.split('\n')[-1]
        base_context = self._convert_to_chat_template(base_prompt)
        base_scores = self._get_logprobs(base_context, labels)

        # Subtract base from prompt to isolate the prompt's informational gain
        pmi_scores = prompt_scores - base_scores

        return pmi_scores

    def _get_logprobs(self, prompt: List[str], labels: List[str]) -> torch.Tensor:
        """Helper to extract the logprobs of the final label tokens across a batch."""
        # 1. Group combinations: C1+L1, C1+L2, C2+L1, C2+L2...
        input_with_answers = [prompt + label for label in labels]
        
        # 2. Extract target token IDs (using the last token of each label)
        labels_tokens = self.tokenizer(labels, add_special_tokens=False)["input_ids"]
        last_label_tokens = [toks[-1] for toks in labels_tokens]
        
        # 3. Expand target tokens to match the flat batch dimension
        # e.g., if labels are [L1, L2], repeated for N contexts -> [L1, L2, L1, L2...]
        batch_target_tokens = last_label_tokens
        
        # Force right padding so length calculations perfectly map to indices
        original_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        input_enc = self.tokenizer(
            input_with_answers,
            return_tensors="pt",
            padding="longest",
        ).to(self.model.device)
        
        self.tokenizer.padding_side = original_padding

        with torch.no_grad():
            logits = self.model(**input_enc).logits
            
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 4. Calculate indices safely (attention mask sums give exact sequence lengths)
        seq_lengths = input_enc["attention_mask"].sum(-1)
        # The target is at length - 1. The token predicting the target is at length - 2.
        predictor_indices = seq_lengths - 2
        
        # 5. Extract scores using advanced indexing (Fixes the matrix/diag crash)
        batch_indices = torch.arange(len(input_with_answers), device=self.model.device)
        target_token_indices = torch.tensor(batch_target_tokens, device=self.model.device)
        
        scores = log_probs[batch_indices, predictor_indices, target_token_indices]
        
        # 6. Reshape back to (Num Labels)
        logits_per_label = scores.squeeze(-1)
        return logits_per_label


qwen2_5_sizes = ['0.5', '7', '32', '72']
gemma3_sizes = ['1', '4', '12', '27']

def load_qwen2_5_agent(model_size: float):
    assert model_size in qwen2_5_sizes, f"Model size must be one of {qwen2_5_sizes}"
    
    model_id = f"Qwen/Qwen2.5-{model_size}B-instruct"

    return InstructedHFAgent(model_id)

def load_gemma3_agent(model_size: float):
    assert model_size in gemma3_sizes, f"Model size must be one of {gemma3_sizes}"
    
    model_id = f"google/gemma-3-{model_size}b-it"
    
    return InstructedHFAgent(model_id)
