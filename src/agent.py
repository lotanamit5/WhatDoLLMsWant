import os
import torch
import torch.nn.functional as F

from typing import List
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer

class Agent(ABC):
    @abstractmethod
    def query(self, prompt: str, labels: List[str]) -> List[float]:
        raise NotImplementedError("Subclasses must implement this method")

class HFAgent(Agent):
    SYSTEM_MESSAGE = "You are a helpful assistant. Answer shortly with only your choice with no explanation.\n\n"
    
    def __init__(self, model_id):
        self.model_id = model_id
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
    def query(self, prompts, labels):
        if not isinstance(prompts, list):
            prompts = [prompts]
        prompts = [self._convert_to_chat_template(p) for p in prompts]
        # concat labels to the corrposnded input text
        input_with_answers = [i + label for label in labels for i in prompts]
        # get labels tokens ids
        labels_tokens = self.tokenizer(labels, add_special_tokens=False)["input_ids"]
        # get the last token id of each label
        labels_tokens = [label[-1] for label in labels_tokens]
        # Ensure pad token exists before padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Get encodings for each input text using direct call instead of batch_encode_plus
        input_enc = self.tokenizer(
            input_with_answers,
            return_tensors="pt",
            padding="longest",
        )
        for k, v in input_enc.items():
            input_enc[k] = v.to(self.model.device)

        # Get model output logits
        with torch.no_grad():
            model_output = self.model(**input_enc)

        # Compute the log probabilities associated with each of the labels
        labels_log_probs = F.log_softmax(model_output.logits, dim=-1)

        # Get the ids of the token before the last token before padding (to see the probablity of the last token given the one before the last token)
        before_padding_ids = (
            input_enc["input_ids"].ne(self.tokenizer.pad_token_id).sum(-1) - 2
        )

        # Collect labels scores from the -2 token in labels_log_probs (the one that predict the last token)
        # and collect for each line the id in labels_tokens
        labels_scores = labels_log_probs[:, before_padding_ids, labels_tokens]

        # Need just the diagonal of the matrix, as it the prob of the label for each line
        labels_scores = torch.diag(labels_scores)

        # metadata = {
        #     'input_ids': input_enc.input_ids,
        #     'logits': model_output.logits,
        # }

        return labels_scores #, metadata


def load_qwen2_5_agent(model_size: float):
    optional_sizes = ['0.5', '7', '32', '72']
    assert model_size in optional_sizes, f"Model size must be one of {optional_sizes}"
    
    model_id = f"Qwen/Qwen2.5-{model_size}B-instruct"

    return InstructedHFAgent(model_id)

def load_gemma3_agent(model_size: float):
    optional_sizes = ['1', '4', '12', '27']
    assert model_size in optional_sizes, f"Model size must be one of {optional_sizes}"
    
    model_id = f"google/gemma-3-{model_size}b-it"
    
    return InstructedHFAgent(model_id)
