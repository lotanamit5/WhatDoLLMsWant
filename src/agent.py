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
    SYSTEM_MESSAGE = "You are a helpful assistant. Answer shortly with only your choice with no explanation."
    
    def __init__(self, model_id,
                 normalize_pmi: bool = True):
        load_dotenv()
        huggingface_hub.login(token=os.getenv("HF_TOKEN"))
        self.model_id = model_id
        self.normalize_pmi = normalize_pmi
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
        # bfloat16 (not float16) for GPU: these models are trained/released in
        # bfloat16, and float16's narrower exponent range overflows to inf/nan
        # on larger models (e.g. Gemma-3 4B+) - bfloat16 has float32's exponent
        # range so it doesn't overflow.

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
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        model.eval() # Set to evaluation mode
        
        return model, tokenizer

    def _get_logprobs(self, prompt: str, labels: List[str]) -> torch.Tensor:
        assert prompt.endswith((" ", "\n"))
        print(f"Getting log probabilities for prompt:\n{prompt}\nwith labels: {labels}\n")
        input_with_answers = [prompt + label for label in labels]
        print(f"Input with answers:")
        for i, inp in enumerate(input_with_answers):
            print(i)
            print(inp)
        
        original_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        input_enc = self.tokenizer(
            input_with_answers,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False 
        ).to(self.model.device)
        
        self.tokenizer.padding_side = original_padding

        with torch.no_grad():
            logits = self.model(**input_enc).logits
            
        log_probs = F.log_softmax(logits, dim=-1)
        
        seq_lengths = input_enc["attention_mask"].sum(-1)
        predictor_indices = seq_lengths - 2

        batch_indices = torch.arange(len(input_with_answers), device=self.model.device)
        # Take the target token from the ENCODED sequence, not by re-tokenising the label on
        # its own. Tokenizers merge across the prompt/label boundary: with a prompt ending in
        # a space, " A" becomes a single token while " " + "1" stays two. So a label tokenised
        # in isolation can be a different id from the one that actually ends the sequence,
        # and the score would be read off the wrong token entirely. Identical for the digit
        # labels used so far (verified); required for any letter label.
        target_token_indices = input_enc["input_ids"][batch_indices, seq_lengths - 1]
        print(f"Target tokens: {target_token_indices.tolist()} "
              f"(decoded: {[self.tokenizer.decode([t]) for t in target_token_indices.tolist()]})\n")

        scores = log_probs[batch_indices, predictor_indices, target_token_indices]
        
        print(f"Log probabilities for each label: {scores}\n")
        
        return scores

class InstructedHFAgent(HFAgent):
    def __init__(self, model_id, normalize_pmi: bool = True,
                 labels: list = None):
        super().__init__(model_id, normalize_pmi)
        self.labels = labels if labels else ["Option 1", "Option 2"]
            
    def query(self, prompt: str, labels: List[str] = None):
        labels = labels if labels else self.labels
        formatted_prompt = self._convert_to_chat_template(prompt)
        prompt_scores = self._get_logprobs(formatted_prompt, labels)
        
        if self.normalize_pmi:
            base_prompt = prompt.split('\n')[-1]
            base_context = self._convert_to_chat_template(base_prompt)
            base_scores = self._get_logprobs(base_context, labels)
            return (prompt_scores - base_scores).tolist()

        return prompt_scores.tolist()

class PretrainedHFAgent(HFAgent):
    # "1"/"2", not " 1"/" 2": the pretrained templates already end with a space, so the
    # label must not carry one too. Either way only the last token is scored and it is
    # the bare digit - the space is context. See src/prompts.py.
    def __init__(self, model_id, normalize_pmi: bool = True,
                 labels: list = None):
        super().__init__(model_id, normalize_pmi)
        self.labels = labels if labels else ["1", "2"]
    
    def query(self, prompt: str, labels: List[str] = None):
        labels = labels if labels else self.labels
        prompt_scores = self._get_logprobs(prompt, labels)
        
        # if self.normalize_pmi:

        #     base_prompt = prompt.split('\n')[-1]
        #     base_scores = self._get_logprobs(base_prompt, labels)

        #     return (prompt_scores - base_scores).tolist()

        return prompt_scores.tolist()    


qwen2_5_sizes = ['0.5', '7', '32', '72']
gemma3_sizes = ['1', '4', '12', '27']

# PMI off since 2026-08-12. The base context is `prompt.split('\n')[-1]`, which is always
# the literal "Answer: " - so PMI subtracted one constant C per run (+2.25 to +10.50).
# A constant cannot touch the feature weights (it lands in the intercept), but it IS the
# intercept, so it corrupted gamma; and it moved the threshold for sign(margin) off zero,
# which silently changed every adherence number. See docs/progress.md 2026-08-12.
def load_qwen2_5_agent(model_size: float, labels: list = None):
    assert model_size in qwen2_5_sizes, f"Model size must be one of {qwen2_5_sizes}"

    model_id = f"Qwen/Qwen2.5-{model_size}B-instruct"

    return InstructedHFAgent(model_id, normalize_pmi=False, labels=labels)

def load_gemma3_agent(model_size: float, labels: list = None):
    assert model_size in gemma3_sizes, f"Model size must be one of {gemma3_sizes}"

    model_id = f"google/gemma-3-{model_size}b-it"

    return InstructedHFAgent(model_id, normalize_pmi=False, labels=labels)

# The base models. normalize_pmi=False is passed explicitly even though the PMI block in
# PretrainedHFAgent is commented out - so that uncommenting it cannot silently turn PMI
# back on for these runs. Base models take the raw prompt: no chat template, no system
# message, and the `pretrained` template set from prompts.py.
def load_qwen2_5_pt_agent(model_size: float, labels: list = None):
    assert model_size in qwen2_5_sizes, f"Model size must be one of {qwen2_5_sizes}"

    model_id = f"Qwen/Qwen2.5-{model_size}B"

    return PretrainedHFAgent(model_id, normalize_pmi=False, labels=labels)

def load_gemma3_pt_agent(model_size: float, labels: list = None):
    assert model_size in gemma3_sizes, f"Model size must be one of {gemma3_sizes}"

    model_id = f"google/gemma-3-{model_size}b-pt"

    return PretrainedHFAgent(model_id, normalize_pmi=False, labels=labels)

# `labels` are the answer strings whose last token is scored. They must match the prompt
# template - a template ending "...I prefer Option " wants ["1", "2"] or ["A", "B"], never
# both - so the template set decides them (see TEMPLATE_SETS in data_collection.py) rather
# than being a separate free-floating flag. None keeps each agent class's own default.
def agent_factory(model_family: str, model_size: str, labels: list = None):
    if model_family == 'qwen':
        return load_qwen2_5_agent(model_size, labels)
    elif model_family == 'gemma':
        return load_gemma3_agent(model_size, labels)
    elif model_family == 'qwen-pt':
        return load_qwen2_5_pt_agent(model_size, labels)
    elif model_family == 'gemma-pt':
        return load_gemma3_pt_agent(model_size, labels)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

if __name__ == "__main__":
    # Example usage
    agent = load_qwen2_5_agent('0.5')
    prompt = "You have two options:\nOption 1: Apple\nOption 2: Orange\nWhich do you prefer?"
    labels = ["Option 1", "Option 2"]
    scores = agent.query(prompt, labels)
    print(scores)