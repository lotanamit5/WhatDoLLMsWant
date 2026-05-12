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
    def from_model(cls, model, tokenizer):
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
            cache_dir=cache_dir,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True
        )
        model.eval() # Set to evaluation mode
        
        return model, tokenizer

class ItayHFAgentV0(HFAgent):
    """ The original model from Itay's paper """
    def get_chat_format_one_side(self, text, role):
        return {
            "role": role,
            "content": text,
        }

    def convert_to_chat_format(self, text, few_shots_texts=None):
        """
        Uses the apply_chat_template function in the tokenizer of the predictor to convert the text to chat format
        """
        SYSTEM_MESSAGE = "You are a helpful assistant. Answer shortly with only your choice with no explanation.\n\n"
        # if there are no few shots, just add the system message to the text
        if few_shots_texts is None:
            messages = [
                self.get_chat_format_one_side(SYSTEM_MESSAGE + text, "user"),
            ]
        # if there are few shots, add the system message to the first shot

        else:
            # the first shot needs the system message before the text
            messages = [
                self.get_chat_format_one_side(
                    SYSTEM_MESSAGE + few_shots_texts[0]["question"], "user"
                ),
                self.get_chat_format_one_side(
                    few_shots_texts[0]["answer"], "assistant"
                ),
            ]

            # after that, add all the other shot in the user and assistent format
            for shot in few_shots_texts[1:]:
                messages.extend(
                    [
                        self.get_chat_format_one_side(shot["question"], "user"),
                        self.get_chat_format_one_side(shot["answer"], "assistant"),
                    ]
                )

            # finaly, add the actual text to the user
            messages.append(self.get_chat_format_one_side(text, "user"))

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # , return_tensors="pt"
        )

        return prompt
    
    def query(self, prompt, labels):
        # concat labels to the corrposnded input text
        input_with_answers = [i + label for label in labels for i in prompt]
        print(f"{input_with_answers=}")

        # get labels tokens ids
        labels_tokens = self.tokenizer(labels, add_special_tokens=False)["input_ids"]
        print(f"{labels_tokens=}")

        # get the last token id of each label
        labels_tokens = [label[-1] for label in labels_tokens]
        print(f"{labels_tokens=}")

        # Ensure pad token exists before padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get encodings for each input text using direct call instead of batch_encode_plus
        input_enc = self.tokenizer(
            input_with_answers,
            return_tensors="pt",
            padding="longest",
        )
        print(f"{input_enc=}")

        for k, v in input_enc.items():
            input_enc[k] = v.to(self.model.device)

        # Get model output logits
        model_output = self.model(**input_enc)

        # Compute the log probabilities associated with each of the labels
        labels_log_probs = F.log_softmax(model_output.logits, dim=-1)
        print(f"{model_output.logits=}")
        print(f"{labels_log_probs=}")

        # Get the ids of the token before the last token before padding (to see the probablity of the last token given the one before the last token)
        before_padding_ids = (
            input_enc["input_ids"].ne(self.tokenizer.pad_token_id).sum(-1) - 2
        )

        # Collect labels scores from the -2 token in labels_log_probs (the one that predict the last token)
        # and collect for each line the id in labels_tokens
        labels_scores = labels_log_probs[:, before_padding_ids, labels_tokens]
        print(f"{labels_scores=}")

        # Need just the diagonal of the matrix, as it the prob of the label for each line
        labels_scores = torch.diag(labels_scores)

        return labels_scores

class ItayHFAgentV1(ItayHFAgentV0):
    "moves system prompt to correct place"
    def convert_to_chat_format(self, text, few_shots_texts=None):
        """
        Uses the apply_chat_template function in the tokenizer of the predictor to convert the text to chat format
        """
        SYSTEM_MESSAGE = "You are a helpful assistant. Answer shortly with only your choice with no explanation.\n\n"
        messages = [self.get_chat_format_one_side(SYSTEM_MESSAGE, "system")]
        # if there are no few shots, just add the system message to the text
        if few_shots_texts is None:
            messages.append(self.get_chat_format_one_side(text, "user"))
        # if there are few shots, add the system message to the first shot

        else:
            # the first shot needs the system message before the text
            messages = [
                self.get_chat_format_one_side(
                    few_shots_texts[0]["question"], "user"
                ),
                self.get_chat_format_one_side(
                    few_shots_texts[0]["answer"], "assistant"
                ),
            ]

            # after that, add all the other shot in the user and assistent format
            for shot in few_shots_texts[1:]:
                messages.extend(
                    [
                        self.get_chat_format_one_side(shot["question"], "user"),
                        self.get_chat_format_one_side(shot["answer"], "assistant"),
                    ]
                )

            # finaly, add the actual text to the user
            messages.append(self.get_chat_format_one_side(text, "user"))

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # , return_tensors="pt"
        )

        return prompt
    
class LotanHFAgentV0(HFAgent):
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
        print(f"query: prompt={prompt}, labels={labels}.")
        prompt = self._convert_to_chat_template(prompt)
        print(f"chat template={prompt}")
        scores = []
        for label in labels:
            print(f"{label=}")
            # 1. Construct the full sequence
            # Note: We must be careful with the space. 
            # If the prompt ends with ':', we typically want " " + label.
            full_text = prompt + " " + label
            
            # 2. Tokenize inputs
            input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.model.device)
            print(f"{input_ids=}")
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            print(f"{prompt_ids=}")
            
            # 3. Find where the label starts
            # (Simple heuristic: label starts after the prompt length)
            # Warning: This can be tricky if tokenization boundaries shift. 
            # A safer way is to check the length diff.
            prompt_len = prompt_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            print(f"{logits=}")

            # 4. Calculate Loss (Negative Log Likelihood) for the label part only
            # Shift logits so we predict the *next* token
            # We want probabilities for input_ids[prompt_len:]
            
            # Slice logits from [prompt_len-1 : -1] -> These predict [prompt_len : end]
            label_logits = logits[0, prompt_len-1 : -1, :] 
            print(f"{label_logits=}")
            label_ids = input_ids[0, prompt_len:]
            print(f"{label_ids=}")
            
            # Cross Entropy creates an average, we usually want SUM for total probability
            # or MEAN for perplexity. 
            # For ranking "Red" vs "Blue", SUM is safer if lengths differ significantly,
            # but MEAN (Perplexity) is standard if lengths are roughly equal.
            loss = F.cross_entropy(label_logits, label_ids, reduction='sum')
            
            # We want a "Score" where higher is better. Loss is lower-is-better.
            # So return negative loss (Log Probability).
            scores.append(-loss.item())

        return scores

class LotanHFAgentV1(LotanHFAgentV0):
    "Seperated encoding of labels and "
    def query(self, prompt: str, labels: List[str]) -> List[float]:
        print(f"query: prompt={prompt}, labels={labels}.")
        prompt = self._convert_to_chat_template(prompt)
        print(f"{prompt=}")
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        print(f"{prompt_ids=}")
        scores = []
        for label in labels:
            print(f"{label=}")
            # 2. Tokenize inputs
            label_ids = self.tokenizer.encode(label, return_tensors="pt").to(self.model.device)
            print(f"{label_ids=}")
            input_ids = torch.cat([prompt_ids, label_ids], dim=1).to(self.model.device)
            print(f"{input_ids=}")
            
            # 3. Find where the label starts
            # (Simple heuristic: label starts after the prompt length)
            # Warning: This can be tricky if tokenization boundaries shift. 
            # A safer way is to check the length diff.
            prompt_len = prompt_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            print(f"{logits=}")

            # 4. Calculate Loss (Negative Log Likelihood) for the label part only
            # Shift logits so we predict the *next* token
            # We want probabilities for input_ids[prompt_len:]
            
            # Slice logits from [prompt_len-1 : -1] -> These predict [prompt_len : end]
            label_logits = logits[0, prompt_len-1 : -1, :] 
            print(f"{label_logits=}")
            label_ids = input_ids[0, prompt_len:]
            print(f"{label_ids=}")
            
            # Cross Entropy creates an average, we usually want SUM for total probability
            # or MEAN for perplexity. 
            # For ranking "Red" vs "Blue", SUM is safer if lengths differ significantly,
            # but MEAN (Perplexity) is standard if lengths are roughly equal.
            loss = F.cross_entropy(label_logits, label_ids, reduction='sum')
            
            # We want a "Score" where higher is better. Loss is lower-is-better.
            # So return negative loss (Log Probability).
            scores.append(-loss.item())

        return scores

class LotanHFAgentV2(LotanHFAgentV1):
    "added prefill"
    def query(self, prompt: str, labels: List[str], prefill: str = "") -> List[float]:
        print(f"query: prompt={prompt}, labels={labels}, prefill='{prefill}'.")
        
        # Apply the chat template (adds <|im_start|>assistant\n at the end)
        prompt = self._convert_to_chat_template(prompt)
        
        # Append the prefill directly to the assistant's generation start
        if prefill:
            prompt += prefill
            
        print(f"prompt_with_prefill={prompt}")
        
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        print(f"{prompt_ids=}")
        
        scores = []
        for label in labels:
            print(f"{label=}")
            
            # 2. Tokenize inputs (ensure label assumes the prefill just happened)
            # Example: if prefill="Option", label should be " A"
            label_ids = self.tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").to(self.model.device)
            print(f"{label_ids=}")
            
            input_ids = torch.cat([prompt_ids, label_ids], dim=1).to(self.model.device)
            print(f"{input_ids=}")
            
            # 3. Find where the label starts
            prompt_len = prompt_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            
            # 4. Calculate Loss (Negative Log Likelihood) for the label part only
            # Slice logits from [prompt_len-1 : -1] -> These predict [prompt_len : end]
            label_logits = logits[0, prompt_len-1 : -1, :] 
            label_ids = input_ids[0, prompt_len:]
            
            loss = F.cross_entropy(label_logits, label_ids, reduction='sum')
            scores.append(-loss.item())

        return scores

class LotanHFAgentV3(LotanHFAgentV2):
    def query(self, prompt: str, labels: List[str], prefill: str = "", normalize: str = None) -> List[float]:
        # 1. Generate the standard scores
        formatted_prompt = self._convert_to_chat_template(prompt) + prefill
        raw_scores = self._get_batched_logprobs(formatted_prompt, labels)
        
        if not normalize:
            return raw_scores

        # 2. Generate the "prior" scores (PMI Normalization)
        # We use an empty string for the user message, preserving the chat structure and prefill
        null_formatted_prompt = self._convert_to_chat_template(normalize) + prefill
        prior_scores = self._get_batched_logprobs(null_formatted_prompt, labels)
        
        # 3. Subtract prior from raw (PMI: log(P(label|prompt)) - log(P(label|null)))
        normalized_scores = [raw - prior for raw, prior in zip(raw_scores, prior_scores)]
        return normalized_scores

    def _get_batched_logprobs(self, formatted_prompt: str, labels: List[str]) -> List[float]:
        # 1. Encode prompt
        prompt_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=True).squeeze(0)
        prompt_len = len(prompt_ids)

        # 2. Encode labels separately
        label_ids_list = [
            self.tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").squeeze(0) 
            for label in labels
        ]
        
        # 3. Construct padded batch manually for absolute boundary safety
        batch_size = len(labels)
        max_label_len = max(len(l_ids) for l_ids in label_ids_list)
        max_seq_len = prompt_len + max_label_len

        input_ids = torch.full((batch_size, max_seq_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=self.model.device)

        label_lengths = []
        for i, l_ids in enumerate(label_ids_list):
            seq_len = prompt_len + len(l_ids)
            input_ids[i, :seq_len] = torch.cat([prompt_ids, l_ids.to(self.model.device)])
            attention_mask[i, :seq_len] = 1
            label_lengths.append(len(l_ids))

        # 4. Forward pass (Batched!)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        log_probs = F.log_softmax(logits, dim=-1)

        # 5. Extract scores using sum reduction
        scores = []
        for i, l_len in enumerate(label_lengths):
            # The logits predicting the label start at prompt_len - 1
            start_idx = prompt_len - 1
            end_idx = start_idx + l_len
            
            label_log_probs = log_probs[i, start_idx:end_idx, :]
            target_ids = input_ids[i, prompt_len : prompt_len + l_len]
            
            # Extract the probabilities of the actual target tokens
            token_scores = label_log_probs[torch.arange(l_len), target_ids]
            
            # reduction='sum' equivalent
            scores.append(token_scores.sum().item())

        return scores

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


def load_qwen2_5_agent(model_size, instructed=False):
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