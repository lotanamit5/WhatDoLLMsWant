import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"

print(f"Attempting to load: {model_id}")
print(f"Cuda: {torch.cuda.is_available()}")
try:
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load Model with automatic device mapping (uses all available GPUs)
    # torch_dtype=torch.bfloat16 is recommended for newer GPUs (Ampere+), otherwise use float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.bfloat16
    )

    print("\nSUCCESS: Model loaded successfully!")
    print(f"Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    print(f"Device map: {model.hf_device_map}")

    # Test Inference
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud."},
        {"role": "user", "content": "Hello, can you hear me?"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=50)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n--- Test Generation ---")
    print(response)

except torch.cuda.OutOfMemoryError:
    print("\nERROR: Out of Memory (OOM). The model is too large for your VRAM.")
    print("Try loading in 8-bit or 4-bit precision using bitsandbytes.")
except Exception as e:
    print(f"\nERROR: Failed to load model. Reason: {e}")