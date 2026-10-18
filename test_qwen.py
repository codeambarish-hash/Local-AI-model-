from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading Qwen model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.to("cpu")
model.eval()

print("\nQwen model loaded successfully!")

messages = [
    {
        "role": "user",
        "content": "Hello! Introduce yourself in one sentence."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

print("Generating response...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("Qwen:", response)
