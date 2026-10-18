from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("=" * 60)
print("           QWEN 2.5 - LOCAL AI CHATBOT")
print("                STREAMING MODE")
print("=" * 60)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded.")

print("\nLoading Qwen2.5 model...")
print("Please wait...\n")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model = model.to("cpu")
model.eval()

print("=" * 60)
print("MODEL LOADED SUCCESSFULLY!")
print("=" * 60)

print("\nCommands:")
print("  /clear    - Clear conversation memory")
print("  /history  - Show conversation history")
print("  /exit     - Exit chatbot")
print("-" * 60)

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Answer questions clearly and accurately."
        )
    }
]


def generate_response(messages):

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05
    }

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )

    thread.start()

    full_response = ""

    for new_text in streamer:

        print(new_text, end="", flush=True)
        full_response += new_text

    thread.join()

    return full_response.strip()


while True:

    user_input = input("\n\nYou: ").strip()

    # Exit
    if user_input.lower() in ["/exit", "exit", "quit"]:

        print("\nAI: Goodbye!")
        break

    # Clear memory
    if user_input.lower() == "/clear":

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Answer questions clearly and accurately."
                )
            }
        ]

        print("\nAI: Conversation memory cleared.")
        continue

    # Show history
    if user_input.lower() == "/history":

        print("\n========== CHAT HISTORY ==========")

        for message in messages:

            if message["role"] == "system":
                continue

            print(
                f'{message["role"].upper()}: '
                f'{message["content"]}'
            )

        print("==================================")
        continue

    # Ignore empty input
    if user_input == "":
        continue

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    print("\nAI: ", end="", flush=True)

    response = generate_response(messages)

    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )