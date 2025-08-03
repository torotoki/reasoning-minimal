import time
import re
import torch


# ========================================
#        Model Evaluation Utilities
# ========================================

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process "
    "in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively."
)


def make_conversation(example):
    direct_answer = ""
    for line in example["answer"].split('\n'):
        if line.startswith("#### "):
            direct_answer = line.replace("#### ", "")
            break
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "direct_answer": direct_answer
    }


def extract_tag(text: str, tag: str) -> str | None:
    pattern = fr".*?<{tag}>(.*?)</{tag}>.*?"
    match = re.match(pattern, text)
    return match.group(1) if match else None


def extract_hashed_answer(text: str) -> str | None:
    return text.split("####")[1].strip()


def generate_with_reasoning(prompt, model, tokenizer):
    full_prompt = " ".join(entry["content"] for entry in prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=500)
    end_time = time.time()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = generated_text[len(full_prompt):]
    duration = end_time - start_time
    num_input_tokens = inputs["input_ids"].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, duration, num_generated_tokens


def print_inference_example(model, tokenizer, dataset):
    prompt = dataset["prompt"][0]
    generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(
        prompt, model, tokenizer)

    print(f"Inference time: {inference_duration:.2f} seconds")
    print(f"Generated tokens: {num_generated_tokens}")
    prompt_text = " ".join(entry["content"] for entry in prompt)
    print("=" * 80)
    print(prompt_text)
    print("=" * 80)
    response_text = generated_text[len(prompt_text):].strip()
    print(response_text)
    print("=" * 80)
