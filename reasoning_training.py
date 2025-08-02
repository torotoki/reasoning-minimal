#!/usr/bin/env python
# coding: utf-8

# ========================================
#             Imports & Configs
# ========================================

import re
import time
from pprint import pprint
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from math_verify import LatexExtractionConfig, parse, verify


# ========================================
#         Load & Preprocess Dataset
# ========================================

dataset_id = "openai/gsm8k"
train_dataset, test_dataset = load_dataset(dataset_id, 'main', split=['train[:5%]', 'test[:5%]'])

print(train_dataset)
pprint(train_dataset[0])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
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


train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

pprint(train_dataset[1])


# ========================================
#              Load Model
# ========================================

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)


# ========================================
#         Apply LoRA Adaptation
# ========================================

# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj"],
# )

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# ========================================
#  Sampling before
# ========================================


# ========================================
#             Reward Functions
# ========================================

def format_reward(completions, **kwargs):
    """Check if the completion matches required XML-like format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    contents = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, c) else 0.0 for c in contents]


def accuracy_reward_hf(completions, **kwargs):
    """Use math_verify to check correctness of answer extraction."""
    solutions = kwargs["answer"]
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(contents, solutions):
        gold = parse(solution, "first_match", [LatexExtractionConfig()])
        pred = parse(content, "first_match", [LatexExtractionConfig()])
        try:
            rewards.append(float(verify(pred, gold)) if gold else 1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions, **kwargs) -> List[float]:
    """Debugging version of accuracy_reward."""
    solutions = kwargs["answer"]
    predictions = [completion[0]["content"] for completion in completions]
    print(len(predictions), len(solutions))
    print("question=")
    pprint(kwargs["question"])
    print("=" * 80)
    print("predictions=")
    pprint(predictions)
    print("=" * 80)
    print("solutions=")
    pprint(solutions)
    print("=" * 80)
    return [0]


# ========================================
#         Training Configuration
# ========================================

output_dir = Path("outputs/Qwen2-0.5B-GRPO-test")

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    remove_unused_columns=False,
    gradient_accumulation_steps=16,  # default 16
    num_train_epochs=100,
    per_device_train_batch_size=8,  # default 8
    bf16=True,
    max_completion_length=64,
    num_generations=4,
    max_prompt_length=128,
    report_to=["wandb"],
    logging_steps=1,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=10,
)


# ========================================
#                 Training
# ========================================

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward_hf],
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
trainer.save_model(training_args.output_dir)


# ========================================
#        Model Evaluation Utilities
# ========================================

def extract_tag(text: str, tag: str) -> str | None:
    pattern = fr".*?<{tag}>(.*?)</{tag}>.*?"
    match = re.match(pattern, text)
    return match.group(1) if match else None


def generate_with_reasoning(prompt, model, tokenizer):
    full_prompt = " ".join(entry["content"] for entry in prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=500)
    end_time = time.time()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    duration = end_time - start_time
    num_input_tokens = inputs["input_ids"].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, duration, num_generated_tokens

def print_inference_example(model, tokenizer):
    prompt = test_dataset["prompt"][0]
    generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(prompt, model, tokenizer)

    print(f"Inference time: {inference_duration:.2f} seconds")
    print(f"Generated tokens: {num_generated_tokens}")
    prompt_text = " ".join(entry["content"] for entry in prompt)
    print("="*80)
    print(prompt_text)
    print("="*80)
    response_text = generated_text[len(prompt_text):].strip()
    print(response_text)
    print("="*80)

# ========================================
#            Inference Example
# ========================================

trained_model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype="auto",
    device_map="auto",
)
trained_tokenizer = AutoTokenizer.from_pretrained(output_dir)

print("Trained:")
print_inference_example(trained_model, trained_tokenizer)

untrained_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
untrained_tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Before Trained:")
print_inference_example(untrained_model, untrained_tokenizer)

# ========================================
#       (Optional) Evaluation Script
# ========================================

def extract_answer_numinamath_tir(text: str) -> str | None:
    pattern = r".*?```output(.*?)```.*?"
    match = re.match(pattern, text)
    return match.group(1) if match else None

# Example usage:
# print(test_dataset['messages'][0])
# extract_answer_numinamath_tir(test_dataset['solution'][0])
