#!/usr/bin/env python
# coding: utf-8

import argparse
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from format_utils import generate_with_reasoning, extract_tag, make_conversation


def evaluation(model, tokenizer, test_dataset):
    correct = 0
    no_answer = 0
    for example in tqdm(test_dataset):
        prompt = example["prompt"]
        gold = example["direct_answer"]
        generated_text, elapsed_time, num_generated_tokens = generate_with_reasoning(
            prompt, model, tokenizer)
        pred = extract_tag(generated_text, "answer")
        if not pred:
            no_answer += 1
            continue

        if pred.strip() == gold.strip():
            correct += 1

    print(f"{correct=}, {len(test_dataset)=}, {no_answer=}")
    return correct / len(test_dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a language model on GSM8K subset.")
    parser.add_argument(
        "--model-path-or-dir",
        type=str,
        default="./outputs/Qwen2-0.5B-GRPO-test",
    )
    args = parser.parse_args()

    dataset_id = "openai/gsm8k"
    test_dataset = load_dataset(dataset_id, 'main', split=['test[:10%]'])[0]
    test_dataset = test_dataset.map(make_conversation)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_dir,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_dir)
    evaluation(model, tokenizer, test_dataset)


if __name__ == "__main__":
    main()
