"""
Module for training reasoning models.
"""
#!/usr/bin/env python
# coding: utf-8

# ========================================
#             Imports
# ========================================

import argparse
import re
from pprint import pprint
from pathlib import Path
from typing import List

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from math_verify import LatexExtractionConfig, parse, verify
from format_utils import (
    extract_hashed_answer,
    extract_tag,
    print_inference_example,
    make_conversation,
)

# ========================================
#             Reward Functions
# ========================================


def format_reward(completions, **kwargs):
    """Check if the completion matches required format."""
    rewards = []
    think_pattern = r"^<think>.*?</think>"
    answer_pattern = r"<answer>.*?</answer>$"
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    contents = [completion[0]["content"] for completion in completions]
    for content in contents:
        score = 0.0
        score += 0.5 if re.match(think_pattern, content) else 0.0
        score += 0.5 if re.search(answer_pattern, content) else 0.0
        score += 1.0 if re.match(pattern, content) else 0.0
        rewards.append(score)
    return rewards

def accuracy_reward_cot(completions, **kwargs):
    """Use math_verify to check correctness of answer extraction."""
    solutions = kwargs["answer"]
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(contents, solutions):
        reward = 0.0
        cot_pred_raw = extract_tag(content, "think")
        if not cot_pred_raw:
            rewards.append(reward)
            continue
        cot_pred = list(filter(lambda x: x.strip(),
                               cot_pred_raw.split('\n')))
        cot_gold = solution.split('####')[0].split('\n')

        # Rewards for the length of Chaing-of-Thought
        if len(cot_pred) == len(cot_gold):
            reward += 1.0
        elif len(cot_pred) < len(cot_gold):
            reward += len(cot_pred) / len(cot_gold)
        elif len(cot_pred) > len(cot_gold):
            reward += 1.0 - len(cot_pred) / len(cot_gold)

        for thought_pred, thought_gold in zip(cot_pred, cot_gold):
            gold = parse(thought_gold, "first_match", [LatexExtractionConfig()])
            pred = parse(thought_pred, "first_match", [LatexExtractionConfig()])

            try:
                reward += float(verify(pred, gold)) if gold else 1.0
            except Exception:
                reward += 0.0
            
            if pred:
                reward += 0.5

        rewards.append(reward)
    return rewards

def accuracy_reward_answer(completions, **kwargs):
    solutions = kwargs["answer"]
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(contents, solutions):
        reward = 0.0
        pred = extract_tag(content, "answer")
        gold = solution.split('####')[1].strip()
        reward += 2.0 if verify(pred, gold) else 1.0
        rewards.append(reward)
    verbose = True
    if verbose:
        print(contents[0])
        print("=" * 80)
    return rewards

def accuracy_reward(completions, **kwargs) -> List[float]:
    """Debugging version of accuracy_reward."""
    solutions = kwargs["answer"]
    predictions = [completion[0]["content"]
                   for completion in completions]
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


def main():
    # ========================================
    #         Argument Parser
    # ========================================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path-or-dir",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )
    args = parser.parse_args()

    # ========================================
    #         Load & Preprocess Dataset
    # ========================================

    dataset_id = "openai/gsm8k"
    train_dataset, test_dataset = \
        load_dataset(dataset_id, 'main', split=['train[:5%]', 'test[:5%]'])

    print(train_dataset)
    pprint(train_dataset[0])

    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation)

    pprint(train_dataset[1])

    # ========================================
    #              Load Model
    # ========================================

    model_id = args.model_path_or_dir
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
    #         Training Configuration
    # ========================================

    output_dir = Path(f"outputs/{model_id}-GRPO")

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        remove_unused_columns=False,
        gradient_accumulation_steps=16,  # default 16
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,  # default 8
        bf16=True,
        max_completion_length=64,
        num_generations=4,
        max_prompt_length=128,
        report_to=["wandb"],  # ["wandb"],
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
        reward_funcs=[
            format_reward,
            accuracy_reward_answer,
            accuracy_reward_cot
        ],
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

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
    print_inference_example(trained_model, trained_tokenizer, test_dataset)

    del trained_model
    del trained_tokenizer

    untrained_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    untrained_tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Before Trained:")
    print_inference_example(untrained_model, untrained_tokenizer, test_dataset)


if __name__ == '__main__':
    main()
