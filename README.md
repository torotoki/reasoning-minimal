# 🧠 reasoning-minimal
Minimal code for making a reasoning model from the base model using **Guided Reward Policy Optimization (GRPO)**.

The model is trained to output

* a private chain-of-thought wrapped in **\<think>...\</think>**
* a final answer wrapped in **\<answer>...\</answer>**

Two custom reward functions enforce **format correctness** and **answer accuracy** simultaneously.

---

## ✨ Highlights
- Lightweight base: **Qwen/Qwen2.5-0.5B-Instruct**
- Training on **GSM8K** grade-school math dataset
- Optional **LoRA** adaptation (uncomment to enable) for low-VRAM training
- Precise math checking with **[math_verify]**
- Plug-and-play multi-reward RL via **TRL’s GRPOTrainer**
- Integrated **Weights & Biases** logging and automatic inference demo

---

## 🚀 Setup & Usage
```bash
pip install torch transformers datasets peft trl math_verify wandb
python train.py
```

## References & Acknowledgements

- [TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl)  
- [Qwen 2.5 models](https://huggingface.co/Qwen)  
- [GSM8K dataset](https://github.com/openai/grade-school-math)  
- [math_verify](https://github.com/huggingface/Math-Verify)
