# Pree: 1B Parameter Custom Language Model

Pree is a 1-billion-parameter, decoder-only transformer language model built entirely from scratch in PyTorch. Designed with a focus on first-principles engineering, the architecture implements modern advancements like Rotary Position Embeddings (RoPE) and is heavily optimized for local GPU training.

This repository contains the complete pipeline: from the raw model architecture and custom data loaders to the training loop and inference scripts.

## Architectural Overview

Pree follows the standard causal language modeling objective but incorporates specific design choices to maximize efficiency and context understanding on consumer-grade hardware.

* Training & OptimizationArchitecture: Decoder-only Transformer \- Maximizes generative capabilities; the industry standard for modern foundational LLMs.  
* Parameter Count: \~1 Billion \- Strikes a balance between local trainability from scratch and emergent reasoning capabilities.  
* Positional Encoding: RoPE (Rotary Position Embeddings) \- Injected directly into the attention mechanism to provide superior relative positional information and length extrapolation compared to absolute embeddings.  
* Framework: PyTorch \- Enables low-level control over the computation graph, custom data structures, and memory management.

## 

Training a 1B parameter model locally requires strict VRAM management. The training pipeline is built around **8-bit AdamW** (typically via `bitsandbytes`), which significantly reduces the optimizer state memory footprint—dropping it from 32-bit to 8-bit without degrading training stability or convergence.

* **Precision Setup:** Mixed precision training (FP16/BF16) integration to accelerate matrix multiplications.  
* **Memory Efficiency:** Designed to saturate local GPU compute while preventing OOM (Out of Memory) errors through gradient accumulation and gradient checkpointing.

## Repository Structure

├── model/  
│   ├── transformer.py       (Core decoder blocks, Multi-Head Attention, and MLP)  
│   ├── embeddings.py        (RoPE implementation and token embeddings)  
│   └── config.py            (Hyperparameter configurations)  
├── data/  
│   └── dataloader.py        \# Custom pre-training data tokenization and batching  
├── train.py                 \# Main training loop with 8-bit AdamW setup  
├── generate.py              \# Autoregressive inference script with temperature scaling  
└── requirements.txt         \# Environment dependencies

## Getting Started

### 1\. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies, especially for bitsandbytes and PyTorch CUDA compatibility.  
python \-m venv pree\_env  
source pree\_env/bin/activate  \# On Windows: pree\_env\\Scripts\\activate  
pip install \-r requirements.txt

### 2\. Configuration

Adjust the model hyperparameters in model/config.py based on your available VRAM. The default configuration is tuned for a standard 1B parameter layout.

### 3\. Kickstarting Training

Ensure your pre-training corpus is formatted and accessible to the data loader.  
python train.py \--batch\_size 4 \--grad\_accum\_steps 8 \--learning\_rate 3e-4

### 4\. Inference

Run the generation script using the latest saved model weights.  
python generate.py \--checkpoint checkpoints/pree\_latest.pt \--prompt "The future of artificial intelligence is" \--max\_tokens 128

## Roadmap & Future Iterations

* **Instruction Tuning:** Implementing LoRA/QLoRA adapter workflows to transition Pree from a base model to a chat-aligned assistant.  
* **Quantization:** Exporting the final weights to GGUF format for highly optimized edge inference.  
* **Context Extension:** Scaling the RoPE base frequency to handle larger context windows during fine-tuning.