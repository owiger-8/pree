# Pree LLM: 1B-Parameter Decoder-Only Transformer

Pree is a custom-built, 1.02 billion parameter decoder-only transformer model engineered specifically for code generation tasks. Implemented entirely from scratch in PyTorch, the primary architectural constraint was to enable full end-to-end training and inference on consumer-grade hardware with a strict 8 GB VRAM limit. 

This constraint necessitated the implementation of state-of-the-art memory optimization techniques and efficient transformer variants, closely mirroring the design principles of modern foundational models.

## 1. Global Hyperparameters

| Parameter | Specification | 
| :--- | :--- | 
| **Total Parameters** | ~1.02 Billion | 
| **Context Length** | 2048 tokens | 
| **Vocabulary Size** | 65,536 ($2^{16}$, optimized for Tensor Core alignment) | 
| **Layers ($L$)** | 24 | 
| **Hidden Dimension ($d_{model}$)** | 1536 | 
| **MLP Expansion Dimension** | 6144 ($4 \times d_{model}$) | 

## 2. Architectural Specifications

Pree diverges from the standard Vaswani transformer architecture, adopting several efficiency improvements designed to maximize throughput and minimize memory footprint.

### 2.1 Pre-Normalization (RMSNorm)
Instead of standard Layer Normalization, Pree employs Root Mean Square Normalization (RMSNorm) prior to both the attention and MLP sub-layers. By removing the mean-centering operation, RMSNorm maintains training stability while reducing computational overhead.

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma $$

### 2.2 Grouped-Query Attention (GQA)
To mitigate the memory bandwidth bottleneck caused by the Key-Value (KV) cache during autoregressive decoding, Pree utilizes Grouped-Query Attention. 

* **Head Dimension ($d_h$):** 64
* **Query Heads ($N_q$):** 24 ($24 \times 64 = 1536$)
* **KV Heads ($N_{kv}$):** 6 ($6 \times 64 = 384$)
* **GQA Ratio:** 4:1 (Every 4 Query heads share 1 Key and 1 Value head).

This 4:1 reduction drastically shrinks the KV cache memory allocation during generation, allowing for longer sequence lengths on limited VRAM.

### 2.3 Rotary Position Embeddings (RoPE)
Absolute positional embeddings are discarded in favor of Rotary Position Embeddings (RoPE). RoPE encodes positional information by rotating the affine-transformed Query and Key representations in the complex plane at each layer. 

This provides a mathematically sound relative position decay, enhancing the model's ability to extrapolate positional relationships across lengthy code contexts without requiring an explicit relative position bias matrix.

### 2.4 Feed-Forward Network (MLP)
The MLP block utilizes a $4\times$ expansion factor mapping the 1536-dimensional input to a 6144-dimensional hidden state. The traditional ReLU activation is replaced with the SiLU (Swish) activation function:

$$ \text{SiLU}(x) = x \cdot \sigma(x) $$

SiLU provides a smooth, non-monotonic gradient landscape, which empirical evidence suggests helps prevent vanishing gradients and improves convergence in deep networks.

## 3. Training & Hardware Optimizations

Training a 1B parameter model typically requires significant VRAM, as optimizer states, gradients, and forward activations easily exceed an 8 GB threshold. Pree implements the following techniques to fit within this strict memory envelope:

### 3.1 Gradient Checkpointing
To prevent out-of-memory (OOM) exceptions during the backward pass, Pree utilizes gradient checkpointing. This trades computational cycles for memory by discarding intermediate forward pass activations and recomputing them during backpropagation. This drastically reduces the memory required to store the computation graph, at the cost of an approximate 20-25% increase in training iteration time.

### 3.2 8-bit AdamW Optimization
Standard AdamW stores two 32-bit floating-point momentum states (first and second moments) per parameter. For a 1B parameter model, these states alone would consume ~8 GB of VRAM. By quantizing these optimizer states to 8-bit precision, the memory footprint of the optimizer is reduced by 75%. This leaves sufficient VRAM for the model weights, batched sequences, and necessary gradients.

### 3.3 Vocabulary Tensor Alignment
The vocabulary size is explicitly set to 65,536 ($2^{16}$). By ensuring all tensor dimensions are multiples of 8 (and ideally 64), the architecture guarantees optimal alignment for NVIDIA Tensor Cores, maximizing matrix multiplication throughput during the final projection and softmax operations.
