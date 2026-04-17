# -RNN-vs-GRU-vs-LSTM-for-Text-Generation

# RNN vs GRU vs LSTM for Text Generation

## Overview

This project compares three recurrent neural network architectures on a next-word prediction task. The goal was straightforward: train each model on the same text data, then see how their memory mechanisms affect what they learn.

The three models tested:

- Simple RNN
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)

---

## Dataset

**Book:** *Alice's Adventures in Wonderland* by Lewis Carroll (public domain)

The full text was used for training. It's a good fit for this kind of experiment — it has natural sentence rhythm, varied structure, and enough length to give the models something meaningful to learn from.

---

## Preprocessing

The raw text went through a few cleanup steps before training:

- Lowercased everything
- Removed punctuation and special characters
- Stripped extra whitespace
- Tokenized using Keras `Tokenizer` (each unique word gets a numeric index)

---

## Sequence Construction

The text was converted into input-output pairs for next-word prediction. Each sequence grows one word at a time:

```
Input: "alice"                        → Target: "was"
Input: "alice was"                    → Target: "beginning"
Input: "alice was beginning"          → Target: "to"
```

All sequences were padded to a fixed length (pre-padding), with a **maximum length of 20 tokens**. Longer sequences were truncated. This kept training stable and stopped the models from drowning in context they couldn't use anyway.

---

## Model Architectures

All three models share the same overall structure. The only thing that changes is the recurrent layer in the middle.

**Shared layers:**
- **Embedding layer** — converts word indices to dense vectors
- **Recurrent layer** — either SimpleRNN, GRU, or LSTM
- **Dense layer** — softmax output over the full vocabulary

### Simple RNN

Uses a basic `SimpleRNN` layer. No gating, no memory control. Works fine for short dependencies, falls apart on anything longer.

### GRU

Uses a `GRU` layer. Two gates (reset and update) give it real control over what to keep and what to discard. More capable than plain RNN, with fewer parameters than LSTM.

### LSTM

Uses an `LSTM` layer. Three gates (forget, input, output) plus a separate cell state. The most expressive of the three — built specifically to handle long-range dependencies.

---

## Training Setup

| Setting | Value |
|---|---|
| Loss function | `categorical_crossentropy` |
| Optimizer | `Adam` |
| Epochs | 20–30 |

All models were trained on identical data under identical conditions.

---

## Results

| Model | Accuracy |
|---|---|
| Simple RNN | 77.89% |
| GRU | 82.96% |
| LSTM | **97.57%** |

The gap between Simple RNN and LSTM isn't surprising — it's the whole reason gated architectures exist. What's worth noting is how much ground GRU gains over plain RNN with relatively modest extra complexity. And then LSTM takes another big step beyond that.

Simple RNN loses the thread of a sentence quickly. GRU does a much better job holding context. LSTM, with its dedicated cell state and three-gate system, handles long-range dependencies better than either of the other two.

---

## Evaluation

Beyond accuracy numbers, the models were tested qualitatively — random seed sequences from the dataset, with predictions compared against the actual next words. Generated text was read for coherence and grammatical plausibility.

---

## Conclusion

LSTM won, and by a wide margin. That said, GRU's result deserves credit — it closes most of the gap between basic RNN and LSTM while staying computationally lighter. Depending on the task, GRU is often the better tradeoff.

The broader takeaway: memory architecture matters a lot in sequence modeling. Gating isn't just an implementation detail — it's what makes these models actually useful for language.

---

## Future Work

- Add temperature sampling for more varied text generation
- Benchmark against a Transformer-based model
- Train on a larger, more diverse corpus

---

## Libraries Used

- Python :
- TensorFlow / Keras
- NumPy
