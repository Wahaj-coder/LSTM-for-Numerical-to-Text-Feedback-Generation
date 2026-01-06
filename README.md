# 🏏 LSTM for Numerical-to-Text Feedback Generation

## 1. Project Objective

The objective of this project is to generate **human-like cricket shot feedback paragraphs**
from **numerical shot data and shot type** using an **Encoder–Decoder LSTM architecture**.

This project is intentionally built with **manual text processing** (no default Keras tokenizer)
to clearly understand how words, sequences, padding, and LSTM states work internally.

---

## 2. What Problem This Solves

Given inputs like:
- confidence
- bat angle
- hit quality
- stance stability
- shot accuracy
- shot type (categorical)

The model generates:
> A meaningful feedback paragraph describing the shot.

---

## 3. High-Level Architecture

This is a **Sequence-to-Sequence (Seq2Seq)** model.

```
Numerical Shot Features + Shot Type
                ↓
             Encoder
          (Dense Layers)
                ↓
        LSTM Initial States
            (h, c)
                ↓
           Decoder LSTM
                ↓
        Softmax Word Output
                ↓
        Feedback Paragraph
```

---

## 4. Encoder Input (Numerical Data)

The encoder takes:
- Scaled numerical values (MinMaxScaler)
- One-hot encoded shot type

These are concatenated into a single vector:

```
X_encoder → shape (samples, features)
```

The encoder **does not see text**.

---

## 5. Manual Text Processing (Core Learning Part)

### 5.1 Text Cleaning
Each feedback paragraph is:
- Lowercased
- Special characters removed
- Wrapped with tokens:

```
<start> sentence words <end>
```

---

### 5.2 Manual Tokenization

No Keras tokenizer is used.

Steps:
1. Split text using `.split(" ")`
2. Build vocabulary manually

```
vocab = ["<pad>"]
```

`<pad>` is **explicitly placed at index 0**.

---

## 6. Vocabulary Mapping

Two dictionaries are created:

```
word2idx[word] = index
idx2word[index] = word
```

The model only understands **numbers**, not words.

---

## 7. Converting Sentences to Sequences

Example:

```
<start> good balance shot <end>
```

Becomes:

```
[1, 23, 45, 78, 2]
```

This creates a **2D list of sequences**.

---

## 8. Padding Sequences

LSTMs require equal-length sequences.

Padding is done manually at the **end**:

```
seq + [0] * (max_len - len(seq))
```

- `0` = `<pad>`
- Shape becomes:

```
(samples, max_len)
```

---

## 9. Decoder Input vs Decoder Target (Key Concept)

Sentence:

```
[1, 23, 45, 2]
```

### Decoder Input
```
[1, 23, 45]
```

### Decoder Target
```
[23, 45, 2]
```

This teaches the model:
> Predict the **next word** given previous words

This is called **teacher forcing**.

---

## 10. One-Hot Encoding Targets

Decoder targets are converted to one-hot vectors:

```
(samples, sequence_length, vocab_size)
```

Required for:
- Softmax output
- Categorical crossentropy loss

---

## 11. Encoder Architecture (Numerical → Memory)

```
Dense (ReLU) → Feature extraction
Dense (tanh) → state_h
Dense (tanh) → state_c
```

- ReLU: learns patterns from numbers
- tanh: bounds values for LSTM memory
- `state_h` and `state_c` initialize the decoder

`x` is **not directly used** by the LSTM.

---

## 12. Decoder Architecture (Text Generation)

### Decoder Input
- Sequence of word indices
- Shape defined using `Input()`

### Embedding Layer
- Converts word IDs → dense vectors
- `mask_zero=True` ignores padding

### Decoder LSTM
- Generates one output per word
- Uses encoder states as initial memory

### Output Layer
- Softmax over vocabulary
- Predicts next word probability

---

## 13. Training Flow

```
model.fit(
  [X_encoder, decoder_input],
  decoder_target
)
```

- Encoder processes numeric data
- Decoder learns next-word prediction

---

## 14. Inference Flow

1. Start with `<start>` token
2. Predict next word
3. Feed predicted word back
4. Stop at `<end>`

This generates **new feedback**, not copied text.

---

## 15. Key Highlights

✔ Manual tokenization  
✔ Manual padding  
✔ Explicit `<start>`, `<end>`, `<pad>` handling  
✔ Clear encoder–decoder separation  
✔ Numeric data drives text generation  

---

## 16. Final Summary

This project explains **how sequence-to-sequence text generation works internally**:

- Numbers → features
- Features → memory states
- Memory + previous words → next word

Designed for **learning, clarity, and full control** over the text pipeline.
