# 📘 Deep_Learning_Basics

Deep_Learning_Basics is a hands-on educational repository focused on understanding deep learning from first principles. The goal is to implement core concepts from scratch — without hiding mechanics behind high-level libraries — so that the mathematical and algorithmic foundations become fully clear.

---

## 🧠 Project Philosophy

> “If you can build it from scratch, you understand it.”

This repository emphasizes:
- Manual implementation of neural networks
- Explicit forward and backward passes
- Understanding optimization and gradient flow
- Language modeling from first principles
- Tokenization internals (byte-level BPE, regex tokenizers, special tokens)

It is designed for learners who want depth, not shortcuts.

---

## 📂 Repository Structure

Each lecture is organized into its own folder:

```
Deep_Learning_Basics/
│
├── Lecture-1-Backpropagation/
├── Lecture-2-Language-Modeling/
├── Lecture-3-MLP/
├── Lecture-4-RNN/
├── Lecture-5-LSTM-GRU/
├── Lecture-7-Tokenization/
└── ...
```

Each lecture typically includes:
- Jupyter notebooks with detailed explanations
- Step-by-step code implementations
- Mathematical reasoning alongside implementation
- Experiments and observations

---

## 🚀 Topics Covered

### 🔹 Neural Network Fundamentals
- Computation graphs
- Automatic differentiation
- Backpropagation mechanics
- Manual gradient calculations

### 🔹 Language Modeling
- Next-token prediction
- Character-level models
- Probability distributions over vocabularies
- Softmax interpretation

### 🔹 Multi-Layer Perceptrons (MLP)
- Embedding lookups
- Forward pass mechanics
- BatchNorm
- Initialization strategies
- Optimization behavior

### 🔹 Recurrent Networks
- Vanilla RNN
- LSTM
- GRU
- Temporal dependencies

### 🔹 Tokenization
- UTF-8 byte encoding
- Byte Pair Encoding (BPE)
- Merge table construction
- Regex-constrained tokenization (GPT-style)
- Special tokens handling
- Deterministic encode/decode pipelines

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/yashraj01-se/Deep_Learning_Basics.git
cd Deep_Learning_Basics
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
```

Install required packages:

```bash
pip install -r requirements.txt
```

Run notebooks:

```bash
jupyter notebook
```

---

## 🎯 Learning Objectives

By completing this repository, you will:

- Understand how neural networks actually compute gradients
- Implement deep learning components without abstraction layers
- Understand tokenization as compression + grammar induction
- Analyze how vocabulary size affects modeling
- Connect discrete tokenization to continuous embedding space
- Gain architectural intuition for modern LLM systems

---

## 📌 Who This Is For

- Students learning deep learning deeply
- Engineers who want system-level understanding
- Researchers interested in internals
- Anyone preparing for advanced ML work

---

## 📜 License

This project is open-source. See the LICENSE file for details.

---

If you’re serious about mastering deep learning from the ground up, this repository is designed to take you there.
