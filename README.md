# AI-Driven Prompt Guardrail Experiments

Implementation and experimental validation framework for the IEEE paper:

**"How AI Can Help Guardrail Prompt Engineering: Algorithms and Experimental Validation"**

This repository provides a complete experimental pipeline for evaluating AI-driven guardrail mechanisms across adversarial prompt detection, bias assessment, adaptive learning, multimodal threat detection, scalability, and output safety classification.

---

## Overview

This project implements six experimental modules:

1. **Adversarial Prompt Detection**  
   Detects jailbreak and instruction override attempts using a BERT-based semantic risk model.

2. **Bias Detection & Mitigation**  
   Identifies biased or discriminatory prompt patterns in user prompts.

3. **Adaptive Guardrail Learning**  
   Reinforcement-learning inspired adaptive improvement over simulated deployment cycles.

4. **Multi-Modal Threat Detection**  
   Risk assessment combining text, image, audio, and metadata embeddings.

5. **Scalability & Performance Study**  
   Simulated load testing under varying operational conditions.

6. **Output Safety Classification**  
   Cross-attention based safety evaluation across:
   - Toxicity  
   - Bias  
   - Hallucination  
   - Privacy violations  

---

## Architecture Highlights

- Transformer-based semantic embedding (BERT)
- Multi-head attention fusion
- Multi-dimensional safety scoring
- Reinforcement-style adaptive learning
- Baseline comparisons:
  - Rule-based classifier
  - SVM
  - Random Forest
  - Logistic Regression

---

## Installation

Install required dependencies:

```bash
pip install torch transformers scikit-learn matplotlib seaborn pandas numpy

---

## Running Experiments

Run all experiments:

```bash
python ai_guardrail_experiments.py

---

python ai_guardrail_experiments.py 1   # Adversarial Detection
python ai_guardrail_experiments.py 2   # Bias Detection
python ai_guardrail_experiments.py 3   # Adaptive Learning
python ai_guardrail_experiments.py 4   # Multi-Modal Detection


---

### Generated Outputs

After execution, the following artifacts are generated:

experiment_log.txt — Detailed execution logs

experiment_results.json — Complete numerical results for all experiments

adversarial_detection_results.png — Performance comparison across baseline methods

adaptive_learning_progress.png — 30-day adaptive learning curve visualization

---

### Example Experimental Results

Representative results reported in the paper include:

Adversarial Prompt Detection: ~95%+ accuracy

Adaptive Guardrail Learning: ~6–7% improvement over a simulated 30-day deployment

Multi-Modal Threat Detection: Fusion-based approach outperforms text-only models

Output Safety Classification: High precision and recall across toxicity, bias, hallucination, and privacy dimensions

Exact values may vary slightly depending on hardware and runtime environment.

--- 
