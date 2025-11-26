# Harmful Meme Rewriting

This repository contains the code and data for a class project on rewriting harmful internet memes into safer memes while preserving the original joke and context.

We compare several large language models on this task and evaluate them with automatic metrics and a small manual review.

---

## Project Overview

Given a meme image and its original caption, the goal is to generate a new caption that:

* Removes or softens hate speech, toxicity, and offensive content  
* Keeps the original situation and target of the joke as much as possible  
* Stays short, punchy, and meme like  

The project explores three model settings:

* Gemini based rewriting  
* Gemma based rewriting  
* LLaVA based multimodal rewriting  

Outputs from these models are evaluated with:

* Semantic similarity between original and rewritten text  
* Toxicity reduction using a pretrained classifier  
* Human ratings of contextual coherence when the new text is paired with the original image  

---

## Repository Structure

```text
harmful-meme-rewriting/
│
├── meme_gen/          # Scripts to generate safe meme captions with each model
├── eval/              # Evaluation scripts and notebooks
├── models/            # Model specific configuration or helper code
├── img/               # Meme image files (subset of hateful_memes)
├── results/           # CSV files with model outputs and evaluation results
│
├── .gitignore
├── requirments.txt    # Python packages for the project
└── README.md          # You are here
