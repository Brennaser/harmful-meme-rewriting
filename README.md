# Harmful Meme Rewriting

This repository contains the code and data for a class project on rewriting harmful internet memes into safer memes while preserving the original joke and context.

We compare several large language models on this task and evaluate them with automatic metrics and a small manual review.

---

## Project Overview

Given a meme image and its original caption, the goal is to generate a new caption that:

* Removes or softens hate speech, toxicity, and offensive content  
* Keeps the original situation and target of the joke as much as possible  

Outputs from these models are evaluated with:

* Semantic similarity between original and rewritten text  
* Toxicity reduction using a pretrained classifier  
* Human ratings of contextual coherence when the new text is paired with the original image  

---

### Directory Overview

**eval/**  
Contains evaluation scripts and outputs, including the automatic metrics and CSV templates for human contextual-coherence review.

**img/**  
Subset of the hateful memes dataset (images + `train190_subset.jsonl`). Used as input to rewriting models.

**meme_gen/**  
Scripts for generating rewritten meme captions (Gemini, Gemma, LLaVA). Includes optional text-overlay tools.

**models/**  
Model wrappers for each rewriting approach:
- Gemini 2.5
- Gemma 2 + BLIP (multimodal)
- LLaVA (vision-language)

**results/**  
All generated meme rewrites, cleaned outputs, and evaluated CSVs. Also includes directories storing full safe meme generations.

Disclaimer: Large language models were used to assist with writing, editing, and troubleshooting code throughout this project. All final decisions, implementations, and validations were performed by the author.
