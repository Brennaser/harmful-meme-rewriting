# Harmful Meme Rewriting

This repository contains code and data for a class project on rewriting harmful internet memes into safer memes while preserving the original joke and context.

We compare several large language models on this task and evaluate them with automatic metrics and a small manual review.

---

## Project Goals

Given a meme image and its original caption, the system aims to generate a new caption that

* removes or softens hate speech, toxicity, and offensive content  
* keeps the original situation, target, and basic joke structure as much as possible  

Outputs are evaluated with

* semantic similarity between original and rewritten text  
* toxicity reduction using a pretrained toxicity classifier  
* human ratings of contextual coherence when the new text is paired with the original image  

---

## Repository Structure

Top level folders:

* `facebook-meme-subset/`  
  Subset of a hateful memes style dataset. Contains meme images and a JSON lines file with entries like  
  ```json
  {"id": "80243", "img": "img/80243.png", "label": 1, "text": "mississippi wind chime"}

├── facebook-meme-subset/
│   ├── img/                        # Original meme images
│   ├── train190_subset.jsonl       # JSONL input file for rewriting
│
├── llms-to-rewrite-text/
│   ├── gemini_rewriter.py          # Uses Gemini to rewrite meme text
│   ├── gemma_rewriter.py           # Uses Gemma (text only) to rewrite memes
│
├── image-generation-scripts/
│   ├── text-overlay.py             # Creates safe meme by overlaying text on original image
│   ├── generate_safe_memes.py      # Uses diffusion to produce new safe images
│
├── text-and-image-results/
│   ├── generated-images-v1/        # Results for images from unsafe model
│   ├── generated-images-v1/        # Results for images from better model
│   ├── memes_llava.csv/            # LLaVA rewritten text (each model has this)
│
├── llm-evaluation-scores/
│   ├── model_metrics_summary.csv   # Summary metrics across models
│   ├── llama_evaluated.csv         # Toxity score & cosine similarity for LLaVA rewritten text
│
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation file
└── .gitignore                      # Files to ignore in version control


