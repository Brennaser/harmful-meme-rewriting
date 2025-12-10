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

