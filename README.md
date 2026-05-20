# LLM Memory & Retrieval: Mechanistic Interpretability + RAG

> An exploration of how Large Language Models store factual knowledge internally and how to augment that knowledge externally using a Retrieval-Augmented Generation (RAG) pipeline.

##  Overview

This repository contains a comprehensive Jupyter Notebook that investigates factual recall in LLMs from two distinct angles:
1. **The Inside-Out Approach (Mechanistic Interpretability):** Peering inside the black box of `GPT-2-XL` using `TransformerLens` to map exactly *where* and *how* factual associations (e.g., "France -> Paris") are stored and retrieved within the model's layers and attention heads.
2. **The Outside-In Approach (RAG):** Building a complete pipeline to inject external knowledge into `Qwen2.5-3B-Instruct` using Wikipedia data, ChromaDB, and an automated LLM-as-a-Judge evaluation framework.

##  Key Features

### Part 1: Mechanistic Interpretability
* **Activation Patching:** Implementing ROME-style activation patching to identify the exact layers where factual retrieval transitions occur.
* **Attention Head Analysis:** Isolating the Output-Value (OV) Circuit to calculate "Relation Scores" and visualize which specific attention heads act as associative memory stores.
* **Heatmap Visualizations:** Clear, plotted metrics showing the "early vs. late" layer dynamics in factual prediction.

### Part 2: Retrieval-Augmented Generation (RAG)
* **Custom Knowledge Base:** Automated Wikipedia scraping and preprocessing for a domain-specific corpus (Cinema/Directors).
* **Semantic Search:** Text chunking, normalization, and embedding generation using `SentenceTransformers` (`all-MiniLM-L6-v2`), stored persistently in **ChromaDB**.
* **Generation & Constraint:** Using `Qwen2.5-3B-Instruct` with strict system prompting to ensure grounded, hallucination-free generation based purely on retrieved context.
* **LLM-as-a-Judge Evaluation:** An automated evaluation loop testing different context sizes ($k \in \{0, 1, 3, 5, 7\}$) to quantify the exact performance lift provided by RAG and demonstrate the concept of diminishing returns in context scaling.

##  Tech Stack

* **Models:** `GPT-2-XL`, `Qwen/Qwen2.5-3B-Instruct`, `all-MiniLM-L6-v2`
* **Libraries:** `TransformerLens`, `PyTorch`, `Hugging Face Transformers`, `ChromaDB`, `SentenceTransformers`, `LangChain` (Text Splitters)
* **Visualization:** `Matplotlib`, `Seaborn`


### Usage

Open the main notebook (`main_notebook.ipynb`) and run the cells sequentially.

* *Note: The notebook contains built-in VRAM management (`torch.cuda.empty_cache()`), but running both parts back-to-back may require a kernel restart depending on your hardware limits.*

##  Key Findings

* **Fact Storage is Localized:** Mechanistic interpretability reveals that factual recall is not distributed evenly; it is heavily concentrated in the mid-to-late layers of the transformer.
* **RAG Diminishing Returns:** Evaluation shows massive accuracy leaps from $k=0$ (no context) to $k=3$, but plateauing gains beyond that, highlighting the need for high-quality retrieval over simply maximizing context size.




