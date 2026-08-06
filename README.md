# Autonomous Code Review Agent (Fine-Tuned LLM + Multi-Tool Agent)

An agentic system that reviews Python pull requests using a **QLoRA fine-tuned
code-review LLM**, orchestrated by a **LangGraph** state machine that also
calls static analysis, test execution, and retrieval-augmented lookup of past
similar bugs — then **self-scores its confidence** and escalates to a human
when unsure.

This project is designed to demonstrate two skill sets that rarely appear
together on a resume:

1. **Model fine-tuning** (QLoRA / PEFT / transformers) — training a small
   open-source code model to do a specific task well, instead of just calling
   a hosted API.
2. **Agentic system design** (LangGraph, tool orchestration, RAG, confidence-
   based self-correction) — the multi-agent, multi-step autonomy that most
   AI Engineer job postings ask for in 2026.

## Architecture

```
PR diff
   │
   ▼
Orchestrator (LangGraph state machine)
   │
   ├── Static Analysis tool (ruff)
   ├── Test Runner tool (pytest, sandboxed)
   ├── Retrieval tool (RAG over past reviewed PRs / known bug patterns)
   └── Fine-tuned Review Model (QLoRA-tuned Qwen2.5-Coder)
   │
   ▼
Confidence Scorer
   │
   ├── High confidence  → auto-post review comments
   └── Low confidence   → re-run with adjusted prompt, or flag for human
```

State flows through a single `AgentState` object (see `agent/state.py`) so
every tool call and decision is inspectable/loggable — this is what let's you
talk about "observability" and "evaluation" in interviews, not just "it
works."

## Repo layout

```
code-review-agent/
├── README.md
├── requirements.txt
├── finetuning/
│   ├── prepare_dataset.py       # builds an instruction dataset for review
│   └── train_qlora_colab.ipynb  # run this in Google Colab (free T4 GPU)
├── agent/
│   ├── state.py                 # shared state object passed through the graph
│   ├── confidence.py            # confidence scoring logic
│   ├── graph.py                 # LangGraph orchestrator (the core agent)
│   └── tools/
│       ├── static_analysis.py   # ruff wrapper
│       ├── test_runner.py       # sandboxed pytest runner
│       ├── retrieval.py         # FAISS-based RAG over past review comments
│       └── review_model.py      # loads & queries the fine-tuned model
└── demo/
    └── app.py                   # Streamlit demo UI
```

## How to run this (in order)

### 1. Fine-tune the model (Google Colab)
Open `finetuning/train_qlora_colab.ipynb` in Colab, select a **T4 GPU**
runtime, and run all cells. It:
- Downloads a small code-review instruction dataset (or your curated one)
- QLoRA fine-tunes `Qwen/Qwen2.5-Coder-1.5B-Instruct` (fits comfortably on a
  free T4; bump to 7B if you have Colab Pro / A100)
- Saves LoRA adapter weights you'll point the agent at

### 2. Set up the agent locally
```bash
pip install -r requirements.txt
```
Set your adapter path and any API keys (if using a hosted model as a
fallback) in a `.env` file — see `.env.example`.

### 3. Run the agent on a PR
```bash
python -m agent.graph --diff path/to/your.diff
```

### 4. Try the demo UI
```bash
streamlit run demo/app.py
```
