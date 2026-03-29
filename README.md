# logprobs
A Python tool for measuring LLM response confidence and detecting hallucinations using token-level log probabilities.

---

## Overview

When an LLM generates text, it doesn't just pick words — it calculates a probability distribution over its entire vocabulary at every step and samples from it. The API can return these raw probabilities (`logprobs`), giving us a mathematical window into how certain the model was about each word it generated.

This tool analyses those probabilities and produces a **Confidence Scorecard** designed to answer one question: *should I trust this response?*

---

## The Problem with Naive Confidence Checks

Asking a model "are you sure?" doesn't work — RLHF training biases models toward confident-sounding answers regardless of their internal uncertainty. Two subtler traps exist when analysing logprobs directly:

**1. The Glue Word Illusion**
A 100-word response might contain 80 structural words ("the", "and", "is") at ~99% confidence. If the model hallucinates one critical fact at 12% confidence, averaging all tokens masks it entirely.

**2. The Hallucination Snowball**
LLMs generate autoregressively — each token becomes the ground truth for the next. A single low-probability guess causes the model to confidently defend a false premise. The joint probability of the sequence captures this:

```
Score = (1/N) × Σ log P(Tᵢ)
```

Working in log space prevents arithmetic underflow on long responses and ensures one catastrophic token mathematically taints the sequence score.

---

## Setup

**Requirements:** Python 3, `openai` package

```bash
pip install openai
```

**Configuration** — edit the top of `logprobs-1` to point at your backend:

```python
# OpenAI public API
BASE_URL = "https://api.openai.com/v1"
API_KEY  = "sk-..."
MODEL    = "gpt-4o"

# Ollama (local) — requires Ollama v0.12.11 or newer for logprobs support
BASE_URL = "http://localhost:11434/v1"
API_KEY  = "ollama"
MODEL    = "llama3.1:8b"
```

> **Ollama note:** logprobs support was added in Ollama v0.12.11. Run `ollama --version` to check. Models confirmed to work: `llama3.1`, `llama3.2`, `qwen2.5`, `phi3.5`.

---

## Running

```bash
python logprobs-1
```

Edit `test_prompt` at the bottom of the script to test your own queries.

---

## Output: The Confidence Scorecard

### 1. Overall Sequence Score
The geometric mean probability across every token — average confidence per word. Above ~85% the model was generally on solid ground. Below 60% the response is statistically shaky even if it reads fluently.

### 2. Weakest Link Score
The single lowest-confidence token. This is the **circuit breaker**: one catastrophically uncertain token can corrupt everything that follows (the hallucination snowball). Fires `⚠️ TRIPS CIRCUIT BREAKER` if below the risk threshold (default 80%).

### 3. Risk Density
What fraction of tokens were generated below the risk threshold. A single weak token may just be an unusual proper noun. High risk density (>20%) means uncertainty is widespread, not isolated.

### 4. Narrowest Decision Margin
The smallest gap between the model's chosen token and its closest competitor. A near-zero margin means the model was genuinely torn at that point. This is especially dangerous on factual tokens — a 0.3% margin between `'February'` and `'March'` is a direct hallucination risk flag that the raw probability score alone would never surface. Fires `⚠️ COIN FLIP` if margin is below 5%.

---

## Output: The Histograms

### Token Probability Distribution
Shows how confidence was distributed across all generated tokens. A healthy response clusters heavily in the 90–100% bucket. Mass in the lower buckets signals widespread guessing.

```
TOKEN PROBABILITY DISTRIBUTION
  How confident was the model for each token it generated? ...
------------------------------------------------------------
    0- 10%                                              4 tokens ( 1.7%)  ' between'
   10- 20%  █                                           6 tokens ( 2.6%)  '.'
   ...
   90-100%  ████████████████████████                  143 tokens (61.4%)  ' Ng'
------------------------------------------------------------
```

### Decision Margin Distribution
Shows how decisive each token choice was. A large margin (right side) means the model had one clear favourite. Small margins (left side) represent coin-flip decisions — the hallucination risk signal that raw probability alone cannot reveal.

### Narrowest Margins Table
Lists the 10 closest decisions in the response with the chosen token, its runner-up, and the gap between them:

```
NARROWEST MARGINS — closest decisions (top 10)
  Chosen  Runner  Margin  Chosen token          Runner-up token
  ------  ------  ------  --------------------  --------------------
   52.1%   51.8%   0.3%  ' February'           ' March'           ⚠️
   18.4%   17.2%   1.2%  ' around'             ' approximately'
   29.6%   24.8%   4.8%  ','                   '.'
```

---

## Architecture: Using the Scorecard as a Circuit Breaker

The scorecard is designed as a **gating mechanism** in multi-agent workflows:

```
Agent response + Scorecard
        │
        ├─ High confidence (all metrics pass)  →  Pass to next agent
        │
        └─ Low confidence (any metric fails)   →  Route to Verifier Agent (RAG)
                                                   or Human-in-the-Loop
```

This turns hallucination management from reactive (catching errors after the fact) to proactive (routing based on measured uncertainty before errors propagate).

---

## Important Limitation: Internal Consistency ≠ Factual Accuracy

Logprobs measure **internal consistency** — how well a token fits the model's training distribution. A model can be 99.9% certain about a widely-repeated internet myth.

To achieve **external truth**, pair this tool with:
- **RAG** — grounds the model in verified documents
- **Human review** — final accountability for high-stakes decisions

The logprob scorecard catches *statistical uncertainty*. RAG catches *factual divergence*. Together they cover most of the hallucination surface area.
