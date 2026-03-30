# logprobs
A Python tool for measuring LLM generation stability and hallucination risk using token-level log probabilities and runner-up decision margins.

---

## Overview

When an LLM generates text, it doesn't just pick words — it calculates a probability distribution over its entire vocabulary at every step and samples from it. The API can return these raw probabilities (`logprobs`), giving us a mathematical window into how certain the model was about each word it generated.

This repo contains two scripts:
- **`logprobs-1`** — the original Confidence Scorecard with histograms, margin tables, and snowball detection.
- **`logprobs-2`** — everything in `logprobs-1` plus a **Token Gap Chart** that visualises the margin between chosen and runner-up tokens as a column chart across the sentence.

Both analyse token probabilities and produce a Confidence Scorecard designed to answer a narrower and more useful question:

**Did the model generate this response along a stable path, or did it make fragile choices that increase the risk of hallucination?**

This is not a fact checker. It does not prove whether a response is true in the real world. Instead, it measures signs of generative instability: low-support token choices, narrow decision margins, and "guess-then-commit" patterns where the model becomes fluent after an uncertain step.

---
## What This Tool Is (and Is Not)

This tool is best understood as a hallucination-risk and generation-stability analyzer.

It does **not** tell you whether a statement is true.
It does **not** replace external verification.
It does **not** convert logprobs into a direct "truth percentage."

What it does do is identify signs that the model may have been guessing while generating:
- low-probability token choices
- narrow margins between the chosen token and the runner-up
- local "fork points" where the model could easily have gone a different way
- snowball patterns where an uncertain choice is followed by highly fluent continuation

In other words:

- **self-reported confidence** is rhetorical confidence
- **logprob analysis** is generative confidence
- **external checking** is factual confidence

This tool focuses on the middle one.

---

## The Problem with Naive Confidence Checks

Asking a model "are you sure?" doesn't work — RLHF training biases models toward confident-sounding answers regardless of their internal uncertainty. 

A useful way to think about the gap is this:
 - When a model says "I'm 95% confident," it is generating a confidence statement as text.
 - When we inspect token logprobs and runner-up margins, we are looking at how contested the model's choices actually were during generation.

Those are not the same thing.

Three subtler traps exist when analysing logprobs directly:

**1. The Glue Word Illusion**
A 100-word response might contain 80 structural words ("the", "and", "is") at ~99% confidence. If the model hallucinates one critical fact at 12% confidence, averaging all tokens masks it entirely.

**2. The Hallucination Snowball**
LLMs generate autoregressively — each token becomes the ground truth for the next. A single low-probability guess causes the model to confidently defend a false premise. The joint probability of the sequence captures this:

```
Score = (1/N) × Σ log P(Tᵢ)
```

Working in log space prevents arithmetic underflow on long responses and ensures one catastrophic token mathematically taints the sequence score.

**3. Margin Blindness**
Raw probability alone doesn't reveal how close a decision was. A token chosen at 52% with its nearest rival at 51% is a near coin-flip. A token chosen at 52% with no close rival is a committed choice. These look identical in a probability-only view but carry very different hallucination risk.

---

## Setup

**Requirements:** Python 3, `openai` package

```bash
pip install openai
```

**Configuration** — edit the top of `logprobs-1` or `logprobs-2` to point at your backend:

```python
BASE_URL = "https://api.openai.com/v1"
API_KEY  = "sk-..."
MODEL    = "gpt-4o"
```

---

## Supported Backends

### OpenAI API

Set `BASE_URL = "https://api.openai.com/v1"` and provide your API key.

**Model compatibility note:** `top_logprobs` combined with `temperature=0.0` triggers HTTP 500 errors on some specific `gpt-5` version. 

| Model | logprobs | Status |
|---|---|---|
| `gpt-4o`  | ✅ | Confirmed working |
| `gpt-5`  | ⚠️ | May 500 on `top_logprobs=2` + `temperature=0.0` |

If you receive an `InternalServerError: 500`, switch to a different model version. Include the `request ID` from the error message if raising a support ticket with OpenAI.

### Ollama (local)

```python
BASE_URL = "http://localhost:11434/v1"
API_KEY  = "ollama"   # any non-empty string
MODEL    = "llama3.1:8b"
```

> **Ollama version requirement:** logprobs support was added in **v0.12.11**. Run `ollama --version` to check — earlier versions silently return `null` for logprobs regardless of the request parameter.

**Models confirmed to return logprobs via Ollama:**

| Model | logprobs | Notes |
|---|---|---|
| `llama3.1:8b` | ✅ | Confirmed working |

---

## Running

```bash
python logprobs-1       # Original scorecard
python logprobs-2       # Scorecard + Token Gap Chart
```

Both scripts share the same scorecard output. `logprobs-2` adds the Token Gap Chart section (see below). Edit `test_prompt` at the bottom of either script to change the query.

---

## Output: The Confidence Scorecard

### 1. Overall Sequence Score + Adjusted Sequence Score

Both scores are displayed side by side with a delta:

```
Overall Sequence Score:   72.47%  →  Adjusted: 58.31%  (-14.16%)
```

**Overall** is the raw geometric mean of all token probabilities — a fast read on average confidence, but it is inflated by three systematic biases.

**Adjusted** applies three corrections to produce a more conservative and honest trust signal:

| Correction | What it fixes |
|---|---|
| **Glue exclusion** | Removes tokens >98% probability (structural words like "the", "and", "is" that carry no factual signal but inflate the mean) |
| **Snowball correction** | Post-pivot tokens are capped at the pivot's probability — their high certainty was conditioned on an uncertain choice, not earned independently |
| **Margin weighting** | Near coin-flip tokens are down-weighted; tokens chosen decisively count fully |

A large negative delta (>10%) means the raw score was significantly flattered. Use the adjusted score as your primary generation-stability signal.  A low score does not prove the answer is false. It indicates that the model's path to the answer was fragile and deserves verification.

### 2. Weakest Link Score
The single lowest-confidence token. This is the **circuit breaker**: one catastrophically uncertain token can corrupt everything that follows (the hallucination snowball). Fires `⚠️ TRIPS CIRCUIT BREAKER` if below the risk threshold (default 80%).  Interpret this as the most fragile point in the generation path, not automatic proof that the answer is wrong. It matters most when the token is semantically meaningful (for example, a date, name, number, or place), not when it is just punctuation or formatting.

### 3. Risk Density
What fraction of tokens were generated below the risk threshold. A single weak token may just be an unusual proper noun — normal. High risk density (>20%) means uncertainty is widespread, not isolated.  Risk density is about how widespread instability was. It does not mean that the same percentage of the answer is false.

### 4. Narrowest Decision Margin
The smallest gap between the model's chosen token and its closest competitor. A near-zero margin means the model was genuinely torn at that point — the response could easily have gone a different way. This is especially dangerous on factual tokens: a 0.3% margin between `'February'` and `'March'` is a direct hallucination risk flag that the raw probability score alone would never surface. Fires `⚠️ COIN FLIP` if margin is below 5%.  Small margins on semantic tokens (meaningful words)are much more interesting than small margins on stylistic or structural tokens (',').  Not all low-confidence tokens are equally important. A weak comma or bracket is usually noise. A weak year, name, place, or quoted term is much more interesting. In practice, semantic instability matters more than structural instability.

---

## Output: The Histograms

Each histogram includes a printed legend and a dynamic summary line that reads the actual data and flags notable patterns.

### Token Probability Distribution
Shows how confidence was distributed across all generated tokens. A healthy response clusters heavily in the 90–100% bucket. Mass in the lower buckets signals widespread guessing.

```
TOKEN PROBABILITY DISTRIBUTION
------------------------------------------------------------
    0- 10%                                              4 tokens ( 1.7%)  ' between'
   10- 20%  █                                           6 tokens ( 2.6%)  '.'
   ...
   90-100%  ████████████████████████                  143 tokens (61.4%)  ' Ng'
------------------------------------------------------------
  ✅ 61% of tokens are in the 90-100% band — the model was largely certain.
  ⚠️  10% of tokens are below 50% confidence — investigate the example tokens above.
```

### Decision Margin Distribution
Shows how decisive each token choice was, independently of raw probability. A large margin (right side) means the model had one clear favourite. Small margins (left side) are coin-flip decisions — the hallucination risk that raw probability alone cannot reveal.

```
DECISION MARGIN DISTRIBUTION  (chosen prob − runner-up prob)
------------------------------------------------------------
    0- 10%  ████                                       22 tokens ( 9.4%)  ' between'
   ...
   90-100%  ██████████████████                         98 tokens (42.1%)  'The'
------------------------------------------------------------
  ✅ Only 9% of tokens had a margin below 10% — most choices were decisive.
```

### Narrowest Margins Table
Lists the 10 closest decisions in the response with the chosen token, its probability, the runner-up token, its probability, and the gap between them:

```
NARROWEST MARGINS — closest decisions (top 10)
  Chosen  Runner  Margin  Chosen token          Runner-up token
  ------  ------  ------  --------------------  --------------------
   52.1%   51.8%   0.3%  ' February'           ' March'           ⚠️
   18.4%   17.2%   1.2%  ' around'             ' approximately'
   29.6%   24.8%   4.8%  ','                   '.'
```

### Token Gap Chart (logprobs-2 only)

`logprobs-2` adds a **Token Gap Chart** — a column-based visualisation that shows the margin between the chosen token and the runner-up for every token in sentence order. This makes it easy to see confidence expanding and contracting as the response progresses.

```
TOKEN GAP CHART — sentence progression
--------------------------------------------------------------------------------
100% │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓
     │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓
 60% │▓▓▓▓    ▓▓▓▓        ▓▓▓▓    ▓▓▓▓
     │▓▓▓▓    ▓▓▓▓        ▓▓▓▓    ▓▓▓▓
     │▓▓▓▓                ▓▓▓▓
     │▓▓▓▓                ▓▓▓▓
 20% │▓▓▓▓░░░░    ░░░░    ▓▓▓▓░░░░
     │░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     │────────────────────────────
      0   1   2   3   4   5   6
      The  of  Wait sign 1840 ,
      80% 12% 45% n/a ⚠3  62% 90%
```

**Anatomy of a column:**

| Zone | Symbol | Meaning |
|---|---|---|
| Upper block | `▓▓` | Chosen token probability — fills **down** from 100%. A tall `▓` block means the model assigned high probability to the token it chose. |
| White gap | (space) | The **margin** between chosen and runner-up. This is the key visual — watch it expand and contract across the sentence. |
| Lower block | `░░` | Runner-up token probability — fills **up** from 0%. A tall `░` block means the second-best alternative was also quite probable. |

**Reading the chart:**

| Pattern | What it means |
|---|---|
| Wide gap (`▓` and `░` far apart) | The model was decisive — it strongly preferred one token. Low hallucination risk. |
| Narrow gap (`▓` and `░` nearly touching) | The model was torn between two options. If this falls on a factual word, treat it as suspect. |
| No gap (`▓` and `░` merge) | A near coin-flip. The response could easily have diverged here. |
| Both blocks short, wide gap | Neither token was very probable — the model spread probability across many candidates. Decisive between top two, but uncertain overall. |

**What to look for:**
- **Clusters of narrow-gap columns** — the model was guessing through an entire phrase, not just one token.
- **A sudden squeeze on a factual word** (name, date, number) — the strongest hallucination signal in the chart.
- **A wide gap following a narrow gap** — may indicate a snowball: the model committed to an uncertain choice and then became confidently wrong.

Below each chart chunk, three label rows show the token index, the token text, and the margin percentage (`⚠` flags margins below 5%).

After the chart, a **Chart Interpretation Summary** reports the average margin, counts of decisive vs coin-flip tokens, cluster detection for consecutive low-gap runs, and an overall assessment.

---

## Output: Snowball Effect Analysis

The snowball section identifies **pivot tokens** — uncertain choices that were immediately followed by a run of high-confidence tokens. This is the hallucination snowball signature: the model committed to a low-confidence guess and then confidently built on it.

```
SNOWBALL EFFECT ANALYSIS
------------------------------------------------------------
  ⚠️  2 potential snowball(s) found

   Pos  Chosen%   Margin  PostAvg    Lift  Pivot token → Context
  ----  -------  -------  -------  ------  ----------------------------------------
    47    5.5%    3.2%    89.1%   +16.6%  ' between' → ' February 5 and March 16'  ⚠️
```

**Lift** is the key number: how much the post-pivot average confidence exceeds the overall response average. A high lift on a factual word means the model guessed, then committed — the specific claim that follows is the one to verify.

The snowball detection also feeds the **Adjusted Sequence Score**: post-pivot tokens are capped at the pivot's probability in the adjusted calculation, preventing a snowball run from masking its own root cause.

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

**Routing signals in priority order:**
1. Adjusted Sequence Score delta > 10% → snowball or widespread uncertainty present
2. Weakest Link trips circuit breaker → single point of failure, halt and verify
3. Narrowest Margin flags coin flip on a factual token → targeted fact-check required
4. Risk Density > 20% → diffuse uncertainty, consider full re-generation with RAG

This turns hallucination management from reactive (catching errors after the fact) to proactive (routing based on measured uncertainty before errors propagate).

---
## How to Interpret Results

This scorecard measures hallucination risk, not truth.

A response can be:
- factually correct but generated fragily
- factually wrong but generated confidently
- factually correct and generated stably
- factually wrong and generated fragily

The scorecard helps identify the stable vs fragile part.

### Stable generation
The model appears to have generated along a well-supported path.
Typical signs:
- relatively strong adjusted score
- low risk density
- few narrow margins
- no meaningful snowball pivots on semantic tokens

### Fragile generation
The answer may be correct, but the generation path was shaky.
Typical signs:
- noticeable raw vs adjusted drop
- some narrow margins
- some snowball pivots
- uncertainty spread across parts of the response

### Probable hallucination pattern
This is the classic "guess then commit" shape.
Typical signs:
- large raw vs adjusted drop
- high risk density
- very weak semantic pivot
- narrow margins on dates, names, numbers, places, references
- fluent continuation after an unstable choice

---

## Important Limitation: Internal Consistency ≠ Factual Accuracy

Logprobs measure **internal consistency** — how well a token fits the model's training distribution. A model can be 99.9% certain about a widely-repeated internet myth.

To achieve **external truth**, pair this tool with:
- **RAG** — grounds the model in verified documents
- **Human review** — final accountability for high-stakes decisions

The logprob scorecard catches *statistical uncertainty*. RAG catches *factual divergence*. Together they cover most of the hallucination surface area.
