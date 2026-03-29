# logprobs
Investigate more deterministic approach to understanding model response confidence and hallucinations. 


# Entropy-Based Routing & Confidence Circuit Breakers for LLMs

This repository provides architectural frameworks and Python implementations for building mathematical guardrails around Large Language Models (LLMs). 

By analyzing token-level logarithmic probabilities (`logprobs`), we can extract a mathematical measure of an LLM's internal uncertainty (predictive entropy) and use it as an automated circuit breaker. This transforms LLM hallucinations from an unpredictable, existential business risk into a measurable, manageable routing workflow.

## 🧠 The Core Philosophy

When deploying generative AI in enterprise environments, the baseline expectation is often "zero defects." However, comparing probabilistic models to deterministic databases is a recipe for stalled adoption. 

Instead, we benchmark LLMs against the **Human Baseline**—treating the model as a highly capable, incredibly fast "intern" that requires supervision. 
* **Human Bias** is emotional and lived. **LLM Bias** is statistical and systemic.
* **Human Confabulation** fills memory gaps to protect the ego. **LLM Hallucinations** are simply statistical misfires where the model is forced to predict a token from a flattened probability curve.

The danger of LLM hallucinations is not that they happen; it is the **blast radius at machine speed**. To mitigate this, we move from trusting the AI's "magic" to measuring its math.

## 📐 The Math Behind the Magic: Defeating the Illusions

Most commercial LLMs are poorly calibrated to self-report confidence due to RLHF training. Asking a model "Are you sure?" will almost always yield a false "Yes." Instead, we evaluate the raw `logprobs` returned by the API.

However, naive averaging of probabilities introduces two major traps:

### 1. The "Glue Word" Illusion
Language models are exceptionally good at grammar. If an LLM generates a 100-word response, 80 of those words might be structural "glue" (the, and, is) with near-100% mathematical certainty. If the model hallucinates a critical fact (like a dollar amount) at 30% certainty, averaging the whole paragraph will mask the hallucination. 

**Solution:** We measure the "Weakest Link"—the localized minimum token probability in the sequence. 

### 2. The Hallucination Snowball Effect (Exposure Bias)
Because LLMs are autoregressive, generated tokens become the absolute ground truth for the next prediction. If a model guesses a low-probability token, it immediately pivots to defending that false premise with high confidence. 

Mathematically, the joint probability of a sequence is:
$$P(T_1, T_2, \dots, T_n) = \prod_{i=1}^{n} P(T_i \mid T_{<i})$$

To prevent arithmetic underflow when calculating this, we use the sum of logarithmic probabilities to calculate a **Length-Normalized Sequence Score**:
$$\text{Score} = \frac{1}{N} \sum_{i=1}^{N} \log P(T_i)$$

This ensures that a single catastrophic guess mathematically taints the fluent, highly confident hallucination that follows it.

## 🛠️ Implementation: The Confidence Scorecard

The following Python script calls the OpenAI API and evaluates the `logprobs` to generate a three-point Confidence Scorecard:
1. **Overall Sequence Score:** The length-normalized joint probability.
2. **Weakest Link Score:** The lowest single token probability (The Circuit Breaker).
3. **Risk Density:** The percentage of tokens falling below the enterprise risk threshold.

```python
import math
from openai import OpenAI

client = OpenAI()

def generate_confidence_scorecard(prompt, model="gpt-4o", risk_threshold=80.0):
    """
    Generates a comprehensive confidence scorecard based on API logprobs.
    """
    response = client.chat.completions.create(
        model=model, 
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        temperature=0.0 
    )
    
    if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
        return "Error: No logprobs returned."

    total_logprob = 0.0
    token_count = 0
    full_text = ""
    min_prob = 100.0
    weakest_token = ""
    weak_token_count = 0

    for token_info in response.choices[0].logprobs.content:
        token_str = token_info.token
        logp = token_info.logprob
        linear_prob = math.exp(logp) * 100
        
        # Track weakest link
        if linear_prob < min_prob:
            min_prob = linear_prob
            weakest_token = token_str
            
        # Track risk density
        if linear_prob < risk_threshold:
            weak_token_count += 1
            
        total_logprob += logp
        token_count += 1
        full_text += token_str

    # Sequence math
    normalized_logprob = total_logprob / token_count
    sequence_confidence = math.exp(normalized_logprob) * 100
    risk_density = (weak_token_count / token_count) * 100

    # Output
    print(f"Generated Text: {full_text.strip()}\n" + "="*60)
    print("CONFIDENCE SCORECARD")
    print("-" * 60)
    print(f"Overall Sequence Score:   {sequence_confidence:.2f}%")
    
    alert = "⚠️ TRIPS CIRCUIT BREAKER" if min_prob < risk_threshold else "✅ PASS"
    print(f"Weakest Link Score:       {min_prob:.2f}% (Token: {repr(weakest_token)}) {alert}")
    print(f"Risk Density:             {risk_density:.1f}% of tokens are below {risk_threshold}% threshold.")
    print("="*60)

# Example Usage
generate_confidence_scorecard("What was the exact date of the Treaty of Waitangi?")
```

## 🤖 Advanced Architecture: Multi-Agent Orchestration

This entropy scorecard is designed to act as a **gating mechanism** in multi-agent workflows. 

If Agent A hallucinates and passes that data to Agent B, the error compounds across the swarm. By forcing sub-agents to return both their text payload AND their Entropy Scorecard to a central Orchestrator, we create dynamic routing:

* **Low Entropy (High Certainty):** The Orchestrator passes the output to the next agent (Straight-through automation).
* **High Entropy (Low Certainty):** The Orchestrator halts the standard flow and routes the task to either a specialized "Verifier Agent" (armed with strict RAG access) or kicks it out to a Human-in-the-Loop for review.

## ⚠️ A Note on Grounding (Internal Consistency vs. External Truth)

Logprobs measure **internal consistency**—how perfectly a token aligns with the statistical distribution of the training data. A model can be 99.9% mathematically certain about a pervasive internet myth. 

To achieve **external truth**, this entropy routing system must be paired with **Retrieval-Augmented Generation (RAG)**. RAG forces the model to synthesize verified documents, while the logprob circuit breakers catch synthesis errors and uncertainty. Finally, human supervisors remain accountable for the ultimate business decision.
