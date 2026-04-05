import math
from openai import OpenAI

# --- Configuration ---
# For OpenAI public API:
#   BASE_URL = "https://api.openai.com/v1"
#   API_KEY  = "sk-..."
#
# For Ollama (local):
#   BASE_URL = "http://localhost:11434/v1"
#   API_KEY  = "ollama"   # Ollama ignores this but the client requires a value

BASE_URL = "http://localhost:11434/v1"
API_KEY  = "ollama"
MODEL    = "llama3.1:8b"
# ---------------------

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def generate_confidence_scorecard(prompt, model=MODEL, risk_threshold=80.0):
    """
    Generates a comprehensive confidence scorecard based on API logprobs.
    """
    print(f"Analyzing Prompt: '{prompt}'\n" + "="*60)
    print(f"[DEBUG] Connecting to: {BASE_URL}")
    print(f"[DEBUG] Model: {model}")
    print(f"[DEBUG] Sending request...")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=2,
        temperature=0.0
    )

    print(f"[DEBUG] Response received. Finish reason: {response.choices[0].finish_reason}")
    print(f"[DEBUG] logprobs present: {response.choices[0].logprobs is not None}")
    if response.choices[0].logprobs:
        content = response.choices[0].logprobs.content
        print(f"[DEBUG] logprobs.content type: {type(content)}, length: {len(content) if content else 'None/empty'}")
        if content:
            first = content[0]
            print(f"[DEBUG] First token: {repr(first.token)}, top_logprobs count: {len(first.top_logprobs) if first.top_logprobs else 0}")
            if first.top_logprobs:
                for i, t in enumerate(first.top_logprobs):
                    print(f"[DEBUG]   top_logprobs[{i}]: token={repr(t.token)}, prob={math.exp(t.logprob)*100:.2f}%")
    else:
        print(f"[DEBUG] Raw response: {response}")

    if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
        print("[DEBUG] No logprobs content — Ollama may not support logprobs for this model.")
        return "Error: No logprobs returned."

    all_probs = []   # (chosen_prob, margin, chosen_token, runner_up_token)
    total_logprob = 0.0
    token_count = 0
    full_text = ""

    # Tracking variables for our scorecard
    min_prob = 100.0
    weakest_token = ""
    weak_token_count = 0
    min_margin = 100.0
    min_margin_chosen = ""
    min_margin_runner = ""

    for token_info in response.choices[0].logprobs.content:
        token_str = token_info.token
        logp = token_info.logprob

        linear_prob = math.exp(logp) * 100

        # Extract runner-up: sort top_logprobs by prob descending, skip duplicates
        runner_up_prob = None
        runner_up_token = ""
        if token_info.top_logprobs and len(token_info.top_logprobs) > 1:
            sorted_top = sorted(token_info.top_logprobs, key=lambda t: t.logprob, reverse=True)
            best_prob = math.exp(sorted_top[0].logprob) * 100
            best_token = sorted_top[0].token
            for candidate in sorted_top[1:]:
                if candidate.token != best_token:
                    runner_up_prob = math.exp(candidate.logprob) * 100
                    runner_up_token = candidate.token
                    break

        margin = (best_prob - runner_up_prob) if runner_up_prob is not None else None
        all_probs.append((linear_prob, margin, token_str, runner_up_token, runner_up_prob))

        # Track the absolute lowest confidence token
        if linear_prob < min_prob:
            min_prob = linear_prob
            weakest_token = token_str

        # Track the narrowest decision margin
        if margin is not None and margin < min_margin:
            min_margin = margin
            min_margin_chosen = token_str
            min_margin_runner = runner_up_token

        # Count how many tokens fall below our enterprise risk threshold
        if linear_prob < risk_threshold:
            weak_token_count += 1

        total_logprob += logp
        token_count += 1
        full_text += token_str

    # Calculate overall sequence confidence
    normalized_logprob = total_logprob / token_count
    sequence_confidence = math.exp(normalized_logprob) * 100

    # Calculate risk density
    risk_density = (weak_token_count / token_count) * 100

    # ── Snowball detection ───────────────────────────────────────────────
    # Must run before adjusted score so we know which post-pivot tokens
    # to correct. pivot_for_index maps token index → lowest pivot prob
    # covering it (handles overlapping windows conservatively).
    SNOWBALL_WINDOW        = 8
    SNOWBALL_MIN_WINDOW    = 3
    SNOWBALL_MARGIN_THRESH = 10.0
    SNOWBALL_LIFT_THRESH   = 15.0

    snowballs = []
    pivot_for_index = {}

    for i, (prob, margin, tok, runner, runner_prob) in enumerate(all_probs):
        is_low_prob   = prob < risk_threshold
        is_close_call = margin is not None and margin < SNOWBALL_MARGIN_THRESH
        if not (is_low_prob or is_close_call):
            continue
        window = all_probs[i + 1 : i + 1 + SNOWBALL_WINDOW]
        if len(window) < SNOWBALL_MIN_WINDOW:
            continue
        post_avg = sum(w[0] for w in window) / len(window)
        lift = post_avg - sequence_confidence
        if lift >= SNOWBALL_LIFT_THRESH:
            post_text = "".join(w[2] for w in window)
            snowballs.append((lift, i, prob, margin, tok, runner, runner_prob, post_avg, post_text))
            for j in range(i + 1, min(i + 1 + SNOWBALL_WINDOW, len(all_probs))):
                if j not in pivot_for_index or prob < pivot_for_index[j]:
                    pivot_for_index[j] = prob

    # ── Adjusted Sequence Score ──────────────────────────────────────────
    # Three corrections applied on top of the raw geometric mean:
    #
    #   1. Glue exclusion   — tokens above GLUE_CEILING are near-certain
    #                         structural words that dilute the mean without
    #                         carrying factual information.
    #
    #   2. Snowball correction — post-pivot tokens are capped at the pivot's
    #                         probability. Their high confidence is conditioned
    #                         on the uncertain choice, not earned independently.
    #
    #   3. Margin weighting — tokens with a narrow decision margin are down-
    #                         weighted. A 52% token with a 51% runner-up is
    #                         genuinely less reliable than a 52% token chosen
    #                         decisively from a flat field.
    GLUE_CEILING          = 98.0
    MARGIN_PENALTY_THRESH = 20.0

    adj_logp_total   = 0.0
    adj_weight_total = 0.0

    for i, (prob, margin, tok, runner, runner_prob) in enumerate(all_probs):
        # 1. Exclude glue words
        if prob > GLUE_CEILING:
            continue
        # 2. Snowball correction
        effective_prob = min(prob, pivot_for_index[i]) if i in pivot_for_index else prob
        logp = math.log(max(effective_prob, 0.01) / 100)
        # 3. Margin weighting
        if margin is not None and margin < MARGIN_PENALTY_THRESH:
            weight = 0.5 + (margin / (MARGIN_PENALTY_THRESH * 2))
        else:
            weight = 1.0
        adj_logp_total   += weight * logp
        adj_weight_total += weight

    adjusted_confidence = math.exp(adj_logp_total / adj_weight_total) * 100 if adj_weight_total > 0 else 0.0

    # Output the Scorecard
    print(f"Generated Text: {full_text.strip()}\n" + "="*60)
    print("CONFIDENCE SCORECARD")
    print(
        "  Each metric looks at model certainty from a different angle.\n"
        "  Together they answer: 'Should I trust this response?'"
    )
    print("-" * 60)

    # 1. Overall Score and Adjusted Score
    delta = adjusted_confidence - sequence_confidence
    delta_str = f"{delta:+.2f}%"
    print(f"Overall Sequence Score:   {sequence_confidence:.2f}%  →  Adjusted: {adjusted_confidence:.2f}%  ({delta_str})")
    print(
        "  Overall  = geometric mean of all token probabilities. Fast read on\n"
        "             average confidence, but inflated by glue words and snowballs.\n"
        "  Adjusted = same mean after three corrections:\n"
        "             • Glue exclusion  — removes tokens >98% (structural words\n"
        "               that carry no factual signal, e.g. 'the', 'and', 'is').\n"
        "             • Snowball correction — post-pivot tokens are capped at the\n"
        "               pivot's probability; their certainty was not earned freely.\n"
        "             • Margin weighting — near coin-flip tokens are down-weighted\n"
        "               relative to tokens chosen decisively.\n"
        f"  A large negative delta (e.g. >10%) means the raw score was significantly\n"
        f"  flattered. The adjusted score is the more conservative trust signal."
    )

    # 2. The Weakest Link Score (Crucial for circuit breakers)
    alert = "⚠️ TRIPS CIRCUIT BREAKER" if min_prob < risk_threshold else "✅ PASS"
    print(f"\nWeakest Link Score:       {min_prob:.2f}% (Token: {repr(weakest_token)}) {alert}")
    print(
        f"  The single lowest-confidence token in the entire response.\n"
        f"  This is the circuit breaker metric: one catastrophically uncertain token\n"
        f"  can corrupt everything that follows it (the hallucination snowball effect).\n"
        f"  Threshold is {risk_threshold}% — anything below fires the alert."
    )

    # 3. Risk Density
    print(f"\nRisk Density:             {risk_density:.1f}% of tokens are below {risk_threshold}% threshold.")
    print(
        f"  What fraction of the response was generated with low confidence?\n"
        f"  A single weak token may be a proper noun or unusual word — normal.\n"
        f"  High risk density (>20%) means uncertainty is widespread, not isolated."
    )

    # 4. Narrowest Decision Margin
    if min_margin < 100.0:
        margin_alert = "⚠️ COIN FLIP" if min_margin < 5.0 else "✅ PASS"
        print(f"\nNarrowest Margin:         {min_margin:.2f}% ({repr(min_margin_chosen)} vs {repr(min_margin_runner)}) {margin_alert}")
        print(
            "  The smallest gap between the chosen token and its closest competitor.\n"
            "  A near-zero margin means the model was genuinely torn — the response\n"
            "  could easily have gone a different way at that point. This is especially\n"
            "  dangerous on factual tokens like dates, names, or numbers."
        )
    print("="*60)

    # Token Probability Distribution histogram (by chosen probability)
    print(
        "\nTOKEN PROBABILITY DISTRIBUTION\n"
        "  Each row is a 10% probability band. The bar shows what proportion of\n"
        "  tokens fell into that band. The token shown is the least confident\n"
        "  example from that band — the one closest to the lower edge.\n"
        "\n"
        "  HOW TO READ THIS:\n"
        "    90-100%  ████████████  — Model was certain. Grammar, common words,\n"
        "                             repeated proper nouns. Expected to dominate.\n"
        "    50- 90%  ███           — Some uncertainty. The model had options but\n"
        "                             leaned toward one. Usually acceptable.\n"
        "     0- 50%  █             — Model was genuinely unsure. Unusual words,\n"
        "                             contested facts, or hallucination territory.\n"
        "\n"
        "  WHAT TO LOOK FOR:\n"
        "    Good  — A tall 90-100% bar with thin or empty lower bands.\n"
        "    Warn  — Visible bars across 20-70%, meaning widespread uncertainty.\n"
        "    Bad   — A flat or spread-out distribution, or a significant 0-10% bar."
    )
    print("-" * 60)
    buckets = [(i, i + 10) for i in range(0, 100, 10)]
    bar_max_width = 40
    bucket_tokens = [[] for _ in buckets]
    for prob, margin, tok, runner, runner_prob in all_probs:
        idx = min(int(prob // 10), 9)
        bucket_tokens[idx].append((prob, margin, tok, runner))

    for (lo, hi), tokens in zip(buckets, bucket_tokens):
        count = len(tokens)
        bar_width = int((count / token_count) * bar_max_width)
        bar = "█" * bar_width
        pct = (count / token_count) * 100
        example = repr(min(tokens, key=lambda x: x[0])[2]) if tokens else ""
        print(f"  {lo:3d}-{hi:3d}%  {bar:<{bar_max_width}}  {count:3d} tokens ({pct:4.1f}%)  {example}")
    print("-" * 60)

    # Post-histogram interpretation based on the actual distribution
    high_pct = (len(bucket_tokens[9]) / token_count) * 100
    low_pct  = sum(len(bucket_tokens[i]) for i in range(5)) / token_count * 100
    if high_pct >= 70:
        shape_comment = f"  ✅ {high_pct:.0f}% of tokens are in the 90-100% band — the model was largely certain."
    elif high_pct >= 50:
        shape_comment = f"  🟡 {high_pct:.0f}% of tokens are in the 90-100% band — moderate confidence overall."
    else:
        shape_comment = f"  ⚠️  Only {high_pct:.0f}% of tokens are in the 90-100% band — confidence is broadly low."
    print(shape_comment)
    if low_pct > 10:
        print(f"  ⚠️  {low_pct:.0f}% of tokens are below 50% confidence — investigate the example tokens above.")

    # Token Gap Chart — sentence progression column chart
    print(
        "\nTOKEN GAP CHART — sentence progression\n"
        "  A column chart showing chosen vs runner-up probability for each token\n"
        "  in generation order. Each column represents one token.\n"
        "\n"
        "  ANATOMY OF A COLUMN:\n"
        "    The Y-axis represents probability from 0% (bottom) to 100% (top).\n"
        "    Each column is divided into three zones:\n"
        "\n"
        "    ▓▓▓▓  Upper block (▓)  — Chosen token probability. Fills DOWN from\n"
        "    ▓▓▓▓                     100%. A tall ▓ block means the model gave\n"
        "                             high probability to its chosen token.\n"
        "      ↕   White gap        — The MARGIN between chosen and runner-up.\n"
        "          (empty space)      This is the key visual: it shows how\n"
        "                             decisive the model was. A wide gap means\n"
        "                             the chosen token dominated. A narrow gap\n"
        "                             means the runner-up was close behind.\n"
        "    ░░░░  Lower block (░)  — Runner-up token probability. Fills UP from\n"
        "    ░░░░                     0%. A tall ░ block means the second-best\n"
        "                             token was also quite probable.\n"
        "\n"
        "  HOW TO READ THE CHART:\n"
        "    • Scan LEFT to RIGHT to follow the sentence as it was generated.\n"
        "    • Watch the white gap — it should stay wide for a confident response.\n"
        "    • Wide gap (▓ and ░ far apart):\n"
        "        The model was decisive. It strongly preferred one token over all\n"
        "        alternatives. Low hallucination risk at this position.\n"
        "    • Narrow gap (▓ and ░ nearly touching):\n"
        "        The model was torn between two options. It nearly chose something\n"
        "        different. If this falls on a factual word, treat it as suspect.\n"
        "    • No gap (▓ and ░ merge):\n"
        "        A near coin-flip. The chosen and runner-up tokens had almost\n"
        "        identical probability. The response could easily have diverged.\n"
        "    • Both blocks short (▓ small at top, ░ small at bottom, wide gap):\n"
        "        Neither the chosen nor the runner-up was very probable. The model\n"
        "        spread probability across many tokens — it was unsure in general,\n"
        "        even though the gap between first and second place was large.\n"
        "\n"
        "  WHAT TO LOOK FOR:\n"
        "    • Columns where the gap narrows or vanishes indicate uncertainty.\n"
        "    • Clusters of narrow-gap columns = sustained indecision — the model\n"
        "      was guessing through an entire phrase, not just one token.\n"
        "    • A sudden squeeze on a factual word (name, date, number) is a red\n"
        "      flag — these are the tokens most likely to be hallucinated.\n"
        "    • A wide gap following a narrow gap may indicate a snowball: the\n"
        "      model committed to an uncertain choice and then became confidently\n"
        "      wrong (see the Snowball Effect Analysis section below).\n"
        "\n"
        "  BELOW EACH CHART:\n"
        "    Row 1 — Token index number (position in the generated sequence).\n"
        "    Row 2 — The actual token text (truncated if long).\n"
        "    Row 3 — The margin percentage. ⚠ flags margins below 5%."
    )
    print("-" * 80)

    CHART_HEIGHT = 20       # rows in the chart
    COL_WIDTH    = 4        # characters per token column
    MAX_COLS     = 16       # tokens per chart row before wrapping

    for chunk_start in range(0, len(all_probs), MAX_COLS):
        chunk = all_probs[chunk_start:chunk_start + MAX_COLS]
        num_cols = len(chunk)

        # Print chart rows top-to-bottom
        # Each row is a 5% band. Filled ▓ if in chosen zone (top), ░ if in
        # runner-up zone (bottom), space for the gap between them.
        for row in range(CHART_HEIGHT):
            level    = 100.0 * (CHART_HEIGHT - row) / CHART_HEIGHT
            level_lo = 100.0 * (CHART_HEIGHT - row - 1) / CHART_HEIGHT

            if row % 4 == 0:
                label = f"{level:3.0f}% "
            else:
                label = "     "

            line = label + "│"
            for prob, margin, tok, runner, runner_prob in chunk:
                chosen_pct = prob
                runner_pct = runner_prob if runner_prob is not None else 0.0

                # Chosen zone: top of chart down to (100 - chosen_pct)
                # Runner zone: bottom of chart up to runner_pct
                # Gap: everything in between
                chosen_floor = 100.0 - chosen_pct
                if level_lo >= chosen_floor:
                    cell = "▓" * COL_WIDTH
                elif level <= runner_pct:
                    cell = "░" * COL_WIDTH
                else:
                    cell = " " * COL_WIDTH

                line += cell
            print(line)

        # Bottom axis
        print(label_blank := "     " + "│" + "─" * (num_cols * COL_WIDTH))

        # Token index labels
        idx_line = "     " + " "
        for i in range(num_cols):
            idx_line += f"{chunk_start + i:<{COL_WIDTH}d}"
        print(idx_line)

        # Token text labels (truncated)
        tok_line = "     " + " "
        for prob, margin, tok, runner, runner_prob in chunk:
            display = tok.strip() or "·"
            if len(display) > COL_WIDTH - 1:
                display = display[:COL_WIDTH - 1]
            tok_line += f"{display:<{COL_WIDTH}}"
        print(tok_line)

        # Margin summary line
        margin_line = "     " + " "
        for prob, margin_val, tok, runner, runner_prob in chunk:
            if margin_val is not None:
                if margin_val < 5.0:
                    m_str = f"⚠{margin_val:.0f}"
                else:
                    m_str = f"{margin_val:.0f}%"
            else:
                m_str = "n/a"
            margin_line += f"{m_str:<{COL_WIDTH}}"
        print(margin_line)

        if chunk_start + MAX_COLS < len(all_probs):
            print()  # blank line between chunks

    print("-" * 80)

    # Post-chart interpretation
    low_gap_tokens = [(i, prob, margin, tok, runner) for i, (prob, margin, tok, runner, runner_prob) in enumerate(all_probs) if margin is not None and margin < 10.0]
    if low_gap_tokens:
        print(f"  ⚠️  {len(low_gap_tokens)} token(s) had a gap below 10% — the model was indecisive at these points.")
        # Identify runs of consecutive low-gap tokens
        runs = []
        current_run = [low_gap_tokens[0]]
        for item in low_gap_tokens[1:]:
            if item[0] == current_run[-1][0] + 1:
                current_run.append(item)
            else:
                if len(current_run) >= 2:
                    runs.append(current_run)
                current_run = [item]
        if len(current_run) >= 2:
            runs.append(current_run)
        if runs:
            print(f"  ⚠️  {len(runs)} cluster(s) of consecutive low-gap tokens detected:")
            for run in runs:
                start_idx = run[0][0]
                end_idx = run[-1][0]
                cluster_text = "".join(r[3] for r in run)
                print(f"      Tokens {start_idx}-{end_idx}: {repr(cluster_text)}")
    else:
        print("  ✅ All tokens had a gap of 10% or more — consistently decisive choices.")

    # Chart interpretation summary
    # Compute stats for the interpretation
    margins_for_chart = [m for _, m, _, _, _ in all_probs if m is not None]
    avg_margin = sum(margins_for_chart) / len(margins_for_chart) if margins_for_chart else 0.0
    narrow_count = sum(1 for m in margins_for_chart if m < 10.0)
    coinflip_count = sum(1 for m in margins_for_chart if m < 5.0)
    wide_count = sum(1 for m in margins_for_chart if m >= 50.0)

    print(
        "\n  CHART INTERPRETATION SUMMARY\n"
        f"    Average margin across all tokens:  {avg_margin:.1f}%\n"
        f"    Tokens with wide gap    (≥50%):    {wide_count:3d}  — decisive, high trust\n"
        f"    Tokens with narrow gap  (<10%):    {narrow_count:3d}  — indecisive, verify these\n"
        f"    Tokens with coin-flip   (< 5%):    {coinflip_count:3d}  — nearly random, high risk"
    )
    if avg_margin >= 50.0:
        print("    ✅ Overall: The model was highly decisive throughout this response.")
    elif avg_margin >= 25.0:
        print("    🟡 Overall: Moderate decisiveness. Some tokens warrant review.")
    else:
        print("    ⚠️  Overall: Low average margin — the model was frequently uncertain.")
    if coinflip_count > 0:
        print(
            f"    ⚠️  {coinflip_count} coin-flip token(s) detected. Cross-reference these with\n"
            "       the Narrowest Margins table below to identify which words were guesses."
        )
    print("=" * 80)

    # Decision Margin histogram (by margin between chosen and runner-up)
    margins_available = [(prob, margin, tok, runner, runner_prob) for prob, margin, tok, runner, runner_prob in all_probs if margin is not None]
    if margins_available:
        print(
            "\nDECISION MARGIN DISTRIBUTION  (chosen prob − runner-up prob)\n"
            "  This histogram shows the GAP between what the model chose and its\n"
            "  closest alternative — not how confident it was, but how decisive.\n"
            "  A token can be low-probability yet still decisive (no close rival).\n"
            "  A token can be high-probability yet still a near coin-flip.\n"
            "\n"
            "  HOW TO READ THIS:\n"
            "    90-100%  ████  — Landslide decisions. The chosen token was the\n"
            "                     only real option. Very low hallucination risk.\n"
            "    10- 90%  ███   — Clear preference but alternatives existed.\n"
            "                     Normal for ambiguous phrasing choices.\n"
            "     0- 10%  █     — Near coin-flip. The model almost chose something\n"
            "                     else. High risk if on a factual word.\n"
            "\n"
            "  WHAT TO LOOK FOR:\n"
            "    Good  — Most mass in the 70-100% range (decisive choices).\n"
            "    Warn  — A visible 0-10% bar, especially if the example tokens\n"
            "             shown are dates, names, numbers, or key facts.\n"
            "    Bad   — A large 0-10% bar combined with a low Weakest Link score.\n"
            "             The Narrowest Margins table below will identify the culprits."
        )
        print("-" * 60)
        margin_buckets = [[] for _ in buckets]
        for prob, margin, tok, runner, runner_prob in margins_available:
            idx = min(int(margin // 10), 9)
            margin_buckets[idx].append((margin, prob, tok, runner))

        for (lo, hi), tokens in zip(buckets, margin_buckets):
            count = len(tokens)
            bar_width = int((count / token_count) * bar_max_width)
            bar = "█" * bar_width
            pct = (count / token_count) * 100
            example = repr(min(tokens, key=lambda x: x[0])[2]) if tokens else ""
            print(f"  {lo:3d}-{hi:3d}%  {bar:<{bar_max_width}}  {count:3d} tokens ({pct:4.1f}%)  {example}")
        print("-" * 60)

        # Post-histogram interpretation based on the actual margin distribution
        coin_flip_count = len(margin_buckets[0])
        coin_flip_pct   = (coin_flip_count / token_count) * 100
        decisive_count  = sum(len(margin_buckets[i]) for i in range(7, 10))
        decisive_pct    = (decisive_count / token_count) * 100
        if coin_flip_pct > 10:
            print(f"  ⚠️  {coin_flip_pct:.0f}% of tokens had a margin below 10% — many near coin-flip decisions.")
            print( "      Check the Narrowest Margins table to see if any fall on factual words.")
        else:
            print(f"  ✅ Only {coin_flip_pct:.0f}% of tokens had a margin below 10% — most choices were decisive.")
        if decisive_pct >= 50:
            print(f"  ✅ {decisive_pct:.0f}% of tokens had a margin above 70% — strong decisiveness overall.")

    # Bottom 10 narrowest-margin tokens
    print(
        "\nNARROWEST MARGINS — closest decisions (top 10)\n"
        "  These are the tokens where the model most nearly chose something else.\n"
        "  Chosen = probability of the token actually generated.\n"
        "  Runner = probability of the next-best alternative the model considered.\n"
        "  Margin = the gap between them. Small margins on factual words (dates,\n"
        "  names, numbers) are the strongest hallucination signal in this report.\n"
        "  ⚠️  flags margins below 5% — effectively a coin flip."
    )
    print("-" * 60)
    print(f"  {'Chosen':>6}  {'Runner':>6}  {'Margin':>6}  {'Chosen token':<20}  Runner-up token")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}  {'-'*20}")
    for prob, margin, tok, runner, runner_prob in sorted(margins_available, key=lambda x: x[1])[:10]:
        flag = " ⚠️" if margin < 5.0 else ""
        print(f"  {prob:5.1f}%  {runner_prob:5.1f}%  {margin:5.1f}%  {repr(tok):<20}  {repr(runner)}{flag}")
    print("="*60)

    # Snowball Effect Analysis — snowballs already computed in calculation section above
    print(
        "\nSNOWBALL EFFECT ANALYSIS\n"
        "  A 'snowball' is a pivot token — one the model was uncertain about —\n"
        "  immediately followed by a run of high-confidence tokens. This matters\n"
        "  because the model has locked in on its uncertain choice and is now\n"
        "  confidently building on it. The text after the pivot may be fluent\n"
        "  and certain while still being wrong.\n"
        "\n"
        "  HOW TO READ THIS:\n"
        "    Pivot    — the uncertain token where the path forked.\n"
        "    Chosen%  — confidence in the pivot token itself.\n"
        "    Margin   — gap to the runner-up (small = near coin-flip).\n"
        "    PostAvg  — average confidence of the next 8 tokens.\n"
        "    Lift     — how much PostAvg exceeds the response average.\n"
        "               A high lift means the model became MORE certain after\n"
        "               an uncertain choice — the classic snowball signature.\n"
        "    Context  — the tokens that immediately followed the pivot.\n"
        "\n"
        "  WHAT TO LOOK FOR:\n"
        "    High lift + low chosen% on a factual word = highest risk.\n"
        "    The model guessed, then committed. Verify that specific claim."
    )
    print("-" * 60)

    if not snowballs:
        print("  ✅ No snowball patterns detected.")
    else:
        print(f"  ⚠️  {len(snowballs)} potential snowball(s) found — review the pivot tokens below.\n")
        print(f"  {'Pos':>4}  {'Chosen%':>7}  {'Margin':>7}  {'PostAvg':>7}  {'Lift':>6}  Pivot token → Context")
        print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*40}")
        for lift, pos, prob, margin, tok, runner, runner_prob, post_avg, post_text in sorted(snowballs, key=lambda x: -x[0]):
            margin_str = f"{margin:6.1f}%" if margin is not None else "   n/a "
            flag = " ⚠️" if (margin is not None and margin < 5.0) or prob < 20.0 else ""
            context = repr(post_text[:40])
            print(f"  {pos:4d}  {prob:6.1f}%  {margin_str}  {post_avg:6.1f}%  {lift:+5.1f}%  {repr(tok)} → {context}{flag}")
    print("="*60)

    # ── Generate self-contained HTML file ───────────────────────────────
    html_filename = "logprobs_output.html"
    _generate_html_report(html_filename, prompt, full_text.strip(), all_probs,
                          sequence_confidence, adjusted_confidence,
                          min_prob, weakest_token, risk_density, risk_threshold,
                          min_margin, min_margin_chosen, min_margin_runner,
                          snowballs)
    print(f"\n  HTML report saved to: {html_filename}")


def _generate_html_report(filename, prompt, generated_text, all_probs,
                          sequence_confidence, adjusted_confidence,
                          min_prob, weakest_token, risk_density, risk_threshold,
                          min_margin, min_margin_chosen, min_margin_runner,
                          snowballs):
    """Write a self-contained HTML file visualising the token gap chart."""
    import html as html_mod
    import json

    # Prepare token data for JS
    tokens_js = []
    for i, (prob, margin, tok, runner, runner_prob) in enumerate(all_probs):
        tokens_js.append({
            "index": i,
            "token": tok,
            "chosen_prob": round(prob, 2),
            "runner_up_prob": round(runner_prob, 2) if runner_prob is not None else 0,
            "runner_token": runner,
            "margin": round(margin, 2) if margin is not None else None,
        })

    snowballs_js = []
    for lift, pos, prob, margin, tok, runner, runner_prob, post_avg, post_text in snowballs:
        snowballs_js.append({
            "pos": pos,
            "prob": round(prob, 2),
            "margin": round(margin, 2) if margin is not None else None,
            "post_avg": round(post_avg, 2),
            "lift": round(lift, 2),
            "token": tok,
            "context": post_text[:40],
        })

    tokens_json = json.dumps(tokens_js)
    snowballs_json = json.dumps(snowballs_js)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Logprobs Confidence Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; font-size: 1.4em; }}
  h2 {{ color: #58a6ff; margin: 24px 0 12px; font-size: 1.15em; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  .prompt {{ color: #8b949e; font-style: italic; margin-bottom: 16px; font-size: 0.9em; }}
  .generated {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-bottom: 16px; font-size: 0.9em; white-space: pre-wrap; }}
  .scorecard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }}
  .metric {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; }}
  .metric .label {{ color: #8b949e; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric .value {{ font-size: 1.5em; font-weight: 700; margin-top: 4px; }}
  .metric .detail {{ color: #8b949e; font-size: 0.78em; margin-top: 4px; }}
  .pass {{ color: #3fb950; }}
  .warn {{ color: #d29922; }}
  .fail {{ color: #f85149; }}

  /* Chart container */
  .chart-wrapper {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin-bottom: 16px; }}
  .chart-scroll {{ overflow-x: auto; overflow-y: hidden; padding-bottom: 8px; }}
  .chart-scroll::-webkit-scrollbar {{ height: 8px; }}
  .chart-scroll::-webkit-scrollbar-track {{ background: #21262d; border-radius: 4px; }}
  .chart-scroll::-webkit-scrollbar-thumb {{ background: #484f58; border-radius: 4px; }}
  .chart-scroll::-webkit-scrollbar-thumb:hover {{ background: #6e7681; }}

  .chart-area {{ display: flex; align-items: flex-end; position: relative; }}
  .y-axis {{ display: flex; flex-direction: column; justify-content: space-between; height: 300px; margin-right: 4px; flex-shrink: 0; }}
  .y-axis span {{ font-size: 10px; color: #484f58; text-align: right; min-width: 32px; }}
  .columns-container {{ display: flex; align-items: stretch; height: 300px; position: relative; }}

  .token-col {{ display: flex; flex-direction: column; align-items: center; width: 38px; min-width: 38px; flex-shrink: 0; position: relative; height: 100%; cursor: pointer; }}
  .token-col:hover {{ background: rgba(88,166,255,0.06); }}
  .token-col .bar-area {{ width: 100%; height: 100%; position: relative; }}
  .token-col .chosen-bar {{ position: absolute; top: 0; left: 4px; right: 4px; background: #3fb950; border-radius: 2px 2px 0 0; }}
  .token-col .runner-bar {{ position: absolute; bottom: 0; left: 4px; right: 4px; background: #f0883e; border-radius: 0 0 2px 2px; }}
  .token-col .snowball-marker {{ position: absolute; top: -2px; left: 50%; transform: translateX(-50%); width: 8px; height: 8px; background: #f85149; border-radius: 50%; z-index: 2; }}

  .labels-row {{ display: flex; }}
  .labels-row .cell {{ width: 38px; min-width: 38px; flex-shrink: 0; text-align: center; font-size: 9px; padding: 2px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .labels-row.index-row .cell {{ color: #484f58; }}
  .labels-row.token-row .cell {{ color: #c9d1d9; font-family: monospace; }}
  .labels-row.margin-row .cell {{ color: #8b949e; }}
  .labels-row.margin-row .cell.coin-flip {{ color: #f85149; font-weight: 700; }}

  /* Tooltip */
  .tooltip {{ display: none; position: fixed; background: #1c2128; border: 1px solid #444c56; border-radius: 6px; padding: 10px 14px; font-size: 12px; z-index: 1000; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.4); max-width: 280px; }}
  .tooltip .tt-token {{ font-family: monospace; font-size: 14px; color: #58a6ff; margin-bottom: 4px; }}
  .tooltip .tt-row {{ display: flex; justify-content: space-between; gap: 16px; margin-top: 2px; }}
  .tooltip .tt-label {{ color: #8b949e; }}
  .tooltip .tt-value {{ font-weight: 600; }}

  /* Legend */
  .legend {{ display: flex; gap: 20px; margin: 8px 0 4px; font-size: 0.8em; color: #8b949e; }}
  .legend span {{ display: flex; align-items: center; gap: 5px; }}
  .legend .swatch {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
  .swatch.chosen {{ background: #3fb950; }}
  .swatch.runner {{ background: #f0883e; }}
  .swatch.snowball {{ background: #f85149; border-radius: 50%; width: 10px; height: 10px; }}

  /* Snowball table */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  th {{ text-align: left; color: #8b949e; font-weight: 600; padding: 6px 10px; border-bottom: 1px solid #30363d; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #161b22; }}
</style>
</head>
<body>

<h1>Logprobs Confidence Report</h1>
<div class="prompt">Prompt: {html_mod.escape(prompt)}</div>
<div class="generated">{html_mod.escape(generated_text)}</div>

<div class="scorecard">
  <div class="metric">
    <div class="label">Sequence Score (Raw &rarr; Adjusted)</div>
    <div class="value">{sequence_confidence:.1f}% &rarr; {adjusted_confidence:.1f}%</div>
    <div class="detail">Delta: {adjusted_confidence - sequence_confidence:+.1f}%</div>
  </div>
  <div class="metric">
    <div class="label">Weakest Link</div>
    <div class="value {'fail' if min_prob < risk_threshold else 'pass'}">{min_prob:.1f}%</div>
    <div class="detail">Token: {html_mod.escape(repr(weakest_token))}</div>
  </div>
  <div class="metric">
    <div class="label">Risk Density</div>
    <div class="value {'fail' if risk_density > 20 else 'warn' if risk_density > 10 else 'pass'}">{risk_density:.1f}%</div>
    <div class="detail">Tokens below {risk_threshold:.0f}% threshold</div>
  </div>
  <div class="metric">
    <div class="label">Narrowest Margin</div>
    <div class="value {'fail' if min_margin < 5 else 'warn' if min_margin < 15 else 'pass'}">{min_margin:.1f}%</div>
    <div class="detail">{html_mod.escape(repr(min_margin_chosen))} vs {html_mod.escape(repr(min_margin_runner))}</div>
  </div>
</div>

<h2>Token Gap Chart</h2>
<div class="chart-wrapper">
  <div class="legend">
    <span><span class="swatch chosen"></span> Chosen token probability</span>
    <span><span class="swatch runner"></span> Runner-up probability</span>
    <span><span class="swatch snowball"></span> Snowball pivot</span>
    <span style="color:#484f58;">Gap = white space between bars</span>
  </div>
  <div class="chart-scroll" id="chartScroll">
    <div class="chart-area">
      <div class="y-axis" id="yAxis"></div>
      <div class="columns-container" id="columns"></div>
    </div>
    <div class="labels-row index-row" id="indexRow"></div>
    <div class="labels-row token-row" id="tokenRow"></div>
    <div class="labels-row margin-row" id="marginRow"></div>
  </div>
</div>

<div id="snowballSection"></div>

<div class="tooltip" id="tooltip"></div>

<script>
const TOKENS = {tokens_json};
const SNOWBALLS = {snowballs_json};
const CHART_H = 300;
const snowballPositions = new Set(SNOWBALLS.map(s => s.pos));

// Y-axis
const yAxis = document.getElementById('yAxis');
for (let i = 100; i >= 0; i -= 10) {{
  const s = document.createElement('span');
  s.textContent = i + '%';
  yAxis.appendChild(s);
}}

const colsEl = document.getElementById('columns');
const indexRow = document.getElementById('indexRow');
const tokenRow = document.getElementById('tokenRow');
const marginRow = document.getElementById('marginRow');

// Add left padding to label rows to match y-axis
[indexRow, tokenRow, marginRow].forEach(row => {{
  row.style.marginLeft = '36px';
}});

TOKENS.forEach((t, i) => {{
  // Column
  const col = document.createElement('div');
  col.className = 'token-col';
  const barArea = document.createElement('div');
  barArea.className = 'bar-area';

  // Chosen bar: fills from top down
  const chosenBar = document.createElement('div');
  chosenBar.className = 'chosen-bar';
  chosenBar.style.height = (t.chosen_prob / 100 * CHART_H) + 'px';
  barArea.appendChild(chosenBar);

  // Runner bar: fills from bottom up
  const runnerBar = document.createElement('div');
  runnerBar.className = 'runner-bar';
  runnerBar.style.height = (t.runner_up_prob / 100 * CHART_H) + 'px';
  barArea.appendChild(runnerBar);

  // Snowball marker
  if (snowballPositions.has(i)) {{
    const marker = document.createElement('div');
    marker.className = 'snowball-marker';
    barArea.appendChild(marker);
  }}

  col.appendChild(barArea);
  colsEl.appendChild(col);

  // Tooltip events
  col.addEventListener('mouseenter', (e) => {{
    const tt = document.getElementById('tooltip');
    const marginStr = t.margin !== null ? t.margin.toFixed(1) + '%' : 'n/a';
    const snowStr = snowballPositions.has(i) ? '<div style="color:#f85149;margin-top:4px;">Snowball pivot</div>' : '';
    tt.innerHTML = '<div class="tt-token">' + escapeHtml(t.token) + '</div>'
      + '<div class="tt-row"><span class="tt-label">Chosen:</span><span class="tt-value" style="color:#3fb950">' + t.chosen_prob + '%</span></div>'
      + '<div class="tt-row"><span class="tt-label">Runner (' + escapeHtml(t.runner_token) + '):</span><span class="tt-value" style="color:#f0883e">' + t.runner_up_prob + '%</span></div>'
      + '<div class="tt-row"><span class="tt-label">Margin:</span><span class="tt-value">' + marginStr + '</span></div>'
      + snowStr;
    tt.style.display = 'block';
  }});
  col.addEventListener('mousemove', (e) => {{
    const tt = document.getElementById('tooltip');
    tt.style.left = (e.clientX + 12) + 'px';
    tt.style.top = (e.clientY - 10) + 'px';
  }});
  col.addEventListener('mouseleave', () => {{
    document.getElementById('tooltip').style.display = 'none';
  }});

  // Index label
  const idxCell = document.createElement('div');
  idxCell.className = 'cell';
  idxCell.textContent = i;
  indexRow.appendChild(idxCell);

  // Token label
  const tokCell = document.createElement('div');
  tokCell.className = 'cell';
  tokCell.title = t.token;
  const display = t.token.trim() || '\u00b7';
  tokCell.textContent = display.length > 4 ? display.slice(0, 4) : display;
  tokenRow.appendChild(tokCell);

  // Margin label
  const mCell = document.createElement('div');
  mCell.className = 'cell';
  if (t.margin !== null) {{
    if (t.margin < 5) {{
      mCell.textContent = t.margin.toFixed(0) + '%';
      mCell.classList.add('coin-flip');
    }} else {{
      mCell.textContent = t.margin.toFixed(0) + '%';
    }}
  }} else {{
    mCell.textContent = '-';
  }}
  marginRow.appendChild(mCell);
}});

// Snowball section
if (SNOWBALLS.length > 0) {{
  const sec = document.getElementById('snowballSection');
  let html = '<h2>Snowball Effect Analysis</h2><div class="chart-wrapper"><table><tr><th>Pos</th><th>Pivot</th><th>Chosen%</th><th>Margin</th><th>PostAvg</th><th>Lift</th><th>Context</th></tr>';
  SNOWBALLS.sort((a, b) => b.lift - a.lift).forEach(s => {{
    const marginStr = s.margin !== null ? s.margin.toFixed(1) + '%' : 'n/a';
    html += '<tr><td>' + s.pos + '</td><td style="font-family:monospace;color:#58a6ff">' + escapeHtml(s.token) + '</td><td>' + s.prob.toFixed(1) + '%</td><td>' + marginStr + '</td><td>' + s.post_avg.toFixed(1) + '%</td><td style="color:#f85149">+' + s.lift.toFixed(1) + '%</td><td style="font-family:monospace;font-size:11px">' + escapeHtml(s.context) + '</td></tr>';
  }});
  html += '</table></div>';
  sec.innerHTML = html;
}}

function escapeHtml(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}
</script>
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)


# Let's test it with a prompt that requires specific retrieval
test_prompt = "What was the exact date and location of the signing of the Treaty of Waitangi?"
generate_confidence_scorecard(test_prompt)
