# Agent Reviewer Background Survey

## 1) What this project is and which paper it corresponds to

This repository implements a simulation framework for **LLM-based peer review dynamics** in an Elo-ranked review ecosystem.

The corresponding paper is:

- **Title:** *Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System*
- **Authors:** Hsiang-Wei Huang, Junbin Lu, Kuang-Ming Chen, Jenq-Neng Hwang
- **arXiv:** 2601.08829
- **Core claim (from the paper abstract):** introducing Elo signals can improve Area Chair (AC) decision quality, while reviewer memory can also induce strategic adaptation that does not necessarily reflect deeper review effort.

In short, this project is both a simulation tool and a behavioral testbed: it studies not only paper-level decisions, but also *agent incentives* under ranking pressure.

---

## 2) Full Agent Reviewer pipeline (input → processing → output)

## 2.1 Inputs

The pipeline consumes four major input groups:

1. **Paper pool**
   - Loaded from `papers.json`.
   - Each entry contains:
     - `id`
     - `url` (typically an OpenReview PDF URL)
     - `actual_rating` (used as a ground-truth-like reference label).

2. **Agent instructions (prompts)**
   - Reviewer policy prompt (`prompt/reviewer.txt`) that forces structured JSON outputs.
   - AC policy prompt (`prompt/ac.txt`) that asks for final decision + review quality scoring.
   - Persona prompts (`prompt/persona/*.txt`) defining behavioral styles (bluffer, critic, expert, harmonizer, optimist, skimmer).

3. **Simulation configuration**
   - API key, random seed, number of rounds, papers per round, reviewers per paper.
   - In this repo default path, the manager runs 3 experiments, 30 rounds each, 2 papers/round, 3 reviewers/paper.

4. **LLM backend**
   - Gemini model (`gemini-2.5-flash`) used for reviewer generation, AC decisioning, and memory strategy updates.

## 2.2 Processing flow

The core runtime is orchestrated by `SimulationManager` and can be viewed as a nested loop:

1. **Experiment loop (Mode 1, 2, 3)**
2. **Round loop**
3. **Paper loop (sampled subset each round)**
4. **Reviewer interaction loop (stage 1, stage 2, AC eval, Elo update)**

### Step A: Initialization

- Build Gemini client.
- Initialize six reviewer agents with identical initial Elo (1500) and fixed personas.
- Reset random seeds at each experiment start to preserve comparability between experiment modes.

### Step B: Round assignment

For each round:

- Randomly sample papers (`num_papers_per_round`).
- For each selected paper, randomly sample a 3-reviewer triplet from 6 reviewers.

This creates a dynamic interaction topology where reviewer combinations vary round-to-round.

### Step C: Stage-1 review generation

Each assigned reviewer:

1. Downloads paper PDF from URL.
2. Uploads PDF to Gemini file API.
3. Generates an initial JSON review under persona + policy constraints.

Expected fields include strengths, weaknesses, justification, and discrete score (0/2/4/6/8/10).

### Step D: Stage-2 review adjustment (discussion phase)

Each reviewer then sees the other two reviewers’ stage-1 outputs and generates a revised JSON review.

Important design choice: stage-2 allows adaptation while asking the agent to remain aligned with its core persona.

### Step E: AC final decision and review-quality scoring

The AC consumes all stage-2 reviews and optionally reviewer Elo scores (depending on experiment mode):

- **Mode 1:** AC does **not** see Elo; reviewers do not update memory.
- **Mode 2:** AC **does** see Elo; reviewers do not update memory.
- **Mode 3:** AC sees Elo + reviewers update memory.

AC produces:

- `final_decision` (Accept/Reject)
- `final_justification`
- `review_evaluation`: per-reviewer quality score (0–10, even numbers) and short rationale.

### Step F: Rank-based Elo updates

For the 3 reviewers on each paper, AC quality scores are ranked and mapped to reward buckets:

- 1st: +100
- 2nd: 0
- 3rd: -100

Ties are handled by averaging the reward pool for tied positions.

This is a **relative**, local-competition update, not a classical probabilistic Elo formula.

### Step G: Memory update (Mode 3 only)

Each reviewer receives feedback from Elo delta + previous review text + AC quality score, then asks Gemini to generate a tactical self-instruction (5–10 sentence style target, single paragraph output).

This memory is prepended in future rounds as a self-reflection layer, while instructing the agent not to break persona identity.

### Step H: Logging

The system records per paper and per round:

- experiment mode, round id, paper id
- actual rating
- AC decision object
- all review artifacts (stage1/stage2 text, ratings)
- memory prompt used
- post-update Elo snapshot

After all experiments, results are written to `simulation_results.json`.

## 2.3 Outputs

The pipeline output is a structured, longitudinal behavioral dataset that supports multiple analyses:

1. **Decision-level metrics**
   - AC accept/reject behavior under different information regimes.

2. **Reviewer-level trajectories**
   - Elo evolution over rounds and across experiment modes.

3. **Behavioral adaptation signals**
   - Stage-1 → stage-2 score shifts.
   - Memory prompt drift and potential gaming behavior.

4. **System-level emergent dynamics**
   - Whether incentive design (Elo + visibility + memory) improves decision quality or induces strategic-but-shallow reviewing.

---

## 3) Paper-grounded deeper research and experimental thought process

Below is a structured “researcher’s lens” for interpreting and extending the paper/project.

## 3.1 Research framing

The project effectively asks a mechanism-design question in AI-mediated peer review:

- If we rank reviewers with Elo-like rewards,
- and optionally expose those rankings to a decision-maker,
- and optionally give reviewers memory for self-optimization,

then what happens to **decision quality**, **review quality**, and **agent strategy**?

This is more than prompt engineering; it is an incentive-system simulation.

## 3.2 Why the three-mode ablation is meaningful

The three experiments isolate causal levers:

- **Mode 1 (no Elo to AC, no memory):** baseline social process.
- **Mode 2 (Elo visible to AC):** tests whether AC can use reviewer reliability priors.
- **Mode 3 (Elo visible + memory):** adds reviewer adaptation loop, enabling strategic response.

Interpretation logic:

- Mode 1 → Mode 2 difference estimates the value of reviewer reputation signals for final decision quality.
- Mode 2 → Mode 3 difference estimates the side effects of adaptive strategic behavior under incentive pressure.

## 3.3 Key hypothesis map

A clean hypothesis set (aligned with the paper abstract) is:

- **H1:** Elo visibility improves AC calibration (better final decisions against `actual_rating`).
- **H2:** Memory improves reviewer *measured* performance (Elo), but not always intrinsic review depth.
- **H3:** Persona effects persist under adaptation, but tactical convergence may occur (different personas learn similarly “safe” strategies to avoid Elo loss).

## 3.4 Suggested metrics for rigorous analysis

To deepen the experimental study, compute at least:

1. **Decision alignment metrics**
   - Convert Accept/Reject into binary labels via thresholding `actual_rating`; report accuracy/F1/AUC by mode.

2. **Review quality consistency**
   - Variance of AC review-quality scores by reviewer and mode.

3. **Adaptation-vs-substance indicators**
   - Length, specificity, and evidence density in reviews over time.
   - Correlation between Elo gains and objective text-quality proxies.

4. **Strategic behavior index**
   - Detect whether reviewers shift toward consensus-chasing or Elo-protective hedging (e.g., narrower score band, less contrarian stance).

5. **Fairness/robustness checks**
   - Across personas, test whether some styles are structurally advantaged by AC policy when Elo is visible.

## 3.5 Potential confounds and validity threats

Important caveats when interpreting findings:

- **AC as single-model judge:** AC and reviewers share the same model family, potentially creating style favoritism.
- **Schema-constrained outputs:** enforced JSON improves parseability but can compress nuanced argumentation.
- **Reward sparsity:** +100/0/-100 per triplet is coarse; small AC score differences can cause large Elo jumps.
- **Data dependence:** if paper pool difficulty is imbalanced, random assignment may still induce exposure bias.
- **Ground truth ambiguity:** `actual_rating` is treated as quality anchor, but conference ratings are noisy social outcomes.

## 3.6 Experimental extensions worth running

1. **Cross-model evaluation**
   - Use one model family for reviewers and a different one for AC to reduce same-model bias.

2. **Alternative reward mechanisms**
   - Compare rank-bucket updates against smooth score-proportional updates.

3. **Blinded vs transparent incentives**
   - Test delayed Elo disclosure or partial disclosure to reduce immediate gaming pressure.

4. **Human-in-the-loop audit subset**
   - Sample reviews for manual quality scoring to validate AC review-quality labels.

5. **Counterfactual replay**
   - Keep generated reviews fixed, then recompute outcomes under different AC Elo-visibility settings to isolate decision-policy effects.

## 3.7 Practical takeaway

The strongest conceptual contribution of this project/paper pair is that reviewer evaluation systems are **incentive environments**.

Once memory and ranking pressure are introduced, agents optimize to the reward function. That can improve measurable performance, but it may also reshape behavior in ways that look competent without increasing genuine review effort. Designing robust AI review systems therefore requires both performance metrics and anti-gaming diagnostics.
