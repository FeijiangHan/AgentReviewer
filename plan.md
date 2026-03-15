# Implementation Plan for Next-Gen Agent Reviewer

## 0) Goal, Scope, and Assumptions

## Goal
Build a production-grade **Agent Review pipeline** that:
1. Removes the existing Elo mechanics and keeps only review + evaluation.
2. Adds OpenAI GPT API key/model support in addition to current Gemini support.
3. Supports local PDF upload → parsing → reviewing → structured output end-to-end.
4. Implements **Dynamic Persona Agent Reviewer** from real-world reviewer candidates mined from references.
5. Compares three settings: fixed persona, dynamic persona, and no persona.

## Scope in this plan
This document is an execution blueprint (not code yet), covering architecture, APIs, modules, milestones, experiments, and risk controls.

## Assumptions
- The current repository structure (`main.py`, `role/`, `prompt/`) remains the baseline and will be extended incrementally.
- Network access is available for online search/enrichment in the dynamic persona pipeline.
- We can add lightweight dependencies if necessary, but should prefer minimal additions and composable adapters.

---

## 1) Current System Diagnosis (What must change)

The current project is simulation-centric with these constraints:
- `SimulationManager` orchestrates rounds and **Elo updates** tightly coupled to reviewer evaluation.
- Reviewer generation depends on Gemini file upload flow (`role/reviewer.py`).
- AC logic (`role/area_chair.py`) currently outputs both decision and reviewer-quality scoring, partly for Elo ranking.
- Input source is remote paper URLs from `papers.json`, not local uploaded PDF pipeline.

### Required refactor direction
- Decouple review generation/evaluation from ranking mechanics.
- Introduce model-provider abstraction layer.
- Introduce document ingestion layer (local file first-class).
- Add dynamic persona discovery + synthesis + reviewer instantiation pipeline.
- Add experiment harness for 3 reviewer conditions.

---

## 2) Target Architecture (High-level)

Create a modular pipeline with the following layers:

1. **Ingestion Layer**
   - Input: local PDF path
   - Output: normalized `PaperContext` (metadata, extracted text, references)

2. **Reviewer Candidate Discovery Layer** (for dynamic persona mode)
   - Input: references list
   - Output: candidate reviewer pool with identity confidence

3. **Persona Construction Layer**
   - Input: reviewer identities
   - Output: structured persona profiles (expertise, style, bias priors)

4. **Review Execution Layer**
   - Supports three modes:
     - `no_persona`
     - `fixed_persona`
     - `dynamic_persona`
   - Runs k reviewer agents and optional AC/meta-evaluator.

5. **Evaluation Layer**
   - Outputs review quality metrics and cross-mode comparison report.

6. **LLM Provider Layer**
   - Uniform API for Gemini and GPT models.

---

## 3) Detailed Work Plan by Milestone

## Milestone A — Remove Elo; preserve review + evaluation core

### A1. Refactor simulation flow
- Replace Elo-centric `SimulationManager` responsibilities with a review workflow manager, e.g. `ReviewPipelineManager`.
- Remove rank-based reward calculation and memory update triggers tied to Elo deltas.
- Keep:
  - multi-reviewer generation
  - stage-1/stage-2 adjustment (optional toggle)
  - AC final decision/evaluation

### A2. Simplify output schema
- Define a new result schema:
  - paper metadata
  - reviewer outputs (per stage)
  - AC decision/evaluation
  - runtime metadata (provider/model, timestamps)
- Exclude Elo fields from stored results.

### A3. Backward-compatibility strategy
- Keep old Elo code behind a deprecated flag only if needed for historical reproduction (`--legacy-elo`), otherwise delete cleanly.

### Validation for A
- Unit tests for round execution without Elo dependencies.
- Snapshot test for output schema consistency.

---

## Milestone B — Add GPT API support (multi-provider LLM)

### B1. Provider abstraction design
Create `llm/` module:
- `base.py`: `LLMProvider` interface
  - `generate_json(system_prompt, user_prompt, schema)`
  - `generate_text(system_prompt, user_prompt)`
  - optional `analyze_document(...)`
- `gemini_provider.py`
- `openai_provider.py`

### B2. Configuration model
- Add runtime config object (CLI/env/file):
  - `LLM_PROVIDER` (`gemini` | `openai`)
  - `LLM_MODEL`
  - `GEMINI_API_KEY` / `OPENAI_API_KEY`
- Never hardcode secrets in code.

### B3. JSON-schema-normalized outputs
- Ensure both providers return schema-valid JSON for reviewer and AC outputs.
- Add retry/repair policy for malformed JSON (single standardized parser).

### Candidate APIs for GPT support
- **OpenAI Responses API** (preferred): structured generation, tool-friendly flows.
- Fallback: Chat Completions for compatibility.

### Validation for B
- Mocked provider tests for schema compliance.
- Integration smoke tests for both providers (if keys available).

---

## Milestone C — Local PDF upload + parse + review full pipeline

### C1. Ingestion module
Create `pipeline/ingest.py`:
- Input: local path(s) to PDF
- Basic checks: exists, readable, file size limit, MIME check
- Output object:
  - `paper_id`
  - `title` (if detected)
  - `full_text`
  - `references` (raw)

### C2. PDF parsing strategy
Use layered fallback:
1. `pypdf` / `pdfplumber` text extraction
2. optional structured parser (e.g., GROBID service) for references if needed

### C3. Reference extraction
- Initial rule-based parser for bibliography section boundaries.
- Extract entries into normalized reference candidates (title guess, author string, venue/year when available).

### C4. Review runner for local PDFs
- Replace URL upload dependency in reviewer agent with context-based prompting:
  - either pass extracted text chunks
  - or provider-specific file upload API abstraction if retained
- Implement chunking strategy for long papers (map-reduce summarization then review).

### Output artifacts
- `outputs/<paper_id>/reviews.json`
- `outputs/<paper_id>/report.md`

### Validation for C
- Targeted test with one local sample PDF.
- Ensure pipeline can run end-to-end without network (except dynamic persona enrichment).

---

## Milestone D — Dynamic Persona Agent Reviewer

## D1. Candidate reviewer pool from references
Input: parsed references from paper.

Process:
1. For each reference, attempt to resolve:
   - canonical paper title
   - author list (full names)
2. Aggregate all authors as initial candidate pool.

### Suggested external sources/APIs for resolution
- **Crossref API**: DOI/title metadata resolution.
- **OpenAlex API**: works + authors + institutions + concept tags.
- **Semantic Scholar API**: paper-author graph, citation metadata.
- Optional: DBLP (CS-focused authority matching).

## D2. Identity verification and disambiguation
- Build `IdentityResolver` that assigns confidence scores.
- Use multi-signal matching:
  - name + coauthor overlap
  - title similarity
  - affiliation/field consistency
- Keep only candidates above confidence threshold.

## D3. Baseline top-k reviewer matching
Before dynamic persona synthesis, build a baseline retriever:
- Represent submitted paper topic via embeddings.
- Represent candidate expertise via publication corpus embeddings.
- Rank by topical similarity + recency + citation impact proxy.
- Select top-k (e.g., 3 or 5).

### Candidate APIs/services
- Embeddings: OpenAI text-embedding model (or sentence-transformers local fallback).
- Metadata sources: OpenAlex/Semantic Scholar.

## D4. Multi-source background enrichment
For each selected candidate reviewer:
- Collect structured profile signals:
  - personal/homepage bio
  - institutional page
  - publication history
  - topic clusters
  - writing tendencies from abstracts/reviews (if available)

### Source priorities
1. OpenAlex/S2 for publication corpus (high precision)
2. Google Scholar-like sources (if policy-compliant; avoid brittle scraping when possible)
3. LinkedIn/homepage (optional, use only public and policy-safe data)

## D5. Persona synthesis
Create `persona/dynamic_builder.py`:
- Input: reviewer profile bundle
- Output: `PersonaCard` JSON + prompt text

`PersonaCard` schema (proposed):
- `name`, `affiliation`, `research_areas`
- `methodological_preferences`
- `common_concerns` (novelty, rigor, ablation, reproducibility, etc.)
- `style_signature` (direct/constructive/critical)
- `potential_biases` (topic preference, benchmark preference)
- `confidence` and `evidence_sources`

## D6. Persona-conditioned review execution
- Spawn one reviewer agent per selected persona.
- Inject persona card into system prompt + keep review policy constraints.
- Produce per-persona reviews and aggregate outcomes.

### Critical safety controls
- Avoid defamation or personal sensitive inferences.
- Restrict persona to professional/publicly available scholarly dimensions.
- Mark all persona attributes as probabilistic, evidence-linked.

### Validation for D
- Unit tests for reference→author extraction.
- Integration test for one paper producing k dynamic personas + reviews.
- Manual audit checklist for persona factuality and harmlessness.

---

## Milestone E — Comparative experiment: fixed vs dynamic vs no persona

## E1. Experiment design
Three arms:
1. **No Persona**: generic reviewer prompt only.
2. **Fixed Persona**: existing handcrafted persona files.
3. **Dynamic Persona**: automatically generated persona cards from references.

Run each arm on the same paper set with controlled randomness.

## E2. Metrics
### Review quality metrics
- AC/meta-evaluator score (structured rubric)
- Specificity score (evidence density, section grounding)
- Consistency score (claim-score alignment)

### Decision-level metrics
- Accept/Reject alignment against available labels (if any)
- Inter-reviewer agreement/disagreement quality

### Process metrics
- Latency and token cost
- Persona construction success rate
- External API coverage/failure rate

## E3. Statistical analysis
- Paired comparisons across the same papers.
- Bootstrap confidence intervals for quality deltas.
- Error analysis on failure categories (hallucinated persona, shallow critique, etc.).

### Validation for E
- Reproducible experiment script `experiments/run_comparison.py`
- Auto-generated `comparison_report.md` with tables/plots.

---

## 4) Suggested New Modules and File Layout

Proposed additive structure:

- `llm/`
  - `base.py`
  - `gemini_provider.py`
  - `openai_provider.py`
- `pipeline/`
  - `ingest.py`
  - `pdf_parser.py`
  - `reference_extractor.py`
  - `review_runner.py`
- `persona/`
  - `fixed/` (existing prompt persona migration)
  - `dynamic_builder.py`
  - `identity_resolver.py`
  - `reviewer_matcher.py`
- `retrieval/`
  - `crossref_client.py`
  - `openalex_client.py`
  - `semanticscholar_client.py`
- `evaluation/`
  - `metrics.py`
  - `comparison.py`
- `configs/`
  - `default.yaml`
  - `experiments/*.yaml`
- `tests/`
  - unit/integration/e2e groups

---

## 5) API & Dependency Research Checklist

## Must research immediately
1. OpenAI structured output best practices (Responses API + JSON schema).
2. OpenAlex rate limits, pagination, and author/work resolution quality.
3. Semantic Scholar API quotas and fields for author disambiguation.
4. Crossref title lookup reliability for noisy references.

## Candidate dependencies
- Core: `openai`, `google-generativeai`, `pydantic`, `httpx`, `tenacity`
- PDF: `pypdf` or `pdfplumber`
- Optional NLP: `rapidfuzz`, `sentence-transformers`
- Data: `pandas` (optional for reports)

Principle: add only what is required per milestone, avoid dependency spikes.

---

## 6) Step-by-step Build Sequence (Execution Order)

1. **Refactor out Elo** while keeping existing reviewer/AC flow runnable.
2. Add provider abstraction and plug Gemini first (no behavior change expected).
3. Add OpenAI provider and run dual-provider smoke tests.
4. Introduce local PDF ingestion and run single-paper review end-to-end.
5. Add reference extraction and author-pool construction.
6. Add baseline top-k reviewer matching (topic similarity only).
7. Add multi-source enrichment and identity confidence filters.
8. Add dynamic persona synthesis and persona-conditioned review runs.
9. Add three-arm comparison harness and metrics report.
10. Harden with retries/caching/logging and finalize docs.

---

## 7) Engineering Risks and Mitigations

1. **Reference parsing is noisy**
   - Mitigation: confidence thresholds + fallback to partial metadata.

2. **Author identity ambiguity**
   - Mitigation: multi-source resolution + explicit low-confidence exclusion.

3. **Persona hallucination risk**
   - Mitigation: evidence-linked persona card fields + conservative templates.

4. **API instability/quota limits**
   - Mitigation: caching layer, exponential backoff, offline fixtures for tests.

5. **Prompt-length/token cost explosion**
   - Mitigation: summarize publication corpus before persona synthesis; enforce caps.

---

## 8) Definition of Done (for this roadmap)

A release is complete when:
- Elo logic is removed from default pipeline.
- Gemini + GPT providers are both supported via config.
- Local PDF input can produce complete reviews and structured outputs.
- Dynamic persona generation works from references with traceable evidence.
- Comparison experiment (no/fixed/dynamic persona) runs reproducibly and outputs a report with quantitative metrics.

---

## 9) Immediate Next Sprint (Concrete TODO)

Week 1 target:
1. Implement Milestone A (Elo removal) and pass regression tests.
2. Implement provider abstraction + OpenAI support (Milestone B).
3. Implement minimal local PDF ingestion/review path (Milestone C core only).
4. Draft API clients with stubs for OpenAlex/Crossref/S2 (Milestone D scaffolding).

Deliverables:
- runnable CLI for local PDF review
- provider switch (`--provider gemini|openai`)
- JSON output schema v2 (no Elo)
- technical design note for dynamic persona phase
