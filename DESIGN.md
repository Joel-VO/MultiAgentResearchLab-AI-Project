# Multi-Agent Research Laboratory and Peer-Review Board: System Design

# 1. Overview

The system is organized by research stage rather than by course module. Six modules (A–F) each form an independent unit with fixed input and output formats, and each borrows the course techniques it needs. Modules communicate only through the shared state object and the evidence ledger defined in Module A.

**Modules at a glance**

| **Module** | **Name**                                           | **What it does**                                                                                                                                                                                                                               |
|:-----------|:---------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A          | Tooling and Report Generation                      | Shared infrastructure for every other module: LLM client with schema validation, shared state and evidence ledger, sandboxed code execution, permission-gated dataset download, citation checking, and the final report built from the ledger. |
| B          | Literature, Knowledge Base, and Conflict Detection | Builds the knowledge base from the seed papers by best-first search over the citation graph, writes a structured review of each paper, and flags contradictory findings and reviewer disagreement.                                             |
| C          | Hypothesis Generation                              | Mines gaps in the knowledge base and turns them into scored, testable hypotheses through a multi-agent discussion.                                                                                                                             |
| D          | Workflow Tracker, CSP, and RL Controller           | Controls the process: the tracker decides which steps may run and replans on failure, the CSP turns a hypothesis into a feasible proposal (or rejects it), and the RL controller chooses the next action each round.                           |
| E          | Experimenter and Evaluation Harness                | Designs experiments for a proposal and runs them for real or in simulation; also runs the whole system through the test scenarios.                                                                                                             |
| F          | Reviewer and Belief Model                          | Estimates how likely each hypothesis is to be supported from literature and experiment evidence, ranks the hypotheses, and writes a structured review whose revision tasks feed back into the loop.                                            |

**Pipeline**

```
B (literature, KB, conflicts)
  → C (gap mining → hypotheses)
  → D.CSP (proposal construction)
  → F.Belief (ranking)
  → E (experiment design and execution)
  → F.Reviewer
  → D.RL (choose next action)
  → loop back to E / F / B until a stop condition
  → A.Report
```

D’s workflow tracker gates which step may run at any time. A’s tooling supports every step.

**Course technique to module mapping**

| **Technique**               | **Where it is implemented**                                                       |
|:----------------------------|:----------------------------------------------------------------------------------|
| Search                      | B: best-first search over the citation graph; citation-path queries               |
| Constraint satisfaction     | D: CSP proposal construction                                                      |
| Planning                    | D: workflow tracker (preconditions, effects, replanning)                          |
| Reasoning under uncertainty | F: Beta-Bernoulli belief model                                                    |
| Reinforcement learning      | D: contextual-bandit controller (LinUCB), trained on the LabSim process simulator |
| Generative AI (multi-agent) | C: hypothesis generation; E: experimenter; F: reviewer                            |

# 2. Stack

- **Orchestration:** LangGraph `StateGraph` over a single shared `LabState`. Each module is a node or subgraph that reads and writes its own keys. SQLite checkpointer for persistence. LangGraph `interrupt` for the dataset-permission gate.

- **Inference APIs:** Ollama (local) and Groq (hosted).

- **Models by role** (names to be filled in):

| **Role**                                                               | **Used by**                                                                                        | **Provider** | **Model**              |
|:-----------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:-------------|:-----------------------|
| Generator / agent model                                                | C (researcher and critic agents), B (paper review, conflict check), F (literature evidence labels) | Ollama       | *Qwen3-8B*             |
| Experimenter model                                                     | E (experiment design, config filling)                                                              | Ollama       | *Qwen3-8B*             |
| Reviewer model (larger, ideally a different family from the generator) | F (reviewer)                                                                                       | Groq         | *Qwen3-26B/Llama3-70B* |
| Embedding model (one model for topic and papers)                       | B                                                                                                  | Local        | *TBD*                  |

- **Structured output:** every LLM call goes through the Module A client, which validates against a Pydantic schema and retries up to 2 times. Thinking mode is disabled for structured calls where the model supports it.

- **Storage:** networkx graph saved as GraphML/JSON, embeddings as `.npy`, API responses and LLM outputs cached in SQLite.

- **Config:** provider, model names, budgets, and thresholds as constants at the top of a single `config.py`.

# 3. Shared Types

All modules use these types. Every module returns its payload plus a list of `LedgerEntry` items, which Module A appends to the ledger.

```
class LedgerEntry:
    id: str
    claim: str
    tag: Literal["fact", "hypothesis", "assumption", "speculation"]
    source_ids: list[str]          # paper IDs or experiment IDs
    produced_by: str               # module / agent name
    confidence: float

class Evidence:
    id: str
    hypothesis_id: str
    source_id: str                 # paper ID or experiment ID
    direction: Literal["support", "contradict", "neutral"]
    type: Literal["direct", "indirect"]
    reliability: float             # 0–1
    similarity: float              # 0–1, setting similarity

class RevisionTask:
    id: str
    text: str
    type: Literal["baseline", "ablation", "analysis", "revise_hypothesis", "other"]
    severity: Literal["minor", "major", "critical"]
    est_cost: float

class LabState:                    # LangGraph state
    request: ResearchRequest
    kb: KBHandle | None
    reviews: list[PaperReview]
    conflicts: list[Conflict]
    hypotheses: list[Hypothesis]
    proposals: list[Proposal]
    beliefs: dict[str, BeliefState]       # keyed by hypothesis_id
    evidence: list[Evidence]
    experiments: list[ExperimentResult]
    review_history: list[Review]
    closed_task_ids: set[str]             # revision tasks already addressed
    tasks: list[Task]
    ledger: list[LedgerEntry]
    budget_left: float
    round: int
```

# 4. Modules

## Module A: Tooling and Report Generation

**Description.** Provides the shared infrastructure every other module calls: the LLM client, the shared state and evidence ledger, sandboxed code execution, permission-gated dataset download, citation checking, and final report generation. Module A defines the schemas in Section 3 and must be delivered first.

**Input:** calls from other modules.

**Output (interfaces):** see the Appendix (Module A).

**Report generation.** The report is assembled from the ledger, not written freely. Each claim keeps its tag (fact, hypothesis, assumption, speculation) and its source IDs. Hypotheses whose belief state is marked `preliminary` cannot be described as established. Every citation passes through `check_citations`, and invalid ones are removed and logged.

## Module B: Literature, Knowledge Base, and Conflict Detection

**Description.** Builds the knowledge base from the seed papers and topic prompt. It retrieves related work with best-first search over the citation graph, connects papers by citation and semantic similarity, extracts a structured review of each paper, and flags contradictory findings and disagreement among prior reviewers. Its output is the grounding for every later module.

**Input and output:** see the Appendix (Module B).

**KB construction**

1.  **Resolve seeds.** Match each seed paper to a Semantic Scholar ID by DOI, arXiv ID, or title. Embed the topic prompt as the query vector.

2.  **Best-first search.** Maintain a priority queue keyed on `cosine(paper_embedding, topic_embedding)`. Pop the highest-priority paper, fetch its references and citations from Semantic Scholar (batch endpoint), embed the neighbours’ abstracts, and push them with their priority. Discard papers below `SIM_THRESHOLD`. Stop at `MAX_PAPERS`, `MAX_DEPTH`, or `FANOUT_PER_NODE`.

3.  **Embeddings.** One embedding model is used for the topic and all papers, so all vectors share one space.

4.  **Edges.**

    - *Citation edges:* directed, taken from the API.

    - *Similarity edges:* mutual kNN. Compute the top-k neighbours of each paper and keep an edge only if each paper is in the other’s top-k. Edge weight is the cosine similarity. A paper left isolated is connected to its single best neighbour.

5.  **Concept layer.** The per-paper LLM review extracts concepts, methods, datasets, and metrics. Each becomes a typed node linked to its papers. Names are normalized (e.g. “ImageNet-1k” → “ImageNet”) so that set overlaps in Module F match. Gap mining (Module C) and CSP domains (Module D) are built from these nodes.

6.  **Paper review.** The generator model reads each abstract (full text for seed papers only, from arXiv) and fills the `PaperReview` schema. Fields not present in the text are set to “not stated” (`idea`, `method`) or left as an empty list (all list fields).

7.  **Queries.**

    - `closest_to_text(text, k)` and `closest_to_paper(paper_id, k)`: cosine top-k over the embeddings. Vectors are L2-normalised, so a dot product is the cosine.

    - `citation_path`: `nx.shortest_path` over citation edges, used to trace influential citation paths.

    - `neighbors`: one-hop citation and similarity neighbours.

    - `get_paper(id)`, `get_review(id)`, `get_embeddings(ids)` and `resolve_id(external_id)` (S2 ID, arXiv ID or DOI to the internal ID).

    Other modules open the KB with `open_kb(handle)`, which also exposes the raw `graph`, `embeddings`, `ids` and `topic_vector`, so a module can run its own queries. The interface is implemented in `module_B_literature_kg/interface.py`.

    **Conventions.** Paper IDs are short internal IDs (`P1`, `P2`, ...); `PaperMeta` holds the real IDs. Embedding row *i* belongs to paper `ids[i]`. The graph is a `networkx.MultiDiGraph`: paper nodes are `P1`...; concept-layer nodes are `<type>:<normalised name>` (e.g. `dataset:truthfulqa`) with `type` in `concept`, `method`, `dataset`, `metric`; every edge has a `kind` of `cites`, `similar` (stored in both directions, with `weight`), `has_concept`, `has_method`, `has_dataset` or `has_metric`.

**Conflict detection**

- *Contradictory findings:* for paper pairs that are mutual-kNN neighbours and share a dataset or metric, the generator model performs an NLI-style comparison of their reported claims and returns `contradict`, `agree`, or `unrelated` with the supporting text. Only `contradict` results become `Conflict` records. The number of pairs checked is capped by `MAX_CONFLICT_PAIRS`.

- *Reviewer disagreement:* for papers with OpenReview data, compute the variance of reviewer scores. Above `REVIEW_VAR_THRESHOLD`, the generator model summarizes the points on which reviewers disagree. Coverage is limited to venues hosted on OpenReview.

Conflicts are written to the ledger. Both papers of a conflict are added to the candidate set when Module F collects literature evidence for a hypothesis, so contradicting work is not missed.

## Module C: Hypothesis Generation

**Description.** Finds research gaps grounded in the knowledge base and turns them into testable hypotheses through a VirSci-style multi-agent discussion. Gap finding and hypothesis generation run as a single loop: each gap is developed directly into a hypothesis with a rationale.

**Input:** `KBHandle`, `list[PaperReview]`, `list[Conflict]`, `ResearchRequest`.

**Output:** see the Appendix (Module C).

**Implementation**

1.  **Gap mining** from the KB:

    - sparse regions: low-density areas of the similarity graph near the topic;

    - missing combinations: concept or method pairs that never co-occur in any paper;

    - conflicts from Module B;

    - limitations: extracted limitations from the KG can be used for extracting the gaps;

    - `future_work` fields from the paper reviews.

2.  **Multi-agent discussion:** `N_AGENTS` researcher and critic agents (generator model) discuss the gaps over `K_ITERATIONS` rounds and produce `Z_HYPOTHESES` hypotheses. Critics challenge novelty and unsupported claims each round.

3.  **Scoring:** a hand-weighted score  
    `score = w_n`$\cdot$`novelty + w_e`$\cdot$`evidence_support + w_f`$\cdot$`feasibility + w_j`$\cdot$`llm_judge`  
    where novelty is `1 `$-$` max cosine` to existing KB papers. Weights are set by hand or tuned on a small labelled set. All terms (novelty, feasibility, evidence_support, llm_judge) must be normalised to the range \[0, 1\]. Novelty will be normalised inherently, the rest must be in the same range.

4.  **Final ranking** of selected hypotheses uses the belief model (Module F), not this score alone.

## Module D: Workflow Tracker, CSP, and RL Controller

**Description.** Controls the process. The workflow tracker decides which steps are allowed to run and replans on failure. The CSP turns a hypothesis into a concrete, feasible proposal or rejects it. The RL controller chooses the next research or revision action in the loop.

### D.1 Workflow tracker (planning)

The `Task` type is in the Appendix (Module D).

- The tracker is the LangGraph router: `ready_tasks()` returns pending tasks whose preconditions are satisfied by the current set of facts, and the router dispatches them. Tasks with no dependency between them can run in parallel.

- Example constraints: no experiment before `dataset_selected` and `metric_selected`; no conclusion before `evidence_obtained`.

- **Replanning:** when a task fails or a resource becomes unavailable, its effects are marked unmet and the task is either re-queued (possibly with a modified config) or replaced by an alternative (for example, a different dataset). A critical reviewer weakness can re-open earlier tasks such as hypothesis revision.

- **Action masking:** the tracker also tells the RL controller which actions are currently allowed.

### D.2 CSP proposal construction

**Input:** one `Hypothesis`, `KBHandle`, and a `Resources` object (compute budget, time, allowed licenses).

**Output:** a `Proposal`, or a rejection with reasons. The types are in the Appendix (Module D).

- **Variables:** research question, dataset, method, baseline, metric, experimental setting.

- **Domains:** taken from the KB’s dataset, method, and metric nodes.

- **Constraints:** novelty (maximum cosine similarity to existing KB papers below `NOVELTY_MAX_SIM`), resource availability, dataset licensing, compute budget, experimental feasibility, ethical requirements, and compatibility among components (the metric fits the task, the baseline runs on the dataset, the experiment can test the hypothesis).

- **Solver:** plain backtracking or `python-constraint`. If no assignment satisfies the constraints, the hypothesis is rejected with reasons, or returned to Module C for revision.

### D.3 RL controller

**Position in the loop.** The controller is the decision point at the end of each round, after the reviewer:

```
E (experiment) → F.Belief → F.Reviewer → D.RL picks next action → executes via B / E / C → next round
```

**Input and output:** the `Observation` and `Action` types are in the Appendix (Module D).

The open-task counts are per type, not a single total: without them the controller cannot tell that `add_baseline` is useful *because* a baseline task is open.

**Featurization (Observation → numeric vector)**

`featurize` converts the `Observation` for the top hypothesis (highest `ci_low`) into a 9-element numeric vector: CI width, budget left (normalized), last overall review score divided by 10, the five open-task counts (`baseline`, `ablation`, `analysis`, `revise_hypothesis`, `other`), and a bias term of 1.0. The code is in the Appendix (Module D).

- The bandit picks only the **action type** (7 arms).

- The **target** is chosen by a fixed rule: the highest-severity open `RevisionTask` matching that type.

- Actions whose preconditions fail are **masked** by the tracker (D.1): no `run_experiment` without a dataset, nothing over budget.

**Algorithm: LinUCB** (one linear model per action)

The `LinUCB` class is in the Appendix (Module D).

**Reward** (immediate):

```
r = w_q·Δreview_score + w_u·Δuncertainty_reduction + w_v·validity − w_c·cost − w_t·time
```

covering proposal quality, uncertainty reduction, experimental validity, reviewer satisfaction, cost, and time.

**Training environment: LabSim (process simulator).** Training takes thousands of steps, and the LLM reviewer and agents cannot be called that often, so the controller is trained on a cheap simulator of the whole loop. Module E’s experiment simulator only covers experiment outcomes; LabSim wraps it.

The `LabSim` interface (`reset` and `step`) is in the Appendix (Module D).

| **Action**                                                           | **Simulated effect**                                                                                                                                                       |
|:---------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `run_experiment`, `ablation`, `add_baseline`             | Sample a result from E’s simulator, then run the real `compute_belief` (no LLM, cheap), so ci_width changes genuinely. Closes a matching open task with probability p. |
| `read_paper`                                                     | Adds a literature evidence item whose direction is drawn from the scenario’s prior (e.g. 70% agrees with `true_effect`).                                               |
| `address_comment`, `analyse_errors`, `revise_hypothesis` | Closes a matching open task with probability p. Revision may reset the belief.                                                                                             |
| all                                                                  | Subtracts the action’s cost from the budget (fixed cost table).                                                                                                            |

The simulated review score is a hand-written function of state, for example `base `$-$` `$\Sigma$` severity of open tasks + bonus if not preliminary`. It stands in for the reviewer and is where most of LabSim’s assumptions live.

**Training loop**

The training loop is in the Appendix (Module D).

**Deployment.** The trained weights are frozen and used in the real LangGraph loop with a lower `alpha` (less exploration). Optionally, `update` keeps running on real rewards from the real reviewer; with only a handful of real steps this matters little.

**Evaluation.**

- The bandit and the three baselines (cheapest action first, reviewer comments in original order, largest expected performance gain) run on **held-out LabSim scenarios**.

- Compared on cumulative reward and cost.

- Compared on whether the final belief points the right way. LabSim knows `true_effect`, so this metric does not depend on the reward design.

- A few real runs as a sanity check.

**Stop condition:** `MAX_ROUNDS` (default 3) reached, budget exhausted, or no open critical revision tasks and the top hypothesis is not preliminary.

## Module E: Experimenter and Evaluation Harness

**Description.** Designs experiments for a proposal, runs them either for real or in simulation, and reports results. It also hosts the evaluation harness that runs the whole system through the test scenarios required by the project brief. Its experiment simulator is reused inside the LabSim process simulator that trains the RL controller (Module D).

**Input:** `Proposal`, optionally a `RevisionTask`.

**Output:** see the Appendix (Module E).

**Implementation**

- **Design and verification:** the experimenter agent drafts the design. A critic agent checks it against the hypothesis (does it isolate the claimed contribution, are baselines adequate) over a small number of iterations.

- **Real runs:** 2–3 hand-written, parameterized experiment scripts. The experimenter agent fills a config (dataset, model, baseline, seeds, ablation flag) rather than writing free-form code. Scripts run through `run_code` in the sandbox. Datasets are obtained only through `fetch_dataset`.

- **Simulated runs:** `metric   Normal(true_effect, noise)` per seed, with failure probability `p_fail`. `true_effect`, `noise`, and `p_fail` are set per scenario.

- **Minimum seeds:** `MIN_SEEDS = 3` for any result used as evidence.

**Evaluation harness**

The `Scenario` and `ScenarioReport` types are in the Appendix (Module E).

Scenarios: closely related prior work, limited resources, contradictory evidence, weak initial hypotheses, failed experiments, conflicting reviewer comments.

## Module F: Reviewer and Belief Model

**Description.** Holds the system’s judgement. The belief model estimates how likely each hypothesis is to be supported given the evidence and updates as new papers and results arrive. The reviewer produces a balanced structured review whose revision tasks feed back into the loop.

### F.1 Belief model

**Input:** `Hypothesis`, `list[Evidence]`.

**Output:** see the Appendix (Module F).

**Bayesian logic.** A Beta-Bernoulli model per hypothesis, where θ is the probability that the hypothesis is supported.

```
prior:        Beta(α0, β0) = Beta(1, 1)

per evidence: w = type × reliability × similarity
              type:        direct = 1.0, indirect = 0.3      (hand-set)
              reliability: peer-reviewed venue, log-scaled citation count,
                           prior reviewer scores where available
              similarity:  closeness of the earlier setting to the proposed one

update:       support    → α += w
              contradict → β += w
              neutral    → no update

posterior:    mean = α / (α + β)
              90% credible interval from Beta quantiles
```

**Why Beta-Bernoulli.** It is conjugate, so each update is a closed-form addition to α or β with no inference engine. It carries uncertainty (the credible interval), which drives the `preliminary` flag and the `ci_low` ranking, and α and β read as weighted pseudo-counts, which keeps the report explainable. Its weaknesses: it assumes independent, exchangeable evidence; the fractional weights are heuristics; it treats “supported” as binary and discards effect sizes; and it cannot represent dependencies between evidence items. Alternatives considered: log-odds accumulation (equally simple, but no interval), a small Bayesian network (represents dependencies, but its tables are equally hand-set and it is more work), and random-effects meta-analysis (principled, but needs comparable effect sizes, which most papers do not report).

**Literature evidence (how α and β get their inputs from papers)**

For each hypothesis `h`:

1.  **Candidate papers:** `h.supporting_paper_ids` plus `kb.closest_to_text(h.statement, k)`, plus both papers of any `Conflict` touching those candidates.

2.  **Labelling:** the generator model receives `h.statement`, the paper’s `PaperReview`, and its abstract, and returns an `EvidenceLabel`: a direction (`support`, `contradict`, or `neutral`), a type (`direct` or `indirect`), and a `quote`, the sentence from the abstract that justifies the label (schema in the Appendix, Module F):

    - **Direct:** the paper tests the same claim (same intervention, same kind of outcome).

    - **Indirect:** a related mechanism, a different task, or a weaker proxy.

    - **Grounding check:** if `quote` is not found verbatim in the abstract, the label is discarded. Temperature 0.

3.  **Reliability** (metadata rules, no LLM):

    ```
    venue       = 1.0 if peer-reviewed venue else 0.6
    cites       = min(1, log10(1 + citations) / log10(1 + C_REF))
    reliability = 0.5·venue + 0.5·cites     # × normalized OpenReview score if available
    ```

4.  **Similarity** (closeness of the paper’s setting to the proposal’s):

    ```
    similarity = Jaccard(paper.datasets ∪ paper.metrics ∪ paper.methods,
                         {proposal.dataset, proposal.metric, proposal.method, *proposal.baselines})
    ```

    Before a proposal exists, `cosine(paper, hypothesis)` is used instead.

**Experiment evidence**

For each `ExperimentResult` with `status == "ok"`:

1.  Paired differences per seed: `d_i = metric_i `$-$` baseline_i` (sign flipped if lower is better).

2.  90% t-interval on the mean of `d`.

    - Interval entirely above 0 → support; entirely below 0 → contradict.

    - Interval containing 0 → recorded in the ledger, no update.

3.  Weight: `type = direct`, `reliability = 1`, `similarity = 1`, multiplied by `min(1, n_seeds / REF_SEEDS)`.

4.  **Ablation:** compares the full method against the method with the component removed. A drop when the component is removed supports the hypothesis that the component contributes.

5.  A failed run (crash or error) is not evidence about the hypothesis and gives no update.

Simulated runs return the same `ExperimentResult` format and go through the same path.

**Update rule (recomputed from the full evidence list each time)**

Recomputing rather than mutating α and β in place keeps the state consistent when a hypothesis is revised or evidence is re-labelled.

The `compute_belief` function that implements this rule is in the Appendix (Module F).

**Update triggers**

| **Trigger**                        | **New evidence**                                                                                                                                               |
|:-----------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hypothesis created (after C / CSP) | Literature pass over candidate papers                                                                                                                          |
| Experiment finishes (E)            | One experiment item                                                                                                                                            |
| RL action `read_paper`         | Literature pass on the newly added paper                                                                                                                       |
| Hypothesis revised                 | All literature for that hypothesis is re-labelled (labels were for the old statement). Experiment items are kept only if the testable prediction is unchanged. |

**Rules**

- **One evidence item per paper per hypothesis**, so a single paper cannot be counted repeatedly.

- **Preliminary flag:** set when `ci_high `$-$` ci_low > CI_WIDTH_MAX` (default 0.4) or `total_weight < MIN_WEIGHT` (default 3). Both defaults are placeholders to be tuned (see Section 9.1).

- **Ranking:** hypotheses are ranked by `ci_low`, so thinly evidenced hypotheses do not rank highly on a high mean alone.

### F.2 Reviewer

**Input:** `Proposal`, `list[ExperimentResult]`, `BeliefState`, ledger excerpt for the hypothesis.

**Output:** see the Appendix (Module F).

**Implementation.** The reviewer model (larger, ideally a different family from the generator) runs once per loop round. Every weakness must reference a ledger entry or state that it is an assumption. Revision tasks go to Module D’s RL controller, and their `type` and `severity` fields are what the controller counts.

# 5. Data Flow: Text and Numbers

Most of the system passes **structured text** between modules. Only a few components need numbers.

- **LLM agents work on text.** They never see vectors.

- **Algorithms work on numbers or symbols.** The bandit, belief model, and CSP never read free text.

- **Pydantic schemas are the bridge.** When an LLM must output `direction: "support" | "contradict" | "neutral"`, `severity: "minor" | "major" | "critical"`, or `overall: 6.5`, its text is forced into categories and numbers that code can count and compute with.

**Conversion points (text → numbers)**

| **\#** | **Where**                  | **What turns into numbers**                                                      |
|:-------|:---------------------------|:---------------------------------------------------------------------------------|
| 1      | B: embedding model         | Abstract and topic text → vectors → cosine similarity                            |
| 2      | Any LLM call with a schema | Free reasoning → enums and scores (evidence direction, task type, review scores) |
| 3      | B / F: metadata rules      | Venue and citation count → reliability (0–1)                                     |
| 4      | F: set overlap             | Extracted dataset/metric/method names → Jaccard similarity                       |
| 5      | E: experiments             | Already numeric (metric values per seed)                                         |

**Per-module data forms**

| **Step**   | **Consumes**                                                | **Produces**                                           | **Form**                          |
|:-----------|:------------------------------------------------------------|:-------------------------------------------------------|:----------------------------------|
| B          | Topic, seed papers                                          | Graph, embeddings, `PaperReview`s, `Conflict`s | Vectors + structured text         |
| C          | Reviews, conflicts, KB                                      | `Hypothesis` list                                  | Text + score numbers              |
| D.CSP      | Hypothesis, KB dataset/method/metric nodes, `Resources` | `Proposal`                                         | Symbols (names chosen from lists) |
| F.Belief   | `Evidence` items                                        | `BeliefState` (α, β, CI)                           | Numbers                           |
| E          | `Proposal`                                              | `ExperimentResult`                                 | Numbers                           |
| F.Reviewer | Proposal, results, belief, ledger                           | `Review`                                           | Text + scores + typed tasks       |
| D.RL       | Whole `LabState`, via `featurize`                   | One action type                                        | Number vector in, enum out        |
| A          | Ledger                                                      | Report                                                 | Text                              |

Everything lives in `LabState`. Each node reads the keys it needs and writes its own.

# 6. LangGraph Integration

```
START → B → C → D.CSP → F.Belief(rank) → E → F.Reviewer → D.RL
D.RL → {E, F.Belief, B, C}   (conditional edge on the chosen action)
D.RL → A.Report → END        (on stop condition)
```

- Every node reads from and writes to `LabState`.

- Every node routes through D’s tracker before running, so preconditions are always checked.

- The dataset-permission gate pauses the graph with `interrupt` and resumes on the user’s answer.

# 7. Worked Example

One run traced end to end, using the example from the project brief. All numbers are illustrative, except the belief and interval values, which are computed with the formulas in Module F.

## 7.1 Test case

- **Request:** “Improve the factual reliability of question answering with a verification agent.” Two seed papers, budget = 150 papers, depth 2.

- **Scenario features covered:** closely related prior work, contradictory evidence, one failed experiment, and reviewer revisions.

- **Config used:** `MAX_ROUNDS = 3`, `REF_SEEDS = 5`, `CI_WIDTH_MAX = 0.4`, `MIN_WEIGHT = 3`, initial budget = 1.0.

## 7.2 Step 1: Module B builds the knowledge base

**Role:** find the literature, structure it, and flag conflicts.

- **Search:** best-first search from the two seeds collects 150 papers, ordered by cosine similarity to the topic embedding.

- **Graph:** citation edges plus mutual-kNN similarity edges. Concept nodes include `self-verification`, `retrieval`, `verifier agent`, `TruthfulQA`, and `accuracy`.

- **Paper reviews:** one `PaperReview` per paper. For example:

```
{ "paper_id": "P1",
  "idea": "Model drafts an answer, plans verification questions, answers them, revises",
  "method": "single-model self-verification", "datasets": ["TruthfulQA"],
  "metrics": ["accuracy"], "future_work": ["use an external verifier"] }
```

- **Conflicts:**

  - *Finding:* P1 reports that self-verification improves accuracy on TruthfulQA, while P4 reports no gain under matched compute. Both are mutual-kNN neighbours sharing a dataset and metric, so the NLI check labels them `contradict`.

  - *Reviewer:* P2 has OpenReview scores \[3, 8, 4\] with high variance. The summary says reviewers disagreed on whether the baselines were compute-matched.

**Ledger:** facts such as “P1 reports X (fact, source P1)” and the two conflicts.

## 7.3 Step 2: Module C generates hypotheses

**Role:** turn gaps into testable hypotheses.

- **Gaps mined:**

  - G1 *(future_work)*: “use an external verifier” appears in 9 papers.

  - G2 *(missing combination)*: no paper combines `verifier agent` with `retrieval` on `TruthfulQA`.

  - G3 *(conflict)*: the P1/P4 disagreement.

- **Discussion:** 4 agents over 3 iterations (generator model). Critics challenge novelty in each round.

- **Output:** 3 hypotheses.

| **ID** | **Statement**                                                                                 | **Score** |
|:-------|:----------------------------------------------------------------------------------------------|:----------|
| H1     | A separate retrieval-grounded verifier agent improves factual accuracy over self-verification | 0.71      |
| H2     | Verification gains disappear under compute-matched comparison                                 | 0.64      |
| H3     | Self-verification improves accuracy on multi-hop QA                                           | 0.58      |

## 7.4 Step 3: Module D.CSP builds proposals

**Role:** turn each hypothesis into a feasible, concrete proposal, or reject it.

- **H3 rejected:** its closest existing paper has cosine 0.93, above `NOVELTY_MAX_SIM = 0.85`, so it duplicates existing work.

- **H1 accepted:**

```
{ "hypothesis_id": "H1", "dataset": "TruthfulQA (500-question subset)",
  "method": "generator + separate retrieval-grounded verifier",
  "baselines": ["self-verification"], "metric": "accuracy",
  "constraint_report": { "satisfied": ["novelty", "license", "compute", "compatibility"],
                         "violated": [] } }
```

- **H2 accepted** with a similar proposal.

**Tracker facts now true:** `dataset_selected`, `metric_selected`, `proposal_ready`.

## 7.5 Step 4: Module F.Belief ranks hypotheses

**Role:** turn evidence into a belief with uncertainty.

H1’s literature evidence (generator labels, each with a verified quote):

| **Paper** | **Direction** | **Type** | **Reliability** | **Similarity** | **w** |
|:----------|:--------------|:---------|:----------------|:---------------|:------|
| P1        | support       | direct   | 0.90            | 0.75           | 0.675 |
| P2        | support       | direct   | 0.70            | 0.50           | 0.350 |
| P3        | support       | indirect | 0.80            | 0.67           | 0.160 |
| P4        | contradict    | direct   | 0.85            | 0.80           | 0.680 |

α = 1 + 1.185 = 2.185 and β = 1 + 0.68 = 1.68.

| **Hypothesis** | **α** | **β** | **Mean** | **90% CI**       | **Rank by ci_low** |
|:---------------|:------|:------|:---------|:-----------------|:-------------------|
| H1             | 2.185 | 1.68  | 0.565    | \[0.179, 0.910\] | **1**              |
| H2             | 1.456 | 1.00  | 0.593    | \[0.128, 0.965\] | 2                  |

H2 has the higher mean but thinner evidence, so H1 ranks first. Both are flagged preliminary.

## 7.6 Step 5: Module E runs the first experiment

**Role:** design, verify, and run the experiment.

1.  The experimenter agent fills the config for the QA-evaluation script: method, baseline, 3 seeds.

2.  A critic agent checks the design. It passes.

3.  **Dataset gate:** `fetch_dataset` shows source, size, and license. The graph pauses on `interrupt` and the user approves.

4.  The script runs in the sandbox.

```
{ "mode": "real", "status": "ok", "seeds": [0, 1, 2],
  "metrics": { "accuracy": [0.612, 0.598, 0.621] },
  "baseline_metrics": { "accuracy": [0.590, 0.585, 0.603] } }
```

**F.Belief update:**

- Per-seed differences are +0.022, +0.013, and +0.018, with a mean of +0.0177.

- The 90% t-interval is \[0.010, 0.025\]. It is above 0, so this counts as support.

- The weight is `1.0 `$\times$` min(1, 3/5) = 0.6`, so α rises to 2.785.

- The posterior mean is 0.624 with a CI of \[0.253, 0.927\] (width 0.674).

Budget left: 0.80.

## 7.7 Step 6: Module F.Reviewer reviews (review 1)

**Role:** a balanced critique that produces typed revision tasks (reviewer model).

```
{ "strengths": ["Directly tests the P1/P4 conflict", "Seeds reported"],
  "weaknesses": ["Gain may come from retrieval, not from verifier separation (ledger: E-exp-1)",
                 "Baseline is not compute-matched; P2's reviewers raised the same issue"],
  "revision_tasks": [
    { "id": "T1", "type": "ablation", "severity": "major",
      "text": "Compare against self-verification given the same retrieval" },
    { "id": "T2", "type": "baseline", "severity": "major",
      "text": "Add a compute-matched self-consistency baseline" },
    { "id": "T3", "type": "analysis", "severity": "minor",
      "text": "Break errors down by question category" } ],
  "scores": { "overall": 5.0 } }
```

## 7.8 Step 7: The loop driven by Module D.RL

**Role:** pick the next action. The bandit was trained beforehand on LabSim.

The context vector is `[ci_width, budget, review/10, #baseline, #ablation, #analysis, #revise, #other, 1]`.

The reward is $\Delta$`review/10 + `$\Delta$`ci_width_reduction `$-$` 0.5 `$\times$` cost`, with the validity term at 0 because the proposal is unchanged.

**Round 1**

- **Context:** `[0.674, 0.80, 0.50, 1, 1, 1, 0, 0, 1]`

- **Bandit picks:** `ablation`, which targets T1 (major).

- **E:** full system vs self-verification with the same retrieval. Differences are +0.011, +0.004, +0.012, giving a CI of \[0.0017, 0.0163\]. This is support with weight 0.6.

- **F.Belief:** α = 3.385. Mean 0.668, CI \[0.317, 0.938\], width 0.621.

- **F.Reviewer:** T1 closed, overall 6.0.

- **Reward:** 0.10 + 0.053 − 0.05 = **+0.103**. Budget 0.70.

**Round 2**

- **Context:** `[0.621, 0.70, 0.60, 1, 0, 1, 0, 0, 1]`

- **Bandit picks:** `add_baseline`, which targets T2.

- **E:** the run crashes (out of memory, because self-consistency samples 8 answers per question). Status is `failed`.

- **F.Belief:** no update, because a failed run is not evidence.

- **D.Tracker:** the effect `baseline_added` is unmet, so the task is re-queued with a reduced batch size. This is the replanning step.

- **Reward:** 0 + 0 − 0.025 = **−0.025**. Budget 0.65.

**Round 3**

- **Context:** `[0.621, 0.65, 0.60, 1, 0, 1, 0, 0, 1]`

- **Bandit picks:** `add_baseline` again.

- **E:** succeeds. Differences are +0.009, +0.010, +0.014, giving a CI of \[0.0065, 0.0155\]. This is support with weight 0.6.

- **F.Belief:** α = 3.985. Mean 0.703, CI \[0.374, 0.947\], width 0.573.

- **F.Reviewer:** T2 closed, overall 6.5. T3 (minor) is still open.

- **Reward:** 0.05 + 0.048 − 0.075 = **+0.023**. Budget 0.50.

**Stop:** `MAX_ROUNDS = 3` reached.

## 7.9 Step 8: Module A generates the report

**Role:** assemble the report from the ledger, check citations, and enforce the preliminary rule.

```
H1 (hypothesis, PRELIMINARY): A separate retrieval-grounded verifier improves factual
accuracy over self-verification.
  Belief: mean 0.70, 90% CI [0.37, 0.95], total weight 3.7.
- [fact] On a 500-question TruthfulQA subset, the method beat self-verification by
  +1.8 points (90% CI +1.0 to +2.5, 3 seeds). Source: E-exp-1
- [fact] With retrieval held equal, the gain shrank to +0.9 points (CI +0.2 to +1.6). Source: E-exp-2
- [fact] Against a compute-matched self-consistency baseline: +1.1 points (CI +0.7 to +1.6). Source: E-exp-4
- [fact] Prior work conflicts: P1 reports gains, P4 reports none under matched compute.
- [assumption] The 500-question subset is representative of the full benchmark.
- [speculation] Gains may not transfer to multi-hop QA (untested).
Open: T3 (error analysis by category).
```

Citations are checked against the KB, and the word “established” is blocked because H1 is preliminary.

## 7.10 Module roles in this run

| **Module** | **What it did in this run**                                                      |
|:-----------|:---------------------------------------------------------------------------------|
| A          | LLM calls, sandbox runs, dataset permission gate, ledger, final report           |
| B          | Built a 150-paper KB, found the P1/P4 conflict and the P2 reviewer disagreement  |
| C          | Mined 3 gaps and produced 3 hypotheses                                           |
| D          | Rejected H3 (CSP), re-queued the failed task (tracker), chose 3 actions (bandit) |
| E          | Ran 4 experiments, 1 of which failed                                             |
| F          | Ranked H1 over H2, updated belief 3 times, wrote 3 reviews with typed tasks      |

# 8. Team Dependencies

- **Define first:** shared types, `LabState`, and the ledger (Module A). All other modules build against them.

- **E → D:** LabSim wraps E’s experiment simulator, so E delivers a stub simulator early.

- **F → D:** LabSim calls F’s `compute_belief`, so F delivers it early (it has no LLM dependency).

- **F → C:** hypothesis ranking needs the belief model’s interface, so F publishes its input and output early, even with a dummy implementation.

- **B → C, F, D:** gap mining, literature evidence, and CSP domains depend on the KB’s concept layer.

| **Module** | **Main techniques**                    | **Notes**                                             |
|:-----------|:---------------------------------------|:------------------------------------------------------|
| A          | Infrastructure                         | Must be delivered first                               |
| B          | Search, embeddings, GenAI              | Most API-dependent                                    |
| C          | GenAI (multi-agent)                    |                                                       |
| D          | Planning, CSP, RL (incl. LabSim)       | Heaviest; CSP could move to C if load needs balancing |
| E          | GenAI, sandboxed execution, simulation | Delivers stub simulator early                         |
| F          | Probabilistic reasoning, GenAI         | Publishes interface and `compute_belief` early    |

# 9. Known Limitations

- **Belief weights** are hand-set heuristics, not calibrated likelihoods.

- **Literature evidence labels** from the generator model are noisy. The quote check catches fabricated support but not misreadings. If evaluation shows high noise, a majority vote over 3 labelling calls is the next fix, at 3$\times$ the cost.

- **Name normalization** limits the Jaccard similarity; unmatched variants of the same dataset lower similarity.

- **Belief ranking favours safe hypotheses**, since novel ones have little direct evidence.

- **The RL policy is trained on LabSim**, so it is only shown to work under LabSim’s assumptions, especially the hand-written simulated review score.

- **The contextual bandit ignores long-term effects** of actions.

- **The workflow tracker checks preconditions but does not search for plans.** If planning is graded separately, a forward-search planner over the same task definitions can be added.

- **If the course’s reasoning-under-uncertainty module centred on Bayesian networks**, graders may expect one; a small Bayesian network is the swap in that case.

- **Reviewer-disagreement detection** covers only papers on OpenReview.

## 9.1 Open issues found by the worked example (not yet applied)

1.  **The preliminary threshold is nearly unreachable.** With a mean around 0.7, a 90% CI width of 0.4 needs α + β ≈ 13–15, i.e. roughly 11–13 units of evidence weight. The example run reached 3.7. As configured, almost every short run ends “preliminary”. Options: tune `CI_WIDTH_MAX` on the evaluation scenarios, or accept that most runs end as preliminary.

2.  **Own experiments weigh only 0.6** with `REF_SEEDS = 5`, less than one strong paper (P1 = 0.675). Setting `REF_SEEDS = 3` would give each 3-seed experiment weight 1.0, which is the recommended change since these are direct tests in the proposal’s exact setting.

3.  **The reviewer runs even when nothing changed** (round 2 of the example, after a failed run). Calling the reviewer only when there are new results would save calls to the most expensive model.

# 10. Appendix: Pseudocode and Interfaces by Module

Input and output definitions and code for each module, collected in one place. Modules appear in the same order as in Section 4.

## Module A: Tooling and Report Generation

**Input:** calls from other modules.

**Output (interfaces)**

```
llm_call(role: str, prompt: str, schema: type[BaseModel]) -> BaseModel
    # role selects the model from the stack table; caching, retries, cost logging

ledger.add(entries: list[LedgerEntry]) -> None
ledger.get(filter) -> list[LedgerEntry]

run_code(code_or_script: str, files: list[str], timeout: int) -> CodeResult
    # Docker container, no network access, enforced timeout
class CodeResult:
    stdout: str
    stderr: str
    exit_code: int
    files: list[str]

fetch_dataset(req: DatasetRequest) -> DatasetRecord
    # Shows source, size, and license, then blocks on LangGraph interrupt
    # until the user approves or rejects.
class DatasetRecord:
    name: str
    url: str
    size: int
    license: str
    path: str | None
    approved: bool

check_citations(text: str) -> CitationCheck
class CitationCheck:
    valid: list[str]
    invalid: list[str]             # cited papers not present in the KB

generate_report(state: LabState) -> str   # path to report.md
```

## Module B: Literature, Knowledge Base, and Conflict Detection

**Input**

```
class ResearchRequest:
    topic: str
    seed_papers: list[str]         # DOI, arXiv ID, or title
    budget: Budget                 # max_papers, max_depth, fanout_per_node
```

**Output**

```
class KBHandle:                    # paths only; stored in LabState
    graph_path: str                # node-link JSON of the MultiDiGraph
    emb_path: str                  # .npy, (n_papers, dim), L2-normalised
    ids_path: str                  # ids.json, paper IDs in embedding-row order
    meta_path: str                 # papers.json, {paper_id: PaperMeta}
    reviews_path: str              # reviews.json, {paper_id: PaperReview}
    topic_emb_path: str            # .npy, (dim,), the topic vector
# open_kb(handle, embed_fn) -> KnowledgeBase with: get_paper(id), get_review(id),
# closest_to_text(text, k), closest_to_paper(id, k), neighbors(id), citation_path(a, b),
# get_embeddings(ids), resolve_id(external_id), and raw graph / embeddings / ids / topic_vector

class PaperMeta:
    paper_id: str                  # internal ID, e.g. "P1"
    title: str
    abstract: str
    year: int | None
    venue: str | None
    citation_count: int | None
    s2_id: str | None
    arxiv_id: str | None
    doi: str | None
    is_seed: bool
    has_full_text: bool

class PaperReview:
    paper_id: str
    title: str
    idea: str                      # "not stated" if absent
    method: str                    # "not stated" if absent
    pros: list[str]                # empty list if not stated (all list fields)
    cons: list[str]
    future_work: list[str]
    concepts: list[str]
    methods: list[str]
    datasets: list[str]
    metrics: list[str]
    evidence_quotes: dict[str, str]    # field name -> supporting quote from the paper

class Conflict:
    id: str
    type: Literal["finding", "reviewer"]
    paper_ids: list[str]           # two papers for "finding", one for "reviewer"
    summary: str
    evidence_text: str
    shared_dataset: str | None     # "finding" only
    shared_metric: str | None      # "finding" only
    claim_a: str | None            # "finding" only
    claim_b: str | None            # "finding" only

class ScoredPaper:                 # returned by closest_to_*
    paper_id: str
    score: float                   # cosine similarity

class Neighbor:                    # returned by neighbors
    paper_id: str
    kind: Literal["cites", "cited_by", "similar"]
    weight: float | None
```

## Module C: Hypothesis Generation

**Input:** `KBHandle`, `list[PaperReview]`, `list[Conflict]`, `ResearchRequest`.

**Output**

```
class Gap:
    id: str
    source: Literal["sparse_region", "missing_combination", "conflict", "future_work"]
    description: str
    paper_ids: list[str]

class Hypothesis:
    id: str
    gap_ids: list[str]
    statement: str
    testable_prediction: str
    rationale: str
    supporting_paper_ids: list[str]
    assumptions: list[str]
    score_parts: dict[str, float]  # novelty, evidence_support, feasibility, llm_judge
    score: float
    status: Literal["candidate", "selected", "rejected", "revised"]
```

## Module D: Workflow Tracker, CSP, and RL Controller

### D.1 Workflow tracker (planning)

```
class Task:
    id: str
    preconditions: set[str]        # facts that must hold, e.g. "dataset_selected"
    effects: set[str]              # facts that hold after success
    status: Literal["pending", "ready", "running", "done", "failed"]
```

### D.2 CSP proposal construction

**Input:** one `Hypothesis`, `KBHandle`, and

```
class Resources:
    compute_budget: float
    time: float
    licenses_allowed: list[str]
```

**Output**

```
class Proposal:
    id: str
    hypothesis_id: str
    research_question: str
    dataset: str
    method: str
    baselines: list[str]
    metric: str
    setting: str
    constraint_report: ConstraintReport   # satisfied: list[str], violated: list[str]
# or a rejection: { hypothesis_id, reasons: list[str] }
```

### D.3 RL controller

**Input**

```
class Observation:
    ci_width: float                # top hypothesis, from F.Belief
    budget_left: float             # normalized by initial budget, from A's cost log
    last_review_score: float       # overall / 10, from the reviewer's JSON
    open_tasks_by_type: dict[str, int]   # counts of open RevisionTasks per type:
                                         # baseline, ablation, analysis,
                                         # revise_hypothesis, other
```

**Featurization (Observation → numeric vector)**

```
def featurize(state):
    top = best_hypothesis(state)                    # highest ci_low
    b = state.beliefs[top.id]
    last = state.review_history[-1]
    open_tasks = [t for t in last.revision_tasks if t.id not in state.closed_task_ids]
    count = lambda kind: sum(t.type == kind for t in open_tasks)
    return np.array([
        b.ci_high - b.ci_low,
        state.budget_left / INITIAL_BUDGET,
        last.scores["overall"] / 10,
        count("baseline"),
        count("ablation"),
        count("analysis"),
        count("revise_hypothesis"),
        count("other"),
        1.0,                                        # bias term
    ])
```

**Output**

```
class Action:
    type: Literal["read_paper", "run_experiment", "add_baseline", "ablation",
                  "analyse_errors", "revise_hypothesis", "address_comment"]
    target_id: str | None
```

**Algorithm: LinUCB** (one linear model per action)

```
import numpy as np

class LinUCB:
    def __init__(self, n_actions, dim, alpha=1.0):
        self.A = [np.eye(dim) for _ in range(n_actions)]
        self.b = [np.zeros(dim) for _ in range(n_actions)]
        self.alpha = alpha

    def select(self, x, allowed):
        best, best_p = None, -np.inf
        for a in allowed:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            if p > best_p:
                best, best_p = a, p
        return best

    def update(self, a, x, r):
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
```

**Training environment: LabSim (process simulator)**

```
class LabSim:                       # one episode = one research project
    def reset(scenario):  # true_effect, noise, p_fail, initial open tasks, budget
    def step(action):     # returns next_obs, reward, done
```

**Training loop**

```
for episode in range(N_EPISODES):
    sim.reset(random.choice(train_scenarios))
    obs, done = sim.observe(), False
    while not done:
        x = featurize(obs)
        a = bandit.select(x, sim.allowed_actions())
        obs, r, done = sim.step(a)
        bandit.update(a, x, r)
```

## Module E: Experimenter and Evaluation Harness

**Input:** `Proposal`, optionally a `RevisionTask`.

**Output**

```
class ExperimentResult:
    id: str
    proposal_id: str
    design: str
    mode: Literal["real", "simulated"]
    metrics: dict[str, list[float]]          # per seed
    baseline_metrics: dict[str, list[float]] # per seed
    n_samples: int
    seeds: list[int]
    status: Literal["ok", "failed"]
    error: str | None
    artifacts: list[str]
```

**Evaluation harness**

```
class Scenario:
    name: str
    kb_fixture: str
    resources: Resources
    simulator_params: dict

class ScenarioReport:
    metrics: dict[str, float]
    traces: list[str]
```

## Module F: Reviewer and Belief Model

### F.1 Belief model

**Input:** `Hypothesis`, `list[Evidence]`.

**Output**

```
class BeliefState:
    hypothesis_id: str
    alpha: float
    beta: float
    mean: float
    ci_low: float                  # 90% credible interval
    ci_high: float
    total_weight: float
    preliminary: bool
```

**Evidence label (literature evidence)**

```
class EvidenceLabel:
    direction: Literal["support", "contradict", "neutral"]
    type: Literal["direct", "indirect"]
    quote: str        # sentence from the abstract that justifies the label
```

**Update rule**

```
from scipy.stats import beta as beta_dist

TYPE_W = {"direct": 1.0, "indirect": 0.3}

def compute_belief(h_id, evidence):
    a, b = 1.0, 1.0
    for e in evidence:
        if e.hypothesis_id != h_id:
            continue
        w = TYPE_W[e.type] * e.reliability * e.similarity
        if e.direction == "support":
            a += w
        elif e.direction == "contradict":
            b += w
    lo, hi = beta_dist.ppf([0.05, 0.95], a, b)
    return a, b, a / (a + b), lo, hi
```

### F.2 Reviewer

**Input:** `Proposal`, `list[ExperimentResult]`, `BeliefState`, ledger excerpt for the hypothesis.

**Output**

```
class Review:
    strengths: list[str]
    weaknesses: list[str]          # each states why the weakness matters
    questions: list[str]
    revision_tasks: list[RevisionTask]
    scores: dict[str, float]       # novelty, soundness, evidence, clarity, overall (0–10)
```