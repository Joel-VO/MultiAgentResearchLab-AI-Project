# Multi-Agent Research Laboratory and Peer-Review Board

A multi-agent system that takes a research topic and a few seed papers and works through the research process: it builds a literature knowledge base, proposes hypotheses, designs and runs experiments, reviews its own work, and writes a report in which every claim is tagged and sourced.

The full architecture, interfaces and a worked example are in [`DESIGN.md`](DESIGN.md).

## What it does

Given a request such as *"Improve the factual reliability of question answering with a verification agent"* plus two seed papers, the system:

1. **Builds a knowledge base** by searching outward from the seed papers over the citation graph, then reviews each paper and flags contradictory findings and reviewer disagreement.
2. **Finds gaps and proposes hypotheses** through a multi-agent discussion grounded in the knowledge base.
3. **Turns hypotheses into feasible proposals** (dataset, method, baselines, metric) with a constraint solver, rejecting those that duplicate existing work or break resource limits.
4. **Ranks hypotheses by belief**, a Bayesian estimate of how likely each is to be supported, with an uncertainty interval that updates as evidence arrives.
5. **Runs experiments** (real or simulated) in a sandbox, then has a reviewer agent critique the results and issue typed revision tasks.
6. **Loops** with a learned controller choosing the next action (read a paper, add a baseline, run an ablation, revise the hypothesis) until the budget or round limit is reached.
7. **Writes a report** assembled from an evidence ledger, where each claim is tagged as fact, hypothesis, assumption or speculation.

## Design principles

- **Grounded:** hypotheses, labels and reviews are tied to papers or experiment results. Evidence labels must quote the source abstract, or they are discarded.
- **Honest about uncertainty:** hypotheses with thin evidence are marked *preliminary* and cannot be described as established in the report.
- **Controlled execution:** code runs in a sandbox with no network access, and datasets are downloaded only after the user approves source, size and license.
- **Modular:** six modules with fixed input and output types that communicate only through a shared state object and the ledger.

## Modules

| Module | What it does |
|---|---|
| **A: Tooling and Report Generation** | Shared infrastructure: LLM client with schema validation, shared state and ledger, sandboxed code execution, dataset permission gate, citation checking, final report |
| **B: Literature, Knowledge Base, and Conflict Detection** | Best-first citation search, knowledge graph (citation and mutual-kNN similarity edges), structured paper reviews, conflict detection |
| **C: Hypothesis Generation** | Gap mining and multi-agent hypothesis generation with scoring |
| **D: Workflow Tracker, CSP, and RL Controller** | Which steps may run and replanning on failure, proposal construction as a constraint problem, and a LinUCB controller trained on a process simulator (LabSim) |
| **E: Experimenter and Evaluation Harness** | Experiment design and runs (real or simulated), and the scenario harness for testing the whole system |
| **F: Reviewer and Belief Model** | Beta-Bernoulli belief model over literature and experiment evidence, and the structured peer review |

## Tech stack

- **Orchestration:** LangGraph over a single shared state, with a SQLite checkpointer
- **Models:** Ollama (local) and Groq (hosted), with the model for each role set in `config.py`
- **Storage:** networkx graph, NumPy embeddings, SQLite cache
- **Validation:** Pydantic schemas on every LLM call

## Getting started

Prerequisites: Python, [Ollama](https://ollama.com) running locally, and a [Groq](https://console.groq.com/docs/models) API key.

```bash
python3.12 -m venv venv # use python3.12 as packages are more stable and support is guaranteed
source venv/bin/activate
pip install -r requirements.txt # Be in the appropriate directory
```

After full integration of the project, the following must be run
```bash
bash install.sh
```

Models, budgets and thresholds are constants at the top of `config.py`.

Modules E and F can be run on their own (from the repo root, with the venv active):

```bash
python3 -m tests.test_e_f            # offline unit tests, no Ollama needed
python3 -m tests.test_ef_standalone  # live demo on the DESIGN.md Section 7 example (needs Ollama)
```

## Repo layout

One package per module. Directory names follow `DESIGN.md` (module letter plus module name).

| Module | Directory | Contents |
|---|---|---|
| A | `module_a_tooling/` | shared types, `LabState`, ledger, LLM client, sandbox, dataset gate, citation check, report |
| B | `module_b_literature_kb/` | citation search, KB graph, paper reviews, conflict detection |
| C | `module_c_hypothesis/` | gap mining, multi-agent discussion, scoring |
| D | `module_d_workflow_csp_rl/` | `tracker.py` (D.1), `csp.py` (D.2), `rl_controller.py` and `labsim.py` (D.3) |
| E | `module_e_experimenter/` | experiment design, runs, simulator, evaluation scenarios |
| F | `module_f_reviewer_belief/` | `belief.py` (F.1), `reviewer.py` (F.2) |

Rules:

- Modules talk to each other only through `LabState` and the ledger (Module A). Do not import one module's internals from another.
- Keep one module per directory. Sub-parts (D.1, D.2, F.1, ...) are files inside their module, named after the sub-part.
- If you add code that does not fit an existing module, say so in the PR and add it to `DESIGN.md` rather than creating an unlisted directory.

## Contributing

### Keep dependencies and ignored files current

Update these in the same commit as the change that needs them:

- **`requirements.txt`**: any package added, removed or version-pinned.
- **`install.sh`**: any new setup step (system package, model pull, folder creation, env var).
- **`.gitignore`**: any new generated or local file (caches, SQLite files, embeddings, downloaded datasets, run outputs, `.env`, API keys).

### Keep the design current

If an edit changes the structure, update `DESIGN.md` in the same commit. That means a new, renamed, merged or removed module; a changed input or output type; or a changed pipeline step. Keep module letters, names and directory names in sync with it.

Code that only fills in the existing design does not need a `DESIGN.md` change.
## License

This project is released under the [MIT License](LICENSE).

In short, anyone may use, copy, modify, merge, publish, distribute, sublicense and sell copies of the software, for private or commercial purposes, provided that:

- the copyright notice and the MIT license text are included in all copies or substantial portions of the software; and
- it is understood that the software is provided "as is", without warranty of any kind, and the authors are not liable for any claim, damages or other liability arising from its use.

Third-party dependencies (see `requirements.txt`) and the models and datasets you use with this project keep their own licenses and terms; check them before redistributing anything.
