import json
from collections import defaultdict
from module_A_tooling.local_llm_inference import OllamaClient
from module_B_literature_kg.interface import (
    KBHandle,
    PaperReview,
    Conflict,
    KnowledgeBase,
    open_kb,
)
from classes import Gap, Hypothesis, ResearchRequest, ResearchAgent

AGENT_SPEC = """
You are a researcher working to identify and evaluate testable hypotheses from research gaps.

Your responsibilities:
1. Propose novel, specific, and testable research hypotheses.
2. Examine relevant papers, evidence, and identified research gaps.
3. Critically evaluate hypotheses proposed by other agents by identifying unsupported assumptions, weaknesses, and alternative explanations.
5. Revise your own hypotheses in response to constructive criticism.
6. Provide evidence-based reasoning and distinguish facts from speculation.
7. State testable predictions, rationale, assumptions, and supporting paper IDs.

During each discussion phase, follow the task assigned in the prompt: research, critique, or revision. You remain capable of all three.
"""

def mine_gaps(
    kb: Any,
    reviews: list[PaperReview],
    conflicts: list[Conflict],
    request: ResearchRequest,
    *,
    max_gaps: int = 50,
    sparse_similarity_threshold: float = 0.65,
    min_sparse_region_size: int = 2,
) -> list[Gap]:
    kb = KnowledgeBase(kbhandle)
    gaps: list[Gap] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    def add_gap(source: str, description: str, paper_ids: list[str]) -> None:
        paper_ids = sorted({pid for pid in paper_ids if pid in valid_ids})
        description = description.strip()
        key = (source, description.casefold(), tuple(paper_ids))

        if key in seen or len(gaps) >= max_gaps:
            return

        seen.add(key)
        gaps.append(Gap(
            id=f"G{len(gaps) + 1}",
            source=source,
            description=description,
            paper_ids=paper_ids,
        ))

    # Merge KB reviews with the current run's reviews.
    # Explicitly supplied reviews take precedence.
    review_by_id = {
        pid: review
        for pid in kb.ids
        if (review := kb.get_review(pid)) is not None
    }
    review_by_id.update({
        review.paper_id: review
        for review in reviews
        if review.paper_id in valid_ids
    })
    all_reviews = list(review_by_id.values())

    # 1. Explicit future work.
    for review in all_reviews:
        for item in review.future_work:
            add_gap(
                "future_work",
                f"Unaddressed future work: {item.strip()}",
                [review.paper_id],
            )

    # 2. Conflicts already identified by Module B.
    for item in conflict:
        if not item.paper_ids:
            continue

        if item.type == "finding":
            description = (
                f"Conflicting findings: {item.summary}. "
                f"Evidence: {item.evidence_text}"
            )
        else:
            description = (
                f"Reviewer-identified issue: {item.summary}. "
                f"Evidence: {item.evidence_text}"
            )

        add_gap("conflict", description, item.paper_ids)

    # 3. Build paper coverage from graph edges.
    # Graph nodes use types and edge kinds defined in interface.py.
    graph_coverage: dict[str, set[str]] = defaultdict(set)

    edge_to_type = {
        "has_concept": "concept",
        "has_method": "method",
        "has_dataset": "dataset",
        "has_metric": "metric",
    }

    for paper_id in kb.ids:
        if paper_id not in kb.graph:
            continue

        for _, target, attrs in kb.graph.out_edges(
            paper_id, data=True
        ):
            kind = attrs.get("kind")
            category = edge_to_type.get(kind)

            if category and target in kb.graph:
                graph_coverage[target].add(paper_id)

    # 4. Sparse regions: graph concepts supported by only one paper.
    for node, paper_ids in sorted(graph_coverage.items()):
        if len(paper_ids) != 1:
            continue

        attrs = kb.graph.nodes[node]
        category = attrs.get("type", "concept")
        name = attrs.get("name", node)

        add_gap(
            "sparse_region",
            (
                f"The {category} '{name}' is connected to only one "
                f"paper in the available knowledge graph. Its role "
                f"in the research topic '{request.topic}' may warrant "
                f"further investigation."
            ),
            list(paper_ids),
        )

    # 5. Missing combinations: methods with datasets or metrics.
    # Use reviewed co-occurrence rather than assuming graph edges
    # establish that a method and dataset were used together.
    category_papers: dict[str, dict[str, set[str]]] = {
        "method": defaultdict(set),
        "dataset": defaultdict(set),
        "metric": defaultdict(set),
    }
    observed_pairs: dict[str, set[tuple[str, str]]] = {
        "method_dataset": set(),
        "method_metric": set(),
    }

    for review in all_reviews:
        methods = {
            value.strip().casefold()
            for value in review.methods if value.strip()
        }
        datasets = {
            value.strip().casefold()
            for value in review.datasets if value.strip()
        }
        metrics = {
            value.strip().casefold()
            for value in review.metrics if value.strip()
        }

        for value in methods:
            category_papers["method"][value].add(review.paper_id)
        for value in datasets:
            category_papers["dataset"][value].add(review.paper_id)
        for value in metrics:
            category_papers["metric"][value].add(review.paper_id)

        observed_pairs["method_dataset"].update(
            (method, dataset)
            for method in methods
            for dataset in datasets
        )
        observed_pairs["method_metric"].update(
            (method, metric)
            for method in methods
            for metric in metrics
        )

    for target_type in ("dataset", "metric"):
        pair_type = f"method_{target_type}"
        target_label = target_type

        methods = category_papers["method"]
        targets = category_papers[target_type]

        for method, method_papers in sorted(methods.items()):
            # Avoid suggesting combinations based on isolated mentions.
            if len(method_papers) < 2:
                continue

            for target, target_papers in sorted(targets.items()):
                if len(target_papers) < 2:
                    continue

                if (method, target) in observed_pairs[pair_type]:
                    continue

                supporting_ids = sorted(method_papers | target_papers)

                add_gap(
                    "missing_combination",
                    (
                        f"The method '{method}' and {target_label} "
                        f"'{target}' appear independently in the "
                        f"reviewed corpus but never co-occur in a "
                        f"review. Investigate whether evaluating this "
                        f"combination would address a meaningful "
                        f"question about '{request.topic}'. This is "
                        f"a corpus-level observation, not proof of "
                        f"global novelty."
                    ),
                    supporting_ids,
                )

    return gaps

def hypothesise(
    kb: KBHandle,
    reviews: list[PaperReview],
    conflict: list[Conflict],
    request: ResearchRequest,
    gaps: list[Gap],
    embed_fn=None,
    n_agents: int = 8,
    k_iterations: int = 5,
    z_hypotheses: int = 3,
    weights: dict[str, float] | None = None,
) -> list[Hypothesis]:
    """Generate and score hypotheses through independent researcher/critic agents.

    Gap discovery remains rule-based in mine_gaps(). Each agent owns a distinct
    OllamaClient and private memory. Messages are exchanged through a shared,
    gap-specific discussion. The lead performs a separate final synthesis call.
    """
    if n_agents < 3 or k_iterations < 1 or z_hypotheses < 1:
        raise ValueError(
            "Require n_agents >= 3, k_iterations >= 1, and z_hypotheses >= 1.")

    kb_obj = open_kb(kb, embed_fn=embed_fn)
    valid_ids = set(kb_obj.ids)
    review_by_id = {
        pid: review for pid in kb_obj.ids
        if (review := kb_obj.get_review(pid)) is not None
    }
    review_by_id.update(
        {r.paper_id: r for r in reviews if r.paper_id in valid_ids})

    weights = weights or {
        "novelty": 0.30, "evidence_support": 0.30, "feasibility": 0.20, "llm_judge": 0.20}
    required_scores = {"novelty", "evidence_support",
                       "feasibility", "llm_judge"}
    if set(weights) != required_scores:
        raise ValueError(f"Weights must have exactly {required_scores}.")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Weights must be finite and non-negative.")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("At least one score weight must be positive.")
    weights = {key: value / weight_sum for key, value in weights.items()}

    def novelty_score(statement: str) -> float:
        if embed_fn is None:
            # No embedding function means novelty cannot be computed faithfully.
            return 0.5
        neighbors = kb_obj.closest_to_text(statement, k=len(kb_obj.ids))
        if not neighbors:
            return 1.0
        max_cosine = max(float(p.score) for p in neighbors)
        return max(0.0, min(1.0, 1.0 - max_cosine))

    conflicts_by_paper: dict[str, list[dict]] = defaultdict(list)
    for item in conflict:
        ids = [pid for pid in item.paper_ids if pid in valid_ids]
        for pid in ids:
            conflicts_by_paper[pid].append({
                "id": item.id, "type": item.type, "summary": item.summary,
                "evidence_text": item.evidence_text, "paper_ids": ids,
                "claim_a": getattr(item, "claim_a", None),
                "claim_b": getattr(item, "claim_b", None),
            })

    all_candidates: list[dict] = []
    failures: list[str] = []

    for gap in gaps:
        evidence_ids = sorted(pid for pid in gap.paper_ids if pid in valid_ids)
        paper_evidence = []
        for pid in evidence_ids:
            meta = kb_obj.get_paper(pid)
            review = review_by_id.get(pid)
            if meta is None:
                continue
            paper_evidence.append({
                "paper_id": pid, "title": meta.title, "abstract": meta.abstract,
                "year": meta.year, "citation_count": meta.citation_count,
                "idea": review.idea if review else "not stated",
                "method": review.method if review else "not stated",
                "pros": review.pros if review else [], "cons": review.cons if review else [],
                "future_work": review.future_work if review else [],
                "concepts": review.concepts if review else [], "methods": review.methods if review else [],
                "datasets": review.datasets if review else [], "metrics": review.metrics if review else [],
                "evidence_quotes": review.evidence_quotes if review else {},
                "conflicts": conflicts_by_paper.get(pid, []),
            })

        # Construct distinct client instances once per gap; identities and private memories persist.
        agents = [
            ResearchAgent(
                name=f"agent_{i + 1}",
                role="research_agent",
                objective=AGENT_SPEC,
                llm=OllamaClient(think=False),
            )
            for i in range(n_agents)
        ]
        lead = ResearchAgent(
            "lead", "Scientific Synthesis Lead",
            "Synthesize the reviewed candidates, resolve critiques, and return the strongest evidence-grounded hypotheses.",
            OllamaClient(think=False),
        )
        # gap-specific: never leaks into another gap
        discussion: list[dict] = []
        gap_candidates: list[dict] = []

        for round_no in range(1, k_iterations + 1):
            # Research phase: independent proposals in round 1; revisions in later rounds.
            for agent in agents:
                phase = "initial_proposal" if round_no == 1 else "revision_after_critique"
                try:
                    msg = agent.respond(topic=request.topic, gap=gap.model_dump(), evidence=paper_evidence,
                                        shared_messages=discussion, round_no=round_no, phase=phase)
                    discussion.append(msg)
                    for candidate in msg["hypotheses"]:
                        if isinstance(candidate, dict):
                            normalized = _normalize_candidate(
                                candidate, valid_ids, gap.id, agent.name, round_no)
                            if normalized:
                                gap_candidates.append(normalized)
                except Exception as exc:
                    failures.append(
                        f"{gap.id} round {round_no} {agent.name}: {exc}")

            # Critic phase: each critic receives proposals and preceding critiques.
            for agent in agents:
                try:
                    msg = agent.respond(topic=request.topic, gap=gap.model_dump(), evidence=paper_evidence,
                                        shared_messages=discussion, round_no=round_no, phase="critique")
                    discussion.append(msg)
                except Exception as exc:
                    failures.append(
                        f"{gap.id} round {round_no} {agent.name}: {exc}")

        # A distinct lead call receives the complete gap-specific discussion and synthesizes final candidates.
        lead_prompt = {
            "topic": request.topic, "seed_papers": request.seed_papers,
            "gap": gap.model_dump(), "evidence": paper_evidence,
            "discussion": discussion,
            "candidate_pool": gap_candidates,
            "instructions": [
                "Synthesize rather than merely repeat the candidate pool.",
                "Address relevant critiques and discard candidates with fatal flaws.",
                "Return at most z_hypotheses candidates.",
                "Supporting paper IDs must come from the supplied evidence IDs.",
                "Predictions must be observable and falsifiable.",
                "Scores evidence_support, feasibility, and llm_judge must be numbers from 0 to 1.",
                "Do not claim global novelty based only on corpus absence.",
                "Return valid JSON only.",
            ],
            "z_hypotheses": z_hypotheses,
            "required_json": {"hypotheses": [{
                "statement": "falsifiable claim", "testable_prediction": "observable prediction",
                "rationale": "synthesis of evidence and critiques", "supporting_paper_ids": [],
                "assumptions": [], "score_parts": {"evidence_support": 0.0, "feasibility": 0.0, "llm_judge": 0.0},
            }]},
        }
        try:
            synthesis = lead.llm.generate(
                user_prompt=json.dumps(
                    lead_prompt, ensure_ascii=False, default=str),
                system_prompt="You are the independent scientific synthesis lead. Critically assess the full discussion and return only valid JSON.",
                temperature=0,
            )
            final_items = synthesis.get("hypotheses", [])
            if not isinstance(final_items, list):
                final_items = []
            for candidate in final_items:
                if isinstance(candidate, dict):
                    normalized = _normalize_candidate(
                        candidate, valid_ids, gap.id, "lead", k_iterations)
                    if normalized:
                        all_candidates.append(normalized)
        except Exception as exc:
            failures.append(f"{gap.id} final synthesis lead: {exc}")
            # Fall back to validated researcher candidates if lead synthesis fails.
            all_candidates.extend(gap_candidates)

    # Exact-text deduplication. Keep the version with strongest evidence score.
    unique: dict[str, dict] = {}
    for item in all_candidates:
        key = " ".join(item["statement"].casefold().split())
        if key not in unique or item["evidence_support"] > unique[key]["evidence_support"]:
            unique[key] = item

    ranked = []
    for item in unique.values():
        parts = {
            "novelty": novelty_score(item["statement"]),
            "evidence_support": item["evidence_support"],
            "feasibility": item["feasibility"],
            "llm_judge": item["llm_judge"],
        }
        score = sum(weights[name] * parts[name] for name in weights)
        ranked.append((score, item, parts))
    ranked.sort(key=lambda entry: entry[0], reverse=True)

    output: list[Hypothesis] = []
    for index, (score, item, parts) in enumerate(ranked[:z_hypotheses], start=1):
        output.append(Hypothesis(
            id=f"H{index}", gap_ids=[item["gap_id"]],
            statement=item["statement"], testable_prediction=item["testable_prediction"],
            rationale=item["rationale"], supporting_paper_ids=item["supporting_paper_ids"],
            assumptions=item["assumptions"], score_parts=parts, score=float(score), status="candidate",
        ))
    return output
