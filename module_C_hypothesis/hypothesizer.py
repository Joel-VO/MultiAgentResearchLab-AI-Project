import json
from collections import defaultdict
from module_A_tooling.local_llm_inference import OllamaClient
from module_B_literature_kg.interface import (
    KBHandle,
    PaperReview,
    Conflict,
    KnowledgeBase,
)
from classes import Gap, Hypothesis, ResearchRequest


def mine_gaps(
    kbhandle: KBHandle,
    reviews: list[PaperReview],
    conflict: list[Conflict],
    request: ResearchRequest,
) -> list[Gap]:
    kb = KnowledgeBase(kbhandle)
    gaps: list[Gap] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    valid_ids = set(kb.ids)

    def add_gap(source: str, description: str, paper_ids: list[str]) -> None:
        paper_ids = sorted({pid for pid in paper_ids if pid in valid_ids})
        description = description.strip()
        key = (source, description.casefold(), tuple(paper_ids))

        if not description or key in seen:
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
) -> list[Hypothesis]:
    knowledge_base = KnowledgeBase(kb)
    llm = OllamaClient()

    valid_ids = set(knowledge_base.ids)
    review_by_id = {
        review.paper_id: review
        for review in reviews
        if review.paper_id in valid_ids
    }

    # Use KB reviews as a fallback for papers not in the supplied list.
    for paper_id in knowledge_base.ids:
        if paper_id not in review_by_id:
            review = knowledge_base.get_review(paper_id)
            if review is not None:
                review_by_id[paper_id] = review

    conflicts_by_paper: dict[str, list[Conflict]] = {}
    for item in conflict:
        for paper_id in item.paper_ids:
            conflicts_by_paper.setdefault(paper_id, []).append(item)

    system_prompt = """
You are a careful scientific research assistant.

Generate testable research hypotheses from the supplied research gap
and evidence. Do not invent paper findings, citations, or experiments
that are claimed to have already occurred.

Requirements:
- Each hypothesis must make a specific, falsifiable claim.
- Its testable_prediction must describe an observable result.
- Its rationale must explain how the supplied evidence motivates it.
- State uncertain premises as assumptions.
- Support hypotheses only with supplied paper IDs.
- Do not claim global novelty from absence in the supplied corpus.
- Prefer a small number of strong hypotheses over many weak ones.
- Scores must be numbers between 0 and 1.
- Scores are preliminary estimates, not verified scientific facts.

Return only valid JSON in this structure:
{
  "hypotheses": [
    {
      "statement": "...",
      "testable_prediction": "...",
      "rationale": "...",
      "supporting_paper_ids": ["P1"],
      "assumptions": [],
      "score_parts": {
        "novelty": 0.5,
        "evidence_support": 0.7,
        "feasibility": 0.8,
        "llm_judge": 0.6
      }
    }
  ]
}

If no defensible hypothesis can be generated, return:
{"hypotheses": []}
"""

    hypotheses: list[Hypothesis] = []
    seen_statements: set[str] = set()

    for gap in gaps:
        # Restrict evidence to the papers associated with this gap.
        paper_ids = [
            paper_id for paper_id in gap.paper_ids
            if paper_id in valid_ids
        ]

        paper_evidence = []
        for paper_id in paper_ids:
            meta = knowledge_base.get_paper(paper_id)
            review = review_by_id.get(paper_id)

            paper_evidence.append({
                "paper_id": paper_id,
                "title": meta.title,
                "abstract": meta.abstract,
                "year": meta.year,
                "idea": review.idea if review else "not stated",
                "method": review.method if review else "not stated",
                "pros": review.pros if review else [],
                "cons": review.cons if review else [],
                "future_work": review.future_work if review else [],
                "concepts": review.concepts if review else [],
                "methods": review.methods if review else [],
                "datasets": review.datasets if review else [],
                "metrics": review.metrics if review else [],
                "evidence_quotes": (
                    review.evidence_quotes if review else {}
                ),
            })

        relevant_conflicts = [
            {
                "type": item.type,
                "paper_ids": item.paper_ids,
                "summary": item.summary,
                "evidence_text": item.evidence_text,
                "claim_a": item.claim_a,
                "claim_b": item.claim_b,
            }
            for item in conflict
            if set(item.paper_ids) & set(paper_ids)
        ]

        user_prompt = json.dumps({
            "research_topic": request.topic,
            "seed_papers": request.seed_papers,
            "gap": gap.model_dump(),
            "papers": paper_evidence,
            "conflicts": relevant_conflicts,
        }, ensure_ascii=False)

        try:
            raw = llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                format="json",
                temperature=0,
            )
            result = json.loads(raw)
        except (ValueError, TypeError, RuntimeError) as exc:
            # A failed inference for one gap should not discard results
            # already generated for other gaps.
            print(f"Hypothesis generation failed for {gap.id}: {exc}")
            continue

        if not isinstance(result, dict):
            continue

        candidates = result.get("hypotheses", [])
        if not isinstance(candidates, list):
            continue

        for item in candidates:
            if not isinstance(item, dict):
                continue

            statement = item.get("statement")
            prediction = item.get("testable_prediction")
            rationale = item.get("rationale")

            if not all(
                isinstance(value, str) and value.strip()
                for value in (statement, prediction, rationale)
            ):
                continue

            statement = statement.strip()
            dedup_key = statement.casefold()

            if dedup_key in seen_statements:
                continue

            # Never accept references to papers outside the KB.
            supporting_ids = item.get("supporting_paper_ids", [])
            if not isinstance(supporting_ids, list):
                supporting_ids = []

            supporting_ids = sorted({
                pid for pid in supporting_ids
                if isinstance(pid, str)
                and pid in valid_ids
                and pid in paper_ids
            })

            assumptions = item.get("assumptions", [])
            if not isinstance(assumptions, list):
                assumptions = []

            assumptions = [
                value.strip()
                for value in assumptions
                if isinstance(value, str) and value.strip()
            ]

            # Sanitize model-generated score components.
            raw_parts = item.get("score_parts", {})
            if not isinstance(raw_parts, dict):
                raw_parts = {}

            score_names = (
                "novelty",
                "evidence_support",
                "feasibility",
                "llm_judge",
            )
            score_parts = {}

            for name in score_names:
                value = raw_parts.get(name, 0.0)

                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    score_parts[name] = max(0.0, min(1.0, float(value)))
                else:
                    score_parts[name] = 0.0

            # Equal-weight preliminary aggregate.
            score = sum(score_parts.values()) / len(score_parts)

            seen_statements.add(dedup_key)
            hypotheses.append(Hypothesis(
                id=f"H{len(hypotheses) + 1}",
                gap_ids=[gap.id],
                statement=statement,
                testable_prediction=prediction.strip(),
                rationale=rationale.strip(),
                supporting_paper_ids=supporting_ids,
                assumptions=assumptions,
                score_parts=score_parts,
                score=score,
                status="candidate",
            ))

    return hypotheses