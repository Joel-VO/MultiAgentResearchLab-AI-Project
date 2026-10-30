import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any
from module_A_tooling.local_llm_inference import OllamaClient
from module_B_literature_kg.interface import (
    KBHandle,
    PaperReview,
    Conflict,
    KnowledgeBase,
)
from classes import Gap, Hypothesis, ResearchRequest


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
    """
    Discover research gaps using deterministic, rule-based methods.

    Gap sources:
      - sparse_region
      - missing_combination
      - conflict
      - future_work

    No LLM calls are made in this function.
    """

    gaps: list[Gap] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    def add_gap(
        source: str,
        description: str,
        paper_ids: list[str] | None = None,
    ) -> None:
        """Add a gap once, using a stable key to avoid duplicates."""
        ids = sorted(set(paper_ids or []))
        key = (
            source,
            " ".join(description.lower().split()),
            tuple(ids),
        )

        if key in seen or len(gaps) >= max_gaps:
            return

        seen.add(key)
        gaps.append(
            Gap(
                id=f"G{len(gaps) + 1}",
                source=source,
                description=description.strip(),
                paper_ids=ids,
            )
        )

    # ---------------------------------------------------------
    # 1. Future work and limitations from paper reviews
    # ---------------------------------------------------------
    valid_reviews = [
        review
        for review in reviews
        if getattr(review, "paper_id", None)
    ]

    for review in valid_reviews:
        paper_id = review.paper_id
        title = getattr(review, "title", "") or paper_id

        future_work = getattr(review, "future_work", []) or []
        limitations = getattr(review, "cons", []) or []

        # Future work is explicitly represented in the Gap schema.
        for item in future_work:
            if not isinstance(item, str):
                continue

            item = item.strip()
            if not item or item.lower() in {
                "not stated",
                "none",
                "n/a",
                "unknown",
            }:
                continue

            add_gap(
                source="future_work",
                description=(
                    f"Future-work opportunity identified in {paper_id} "
                    f"({title}): {item}"
                ),
                paper_ids=[paper_id],
            )

        # The shared Gap schema has no separate "limitation" source,
        # so limitations are represented as future-work opportunities.
        for item in limitations:
            if not isinstance(item, str):
                continue

            item = item.strip()
            if not item or item.lower() in {
                "not stated",
                "none",
                "n/a",
                "unknown",
            }:
                continue

            add_gap(
                source="future_work",
                description=(
                    f"Potential limitation to investigate in {paper_id} "
                    f"({title}): {item}"
                ),
                paper_ids=[paper_id],
            )

    # ---------------------------------------------------------
    # 2. Contradictory findings and reviewer disagreements
    # ---------------------------------------------------------
    for conflict in conflicts:
        paper_ids = getattr(conflict, "paper_ids", []) or []
        summary = getattr(conflict, "summary", "") or ""
        evidence = getattr(conflict, "evidence_text", "") or ""
        conflict_type = getattr(conflict, "type", "finding")

        description = (
            f"Research gap arising from {conflict_type} conflict "
            f"{getattr(conflict, 'id', 'unknown')}: {summary}"
        )

        if evidence.strip():
            description += f" Evidence: {evidence.strip()}"

        add_gap(
            source="conflict",
            description=description,
            paper_ids=paper_ids,
        )

    # ---------------------------------------------------------
    # 3. Missing concept/method and method/dataset combinations
    # ---------------------------------------------------------
    concept_papers: dict[str, set[str]] = defaultdict(set)
    method_papers: dict[str, set[str]] = defaultdict(set)
    dataset_papers: dict[str, set[str]] = defaultdict(set)

    concept_method_pairs: set[tuple[str, str]] = set()
    method_dataset_pairs: set[tuple[str, str]] = set()

    def clean_items(values: Any) -> list[str]:
        """Normalize strings and remove empty or unspecified entries."""
        if not values:
            return []

        if isinstance(values, str):
            values = [values]

        result = []
        for value in values:
            if not isinstance(value, str):
                continue

            value = " ".join(value.strip().split())
            if value and value.lower() not in {
                "not stated",
                "none",
                "n/a",
                "unknown",
            }:
                result.append(value)

        return sorted(set(result), key=str.casefold)

    for review in valid_reviews:
        paper_id = review.paper_id

        concepts = clean_items(getattr(review, "concepts", []))
        methods = clean_items(getattr(review, "methods", []))
        datasets = clean_items(getattr(review, "datasets", []))

        for concept in concepts:
            concept_papers[concept].add(paper_id)

        for method in methods:
            method_papers[method].add(paper_id)

        for dataset in datasets:
            dataset_papers[dataset].add(paper_id)

        for concept in concepts:
            for method in methods:
                concept_method_pairs.add((concept, method))

        for method in methods:
            for dataset in datasets:
                method_dataset_pairs.add((method, dataset))

    all_paper_ids = {
        review.paper_id for review in valid_reviews
    }

    # Only report pairs among terms observed in the reviewed literature.
    concepts = sorted(concept_papers, key=str.casefold)
    methods = sorted(method_papers, key=str.casefold)
    datasets = sorted(dataset_papers, key=str.casefold)

    for concept in concepts:
        for method in methods:
            if (concept, method) in concept_method_pairs:
                continue

            relevant_ids = sorted(
                concept_papers[concept] | method_papers[method]
            )

            add_gap(
                source="missing_combination",
                description=(
                    f"No reviewed paper combines concept '{concept}' "
                    f"with method '{method}'. Investigate whether "
                    f"combining them could address the research topic: "
                    f"{request.topic}"
                ),
                paper_ids=relevant_ids,
            )

            if len(gaps) >= max_gaps:
                return gaps

    for method in methods:
        for dataset in datasets:
            if (method, dataset) in method_dataset_pairs:
                continue

            relevant_ids = sorted(
                method_papers[method] | dataset_papers[dataset]
            )

            add_gap(
                source="missing_combination",
                description=(
                    f"No reviewed paper combines method '{method}' "
                    f"with dataset '{dataset}'. Evaluate whether this "
                    f"combination is appropriate for the research topic: "
                    f"{request.topic}"
                ),
                paper_ids=relevant_ids,
            )

            if len(gaps) >= max_gaps:
                return gaps

    # ---------------------------------------------------------
    # 4. Sparse regions in the similarity graph near the topic
    # ---------------------------------------------------------
    graph = getattr(kb, "graph", None)

    if graph is not None and hasattr(graph, "nodes"):
        # Restrict analysis to paper nodes.
        paper_nodes = {
            node
            for node, attrs in graph.nodes(data=True)
            if attrs.get("type") == "paper"
        }

        # Use the knowledge base's topic vector when available.
        topic_vector = getattr(kb, "topic_vector", None)

        if callable(topic_vector):
            try:
                topic_vector = topic_vector()
            except Exception:
                topic_vector = None

        # Determine which papers are near the topic.
        # If no usable topic vector is available, retain the paper
        # nodes in the graph rather than inventing topic similarities.
        near_topic: set[str] = set()

        if topic_vector is not None and hasattr(kb, "closest_to_text"):
            # This method expects text, so do not pass a vector to it.
            # Use the request topic for the nearest-paper query instead.
            try:
                nearest = kb.closest_to_text(
                    request.topic,
                    k=max(1, len(paper_nodes)),
                )

                for item in nearest or []:
                    if isinstance(item, str):
                        near_topic.add(item)
                    elif isinstance(item, (tuple, list)) and item:
                        near_topic.add(str(item[0]))
                    elif hasattr(item, "paper_id"):
                        near_topic.add(str(item.paper_id))

            except (TypeError, ValueError, AttributeError):
                near_topic = set()

        # If topic-neighbor lookup is unavailable, skip sparse-region
        # discovery rather than incorrectly claiming papers are near topic.
        if near_topic:
            near_topic &= paper_nodes

            # Similarity edges indicate semantic closeness. Citation
            # edges are not treated as semantic similarity.
            similarity_graph = graph.subgraph(near_topic).copy()

            for u, v, attrs in graph.edges(data=True):
                if u not in near_topic or v not in near_topic:
                    continue

                if attrs.get("kind") != "similar":
                    continue

                weight = attrs.get("weight")

                try:
                    weight = float(weight)
                except (TypeError, ValueError):
                    continue

                if weight >= sparse_similarity_threshold:
                    similarity_graph.add_edge(u, v, weight=weight)

            # Low-degree papers in the topic neighborhood are candidates
            # for sparse regions. Group connected candidates to avoid
            # emitting one gap for every isolated paper.
            sparse_nodes = {
                node
                for node in near_topic
                if similarity_graph.degree(node) < min_sparse_region_size
            }

            sparse_subgraph = similarity_graph.subgraph(sparse_nodes)

            for component in __import__("networkx").connected_components(
                sparse_subgraph
            ):
                component_ids = sorted(str(node) for node in component)

                if not component_ids:
                    continue

                add_gap(
                    source="sparse_region",
                    description=(
                        f"Potentially sparse semantic region near the "
                        f"research topic '{request.topic}', containing "
                        f"{len(component_ids)} paper(s): "
                        f"{', '.join(component_ids)}. Investigate whether "
                        f"this region represents an underexplored direction."
                    ),
                    paper_ids=component_ids,
                )

                if len(gaps) >= max_gaps:
                    return gaps

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