"""
Sample inputs for Modules E and F, taken from the worked example in DESIGN.md Section 7
("Improve the factual reliability of question answering with a verification agent").

These stand in for what Modules B, C and D will produce, so E and F can be run
and demonstrated on their own.
"""
from module_A_tooling.schemas import ConstraintReport, Hypothesis, Proposal, RevisionTask
from module_B_literature_kg.interface import PaperMeta, PaperReview

HYPOTHESIS = Hypothesis(
    id="H1",
    gap_ids=["G1", "G2"],
    statement="A separate retrieval-grounded verifier agent improves factual accuracy "
              "over self-verification.",
    testable_prediction="On TruthfulQA, generator + separate retrieval-grounded verifier "
                        "scores higher accuracy than single-model self-verification.",
    rationale="Nine papers list an external verifier as future work, and no paper combines "
              "a verifier agent with retrieval on TruthfulQA.",
    supporting_paper_ids=["P1", "P2"],
    assumptions=["The 500-question subset is representative of the full benchmark."],
    score_parts={"novelty": 0.62, "evidence_support": 0.70, "feasibility": 0.85, "llm_judge": 0.68},
    score=0.71,
    status="selected",
)

PROPOSAL = Proposal(
    id="PR1",
    hypothesis_id="H1",
    research_question="Does a separate retrieval-grounded verifier improve factual accuracy "
                      "over self-verification?",
    dataset="TruthfulQA",
    method="generator + separate retrieval-grounded verifier",
    baselines=["self-verification"],
    metric="accuracy",
    setting="500-question subset, zero-shot, 8B generator",
    constraint_report=ConstraintReport(
        satisfied=["novelty", "license", "compute", "compatibility"], violated=[]),
)

# Revision tasks as a reviewer would raise them (DESIGN 7.7).
REVISION_TASKS = [
    RevisionTask(id="T1", type="ablation", severity="major",
                 text="Compare against self-verification given the same retrieval."),
    RevisionTask(id="T2", type="baseline", severity="major",
                 text="Add a compute-matched self-consistency baseline."),
    RevisionTask(id="T3", type="analysis", severity="minor",
                 text="Break errors down by question category."),
]

# Literature from the knowledge base (Module B output format).
PAPERS = [
    PaperMeta(
        paper_id="P1", title="Chain-of-Verification Reduces Hallucination in Large Language Models",
        abstract="We study the ability of language models to deliberate on the responses they give "
                 "in order to correct their mistakes. The model drafts an answer, plans verification "
                 "questions, answers them independently, and revises its answer. "
                 "Self-verification improves accuracy on TruthfulQA compared to the baseline model.",
        year=2023, venue="ACL Findings", citation_count=600, arxiv_id="2309.11495", is_seed=True),
    PaperMeta(
        paper_id="P2", title="Verifier Agents for Grounded Question Answering",
        abstract="We introduce a two-agent setup in which a generator answers questions and a "
                 "separate verifier agent checks each claim against retrieved documents. "
                 "A separate verifier with retrieval improves factual accuracy over single-model "
                 "self-checking on open-domain QA.",
        year=2024, venue="ICLR", citation_count=120),
    PaperMeta(
        paper_id="P3", title="Retrieval Helps Models Catch Their Own Errors",
        abstract="We analyse how retrieval changes error detection in language models. "
                 "Access to retrieved evidence makes models better at detecting their own factual "
                 "errors in summarisation tasks.",
        year=2023, venue=None, citation_count=45),
    PaperMeta(
        paper_id="P4", title="Does Verification Help Under Matched Compute?",
        abstract="Many verification methods spend more inference compute than their baselines. "
                 "We re-run several verification pipelines against compute-matched sampling baselines. "
                 "Under matched compute, verification gives no accuracy gain on TruthfulQA.",
        year=2024, venue="EMNLP", citation_count=300),
]

PAPER_REVIEWS = {
    "P1": PaperReview(paper_id="P1", title=PAPERS[0].title,
                      idea="Model drafts an answer, plans verification questions, answers them, revises",
                      method="single-model self-verification", datasets=["TruthfulQA"],
                      metrics=["accuracy"], methods=["self-verification"],
                      future_work=["use an external verifier"]),
    "P2": PaperReview(paper_id="P2", title=PAPERS[1].title,
                      idea="Separate verifier agent checks claims against retrieved documents",
                      method="generator + verifier agent with retrieval", datasets=["NaturalQuestions"],
                      metrics=["accuracy"], methods=["verifier agent", "retrieval"]),
    "P3": PaperReview(paper_id="P3", title=PAPERS[2].title,
                      idea="Retrieval improves self error detection", method="retrieval-augmented self-check",
                      datasets=["XSum"], metrics=["error detection F1"], methods=["retrieval"]),
    "P4": PaperReview(paper_id="P4", title=PAPERS[3].title,
                      idea="Verification gains vanish under compute-matched baselines",
                      method="compute-matched comparison", datasets=["TruthfulQA"],
                      metrics=["accuracy"], methods=["self-verification", "self-consistency"]),
}
