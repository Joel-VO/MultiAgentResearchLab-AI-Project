import json
from typing import Any
from typing import Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from module_A_tooling.local_llm_inference import OllamaClient


@dataclass
class ResearchAgent:
    """One independent agent identity, client, and private memory."""
    name: str
    role: str
    objective: str
    llm: OllamaClient
    memory: list[dict[str, Any]] = field(default_factory=list)

    def respond(
        self,
        *,
        topic: str,
        gap: dict,
        evidence: list[dict],
        shared_messages: list[dict],
        round_no: int,
        phase: str,
    ) -> dict:

        def safe_json(raw: str) -> dict:
            """Parse JSON returned by Ollama, tolerating a surrounding code fence."""
            if not isinstance(raw, str):
                raise ValueError("LLM response was not text")
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("LLM response did not contain a JSON object")
            parsed = json.loads(text[start:end + 1])
            if not isinstance(parsed, dict):
                raise ValueError("LLM JSON response must be an object")
            return parsed

        prompt = {
            "topic": topic,
            "gap": gap,
            "evidence": evidence,
            "round": round_no,
            "phase": phase,
            "agent": {"name": self.name, "role": self.role,
                      "objective": self.objective},
            "private_memory": self.memory[-12:],
            "shared_discussion": shared_messages[-40:],
            "rules": [
                "Act only in your assigned role and phase.",
                "Address specific earlier messages by agent name when relevant.",
                "Use only supplied evidence for claims about papers.",
                "Never invent paper IDs, findings, quotations, or citations.",
                "Clearly distinguish evidence, interpretation, and assumptions.",
                "A corpus gap does not prove global novelty.",
                "For critique phase, focus on critiques; do not introduce unrelated hypotheses.",
                "For revision phase, explicitly respond to critiques and revise, defend, or reject candidates.",
                "Return valid JSON only, with no Markdown fences.",
            ],
            "required_json": {
                "message": "A concise message to the team",
                "addressed_agents": ["agent names being addressed"],
                "critiques": [
                    {"target": "candidate statement or agent",
                        "issue": "problem", "suggestion": "how to address it"}
                ],
                "hypotheses": [
                    {
                        "statement": "falsifiable claim",
                        "testable_prediction": "observable result that could falsify the claim",
                        "rationale": "evidence-based rationale",
                        "supporting_paper_ids": [],
                        "assumptions": [],
                        "score_parts": {
                            "evidence_support": 0.0,
                            "feasibility": 0.0,
                            "llm_judge": 0.0,
                        },
                    }
                ],
            },
        }
        raw = self.llm.generate(
            user_prompt=json.dumps(prompt, ensure_ascii=False, default=str),
            system_prompt=(
                f"You are {self.name}, the {self.role}. {self.objective} "
                "You are an independent scientific agent communicating through a shared discussion. "
                "Be skeptical, specific, and evidence-bound. Return only valid JSON."
            ),
            temperature=0,
        )
        result = safe_json(raw)
        if not isinstance(result, dict):
            raise ValueError("Agent response must be a JSON object")
        message = {
            "round": round_no,
            "phase": phase,
            "agent": self.name,
            "role": self.role,
            "message": str(result.get("message", "")).strip(),
            "addressed_agents": result.get("addressed_agents", []),
            "critiques": result.get("critiques", []),
            "hypotheses": result.get("hypotheses", []),
        }
        self.memory.append(message)
        return message


class Gap(BaseModel):
    id: str
    source: Literal["sparse_region",
                    "missing_combination", "conflict", "future_work"]
    description: str
    paper_ids: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    id: str
    gap_ids: list[str] = Field(default_factory=list)
    statement: str
    testable_prediction: str
    rationale: str
    supporting_paper_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    # novelty, evidence_support, feasibility, llm_judge
    score_parts: dict[str, float] = Field(default_factory=dict)
    score: float
    status: Literal["candidate", "selected", "rejected", "revised"]


# Dummy classes of module B

class ResearchRequest:
    topic: str
    seed_papers: list[str]  # DOI, arXiv ID, or title
    # TODO: budget: Budget  # max_papers, max_depth, fanout_per_node
