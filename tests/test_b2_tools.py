"""Tests for A2 (llm_client, local server only), quote_check, B2a (b2a_review) and compare_models.
No internet, no real LLM: a fake server plays the part of the model."""
import json
import tempfile
from pathlib import Path

import pytest

from module_A_tooling.cache import Cache
import requests

from module_A_tooling.llm_client import (LLMClient, LLMConfigError, LLMError, LLMInputTooLong, LLMTruncated,
                                         LocalServer, extract_json)
from module_B_literature_kg.b2 import compare_models as cm
from module_B_literature_kg.b2.b2a_review import (Claim, ReviewDraftB2a, build_review, clean_abstract,
                                                  review_paper)
from module_B_literature_kg.b2.quote_check import name_supported, quote_supported

TITLE = "Retrieval Augmented Generation for Question Answering"
ABSTRACT = ("We propose a retrieval-augmented generator that combines a dense retriever with a seq2seq model. "
            "On Natural Questions and TriviaQA it improves exact match by 5 points over strong baselines. "
            "A limitation is the high cost of indexing large corpora.")
Q_IDEA = "We propose a retrieval-augmented generator that combines a dense retriever with a seq2seq model"
Q_PRO = "On Natural Questions and TriviaQA it improves exact match by 5 points over strong baselines"
Q_CON = "A limitation is the high cost of indexing large corpora"      # a limitation sentence, which B2a must NOT turn into a con

GOOD = {"idea": {"text": "A retrieval-augmented generator.", "quote": Q_IDEA},
        "method": {"text": "Dense retriever plus seq2seq.", "quote": Q_IDEA},
        "pros": [{"text": "+5 exact match", "quote": Q_PRO}],
        "concepts": ["retrieval", "question answering"],
        "methods": ["dense retriever"], "datasets": ["Natural Questions", "TriviaQA"], "metrics": ["exact match"]}

URL = "http://localhost:11434/v1"
CFG = {"fake": {"base_url": URL, "model": "fake-1", "temperature": 0.0, "max_tokens": 100},
       "fake2": {"base_url": URL, "model": "fake-2", "temperature": 0.0, "max_tokens": 100},
       "small": {"base_url": URL, "model": "fake-3", "temperature": 0.0, "max_tokens": 100, "context_tokens": 200},
       "unset": {"base_url": URL, "model": "SET_ME", "temperature": 0.0, "max_tokens": 100}}


class FakeProvider:
    """Returns the queued replies one by one (a string, or an Exception to raise); records the prompts."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, spec, system, prompt, schema):
        self.prompts.append(prompt)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return (r if isinstance(r, str) else json.dumps(r)), 100, 50


def make_client(replies, cache=None):
    fake = FakeProvider(replies)
    client = LLMClient(cache=cache or Cache(":memory:"), models=CFG, provider=fake, sleep=lambda s: None)
    return client, fake


# ------------------------------------------------------------------ llm_client

def test_extract_json_handles_fence_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 1} Hope it helps') == {"a": 1}
    assert extract_json('<think>maybe {"b": 2} first</think>\n{"a": 1}') == {"a": 1}     # thinking models
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_call_returns_checked_data_and_tokens():
    client, _ = make_client([GOOD])
    res = client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake")
    assert res.data["datasets"] == ["Natural Questions", "TriviaQA"]
    assert (res.input_tokens, res.output_tokens, res.cached, res.attempts) == (100, 50, False, 1)


def test_second_identical_call_comes_from_cache():
    client, fake = make_client([GOOD])
    client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake")
    res = client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake")
    assert res.cached and len(fake.prompts) == 1


def test_different_model_or_version_is_a_different_question():
    client, fake = make_client([GOOD])
    client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake")
    client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake2")
    client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake", version="2")
    assert len(fake.prompts) == 3


def test_bad_json_is_sent_back_with_feedback_then_accepted():
    client, fake = make_client(["not json at all", GOOD])
    res = client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake")
    assert res.attempts == 2 and "could not be used" in fake.prompts[1]
    assert res.input_tokens == 200                       # both attempts are counted


def test_network_error_is_retried():
    client, fake = make_client([ConnectionError("boom"), GOOD])
    assert client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake").attempts == 2


def test_gives_up_after_max_attempts_and_stores_nothing():
    cache = Cache(":memory:")
    client, fake = make_client([ConnectionError("down")], cache=cache)
    with pytest.raises(LLMError):
        client.call("t", "sys", "prompt", ReviewDraftB2a, model="fake")
    assert len(fake.prompts) == 3 and cache.stats()["entries"] == 0


def test_config_errors_are_not_retried():
    client, fake = make_client([GOOD])
    with pytest.raises(LLMConfigError):
        client.call("t", "sys", "p", ReviewDraftB2a, model="nope")
    with pytest.raises(LLMConfigError):
        client.call("t", "sys", "p", ReviewDraftB2a, model="unset")
    assert fake.prompts == []


def test_input_too_long_for_the_context_is_refused_before_asking():
    client, fake = make_client([GOOD])
    with pytest.raises(LLMInputTooLong):
        client.call("t", "sys", "word " * 500, ReviewDraftB2a, model="small")
    assert fake.prompts == []


def test_one_question_too_long_gives_failed_review_not_a_crash():
    client, _ = make_client([GOOD])
    o = review_paper("P1", TITLE, ABSTRACT, client, "small")
    assert o.status == "failed" and "tokens needed" in o.error


# ------------------------------------------------------------------ the local server (fake HTTP)

class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def ok(text, p_tok=10, c_tok=5, finish="stop"):
    return FakeResp(200, {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                          "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok}})


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, timeout=None, **kw):
        self.calls.append({"url": url, "json": json, "timeout": timeout, **kw})
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def local_client(responses, models=None):
    session = FakeSession(responses)
    client = LLMClient(cache=Cache(":memory:"), models=models or CFG, provider=LocalServer(session),
                       sleep=lambda s: None)
    return client, session


def test_local_request_shape_has_no_key_and_asks_for_json_schema():
    client, session = local_client([ok(json.dumps(GOOD))])
    res = client.call("t", "SYS", "PROMPT", ReviewDraftB2a, model="fake")
    call = session.calls[0]
    assert call["url"] == URL + "/chat/completions" and "headers" not in call       # no Authorization header at all
    body = call["json"]
    assert body["model"] == "fake-1" and body["temperature"] == 0.0 and body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"]
    assert (res.input_tokens, res.output_tokens) == (10, 5) and res.data["datasets"] == GOOD["datasets"]


def test_structured_output_modes():
    for mode, expected in (("json_object", {"type": "json_object"}), ("none", None)):
        models = {"m": {**CFG["fake"], "structured_output": mode}}
        client, session = local_client([ok(json.dumps(GOOD))], models)
        client.call("t", "s", "p", ReviewDraftB2a, model="m")
        assert session.calls[0]["json"].get("response_format") == expected


def test_server_not_running_stops_with_a_clear_message():
    client, session = local_client([requests.ConnectionError("refused")])
    with pytest.raises(LLMConfigError) as e:
        client.call("t", "s", "p", ReviewDraftB2a, model="fake")
    assert "is it running" in str(e.value) and len(session.calls) == 1


def test_model_not_found_is_a_config_error():
    client, _ = local_client([FakeResp(404, text='{"error":"model not found, try pulling it first"}')])
    with pytest.raises(LLMConfigError) as e:
        client.call("t", "s", "p", ReviewDraftB2a, model="fake")
    assert "not found" in str(e.value)


def test_busy_server_is_retried():
    client, session = local_client([FakeResp(503), ok(json.dumps(GOOD))])
    assert client.call("t", "s", "p", ReviewDraftB2a, model="fake").attempts == 2 and len(session.calls) == 2


def test_slow_server_timeout_is_retried():
    client, session = local_client([requests.ReadTimeout("slow"), ok(json.dumps(GOOD))])
    assert client.call("t", "s", "p", ReviewDraftB2a, model="fake").attempts == 2


def test_cut_off_answer_is_reported_not_retried():
    client, session = local_client([ok('{"idea": {"text": "unfini', finish="length")])
    with pytest.raises(LLMTruncated):
        client.call("t", "s", "p", ReviewDraftB2a, model="fake")
    assert len(session.calls) == 1
    o = review_paper("P1", TITLE, ABSTRACT, client, "fake")
    assert o.status == "failed" and "max_tokens" in o.error


def test_server_side_context_error_is_one_failed_question():
    client, _ = local_client([FakeResp(400, text="This model's maximum context length is 4096 tokens")])
    with pytest.raises(LLMInputTooLong):
        client.call("t", "s", "p", ReviewDraftB2a, model="fake")


def test_thinking_model_reply_is_accepted():
    client, _ = local_client([ok("<think>let me see {x}</think>" + json.dumps(GOOD))])
    assert client.call("t", "s", "p", ReviewDraftB2a, model="fake").data["metrics"] == ["exact match"]


# ------------------------------------------------------------------ quote_check

def test_quote_must_be_real_long_enough_and_clean():
    texts = [TITLE, ABSTRACT]
    assert quote_supported(Q_PRO, texts)
    assert quote_supported(Q_PRO.upper().replace("  ", " "), texts)          # case does not matter
    assert not quote_supported("it improves exact match by 9 points", texts)  # changed number
    assert not quote_supported("exact match", texts)                         # real but too short
    assert not quote_supported("We propose ... a seq2seq model", texts)      # stitched
    assert not quote_supported("", texts)


def test_quote_survives_typography_differences():
    assert quote_supported("we propose a \u201cretrieval-augmented\u201d generator that combines",
                           ["We propose a \"retrieval-augmented\" generator that combines a dense retriever"])


def test_name_must_appear_as_a_whole_word():
    texts = [TITLE, ABSTRACT]
    assert name_supported("TriviaQA", texts) and name_supported("natural questions", texts)
    assert name_supported("retrieval augmented", texts)                      # hyphen vs space
    assert not name_supported("SQuAD", texts)
    assert not name_supported("QA", ["We use SQuAD"])                        # inside another word


# ------------------------------------------------------------------ b2a core

def test_clean_abstract():
    assert clean_abstract("<jats:p>Abstract: We  test\n p < 0.05 &amp; more</jats:p>") == "We test p < 0.05 & more"
    assert len(clean_abstract("word " * 2000, 100)) <= 100


def test_build_review_keeps_supported_and_records_quotes():
    review, counts, failed = build_review("P1", TITLE, ABSTRACT, GOOD)
    assert review.idea == "A retrieval-augmented generator." and review.pros == ["+5 exact match"]
    assert review.evidence_quotes == {"idea": Q_IDEA, "method": Q_IDEA, "pros.0": Q_PRO}
    assert counts["claims_made"] == counts["claims_kept"] == 3 and failed == []
    assert review.datasets == ["Natural Questions", "TriviaQA"]


def test_build_review_drops_unsupported_claims_and_invented_names():
    bad = json.loads(json.dumps(GOOD))
    bad["idea"]["quote"] = "the paper presents a completely new way of doing things"
    bad["pros"].append({"text": "state of the art", "quote": "it achieves state of the art results everywhere"})
    bad["datasets"].append("SQuAD")
    bad["metrics"] = ["F1", "exact match", "exact match"]
    review, counts, failed = build_review("P1", TITLE, ABSTRACT, bad)
    assert review.idea == "not stated" and "idea" not in review.evidence_quotes
    assert review.pros == ["+5 exact match"] and len(failed) == 2
    assert review.datasets == ["Natural Questions", "TriviaQA"] and review.metrics == ["exact match"]
    assert (counts["entities_made"], counts["entities_kept"]) == (6, 4)          # 6 names offered, 4 really in the text


def test_cons_and_future_work_stay_empty_even_if_the_model_offers_them():
    greedy = dict(GOOD, cons=[{"text": "indexing is costly", "quote": Q_CON}],
                  future_work=[{"text": "scale up", "quote": "we will scale this up to larger corpora"}])
    review, counts, _ = build_review("P1", TITLE, ABSTRACT, greedy)
    assert review.cons == [] and review.future_work == []
    assert not any(k.startswith(("cons", "future_work")) for k in review.evidence_quotes)
    assert counts["claims_made"] == 3                       # the extra claims are not even counted


def test_prompt_does_not_ask_for_cons_or_future_work():
    from module_B_literature_kg.b2.b2a_review import build_prompt
    prompt = build_prompt(TITLE, ABSTRACT)
    assert '"cons"' not in prompt and '"future_work"' not in prompt and "pros" in prompt


def test_concepts_are_capped_and_deduplicated():
    d = dict(GOOD, concepts=[f"c{i}" for i in range(20)] + ["C1"])
    review, _, _ = build_review("P1", TITLE, ABSTRACT, d)
    assert len(review.concepts) == 8


def test_review_paper_ok():
    client, fake = make_client([GOOD])
    o = review_paper("P7", TITLE, ABSTRACT, client, "fake")
    assert o.status == "ok" and not o.retried and o.review.paper_id == "P7" and o.claims_kept == 3
    assert len(fake.prompts) == 1


def test_review_retries_once_when_most_quotes_fail_and_keeps_the_better_answer():
    bad = {**GOOD, "idea": {"text": "x", "quote": "made up words that are not in the abstract at all"},
           "method": {"text": "y", "quote": "also made up words that are not in the abstract"},
           "pros": [{"text": "z", "quote": "yet another invented sentence that is not present here"}]}
    client, fake = make_client([bad, GOOD])
    o = review_paper("P1", TITLE, ABSTRACT, client, "fake")
    assert o.retried and len(fake.prompts) == 2 and "NOT found word for word" in fake.prompts[1]
    assert o.claims_kept == 3 and o.input_tokens == 200


def test_review_does_not_retry_when_only_a_few_fail():
    one_bad = json.loads(json.dumps(GOOD))
    one_bad["pros"].append({"text": "fast", "quote": "the system is very fast in every situation we tried"})
    client, fake = make_client([one_bad])
    o = review_paper("P1", TITLE, ABSTRACT, client, "fake")
    assert not o.retried and len(fake.prompts) == 1 and o.claims_kept == 3 and o.claims_made == 4


def test_unusable_llm_gives_empty_review_not_a_crash():
    client, _ = make_client([ConnectionError("down")])
    o = review_paper("P3", TITLE, ABSTRACT, client, "fake")
    assert o.status == "failed" and o.review.idea == "not stated" and o.review.pros == [] and o.error


def test_nothing_supported_is_flagged_empty():
    client, _ = make_client([{"idea": {"text": "not stated"}, "method": {"text": "not stated"}}])
    assert review_paper("P1", TITLE, ABSTRACT, client, "fake").status == "empty"


def test_setup_problems_stop_the_run():
    client, _ = make_client([GOOD])
    with pytest.raises(LLMConfigError):
        review_paper("P1", TITLE, ABSTRACT, client, "unset")


# ------------------------------------------------------------------ compare_models

def _papers():
    long_abs = ABSTRACT
    return {"P1": {"paper_id": "P1", "title": "seed", "abstract": long_abs, "is_seed": True},
            "P2": {"paper_id": "P2", "title": TITLE, "abstract": long_abs, "is_seed": False},
            "P3": {"paper_id": "P3", "title": "short", "abstract": "too short", "is_seed": False},
            "P4": {"paper_id": "P4", "title": TITLE + " II", "abstract": long_abs, "is_seed": False}}


def test_sample_skips_seeds_and_short_abstracts_and_is_repeatable():
    s1, s2 = cm.sample_papers(_papers(), 10), cm.sample_papers(_papers(), 10)
    assert sorted(p["paper_id"] for p in s1) == ["P2", "P4"] and s1 == s2
    assert len(cm.sample_papers(_papers(), 1)) == 1


def test_run_comparison_table_files_and_skips():
    papers = cm.sample_papers(_papers(), 10)
    client, _ = make_client([GOOD])
    with tempfile.TemporaryDirectory() as d:
        rep = cm.run_comparison(papers, ["fake", "fake2", "unset"], client, Path(d))
        assert (Path(d) / "summary.json").exists() and (Path(d) / "fake.json").exists()
        assert "TriviaQA" in (Path(d) / "side_by_side.md").read_text()
    s = {x["model"]: x for x in rep["summaries"]}
    assert set(s) == {"fake", "fake2"} and "unset" in rep["skipped"]
    assert s["fake"]["quote_pass_rate"] == 1.0 and s["fake"]["claims_kept_per_paper"] == 3.0
    assert rep["agreement"]["fake vs fake2"] == 1.0
    assert "SKIPPED unset" in rep["table"] and "quote pass" in rep["table"]


def test_summary_counts_failures_and_cost():
    client, _ = make_client([ConnectionError("down")])
    papers = cm.sample_papers(_papers(), 10)
    outcomes = [review_paper(p["paper_id"], p["title"], p["abstract"], client, "fake") for p in papers]
    s = cm.summarise("fake", outcomes)
    assert s["failed"] == 2 and s["quote_pass_rate"] is None and s["est_minutes_per_150_papers"] == 0.0
