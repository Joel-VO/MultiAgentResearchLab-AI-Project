import unittest
from unittest import mock

import requests

from module_A_tooling.cache import Cache
from module_B_literature_kg.b0 import clients
from module_B_literature_kg.b0.clients import (ArxivClient, HttpError, NotFound, S2Client, call_with_retry,
                                 normalize_arxiv_id, normalize_doi, parse_arxiv_feed)
from tests.Module_B.B0.fakes  import (ARXIV_EMPTY_FEED, ARXIV_ERROR_FEED, ARXIV_FEED, FakeResponse, FakeSession,
                         s2_raw)


class IdTests(unittest.TestCase):
    def test_arxiv_ids(self):
        for text in ("2305.12345", "arXiv:2305.12345v3", "https://arxiv.org/abs/2305.12345",
                     "https://arxiv.org/pdf/2305.12345v2.pdf", " 2305.12345 "):
            self.assertEqual(normalize_arxiv_id(text), "2305.12345", text)
        self.assertEqual(normalize_arxiv_id("hep-th/9901001v1"), "hep-th/9901001")
        for text in ("Attention Is All You Need", "10.1145/3292500", "", None, "12345"):
            self.assertIsNone(normalize_arxiv_id(text), text)

    def test_dois(self):
        self.assertEqual(normalize_doi("https://doi.org/10.1145/ABC.1"), "10.1145/abc.1")
        self.assertEqual(normalize_doi("doi:10.1000/xyz"), "10.1000/xyz")
        self.assertIsNone(normalize_doi("2305.12345"))


class RetryTests(unittest.TestCase):
    def run_retry(self, replies):
        sleeps, it = [], iter(replies)
        def send():
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r
        try:
            out = call_with_retry(send, sleep=sleeps.append, base=1.0, factor=2.0, max_attempts=3)
        except Exception as exc:
            return exc, sleeps
        return out, sleeps

    def test_retries_then_succeeds_with_growing_waits(self):
        out, sleeps = self.run_retry([FakeResponse(503), FakeResponse(503), FakeResponse(200, {})])
        self.assertEqual(out.status_code, 200)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_gives_up_after_three_tries(self):
        out, sleeps = self.run_retry([FakeResponse(503)] * 3)
        self.assertIsInstance(out, HttpError)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_404_is_not_retried(self):
        out, sleeps = self.run_retry([FakeResponse(404)])
        self.assertIsInstance(out, NotFound)
        self.assertEqual(sleeps, [])

    def test_other_4xx_is_not_retried(self):
        out, sleeps = self.run_retry([FakeResponse(400)])
        self.assertIsInstance(out, HttpError)
        self.assertEqual(sleeps, [])

    def test_network_errors_are_retried(self):
        out, sleeps = self.run_retry([requests.ConnectionError("x"), FakeResponse(200, {})])
        self.assertEqual(out.status_code, 200)
        self.assertEqual(sleeps, [1.0])

    def test_retry_after_header_is_used(self):
        out, sleeps = self.run_retry([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {})])
        self.assertEqual(sleeps, [7.0])


class S2Tests(unittest.TestCase):
    def make(self, router):
        self.session = FakeSession(router)
        return S2Client(cache=Cache(":memory:"), session=self.session, api_key=None,
                        min_interval=0.0, sleep=lambda s: None)

    def test_get_paper_normalizes_and_caches(self):
        s2 = self.make(lambda path, params: FakeResponse(200, s2_raw("p1", "T", arxiv="2305.12345",
                                                                       doi="10.1000/ABC")))
        paper = s2.get_paper("ARXIV:2305.12345")
        self.assertEqual((paper["s2_id"], paper["arxiv_id"], paper["doi"], paper["key"]),
                         ("p1", "2305.12345", "10.1000/abc", "s2:p1"))
        self.assertEqual(s2.get_paper("ARXIV:2305.12345"), paper)
        self.assertEqual(len(self.session.calls), 1)          # the second call came from the cache

    def test_not_found_is_none_and_remembered(self):
        s2 = self.make(lambda path, params: FakeResponse(404))
        self.assertIsNone(s2.get_paper("DOI:10.1/x"))
        self.assertIsNone(s2.get_paper("DOI:10.1/x"))
        self.assertEqual(len(self.session.calls), 1)

    def test_api_key_header_is_sent(self):
        seen = {}
        class Sess(FakeSession):
            def get(self, url, params=None, headers=None, timeout=None):
                seen.update(headers or {})
                return super().get(url, params, headers, timeout)
        s2 = S2Client(cache=Cache(":memory:"), session=Sess(lambda p, q: FakeResponse(200, s2_raw("p", "T"))),
                      api_key="SECRET", min_interval=0.0, sleep=lambda s: None)
        s2.get_paper("p")
        self.assertEqual(seen.get("x-api-key"), "SECRET")

    def references_router(self, total, with_unresolved_at=None):
        def router(path, params):
            offset, limit = params["offset"], params["limit"]
            idx = list(range(offset, min(offset + limit, total)))
            data = []
            for i in idx:
                cited = {"paperId": None, "title": "unresolved"} if i == with_unresolved_at \
                    else s2_raw(f"r{i}", f"Ref {i}")
                data.append({"citedPaper": cited})
            body = {"offset": offset, "data": data}
            if offset + limit < total:
                body["next"] = offset + limit
            return FakeResponse(200, body)
        return router

    def test_references_pagination_and_cap(self):
        with mock.patch.object(clients, "S2_REFERENCES_PAGE_SIZE", 2):
            s2 = self.make(self.references_router(total=5))
            got = s2.get_references("seed", cap=10)
            self.assertEqual([r["s2_id"] for r in got["references"]], ["r0", "r1", "r2", "r3", "r4"])
            self.assertFalse(got["partial"])

            s2 = self.make(self.references_router(total=5))
            got = s2.get_references("seed", cap=3)
            self.assertEqual(len(got["references"]), 3)
            self.assertTrue(got["partial"])                    # more existed than the cap allowed

            s2 = self.make(self.references_router(total=3))    # exactly the cap, nothing beyond it
            got = s2.get_references("seed", cap=3)
            self.assertEqual(len(got["references"]), 3)
            self.assertFalse(got["partial"])

    def test_references_skip_unresolved_and_cache(self):
        with mock.patch.object(clients, "S2_REFERENCES_PAGE_SIZE", 2):
            s2 = self.make(self.references_router(total=4, with_unresolved_at=1))
            got = s2.get_references("seed", cap=10)
            self.assertEqual(len(got["references"]), 3)
            self.assertEqual(got["skipped_no_id"], 1)
            n = len(self.session.calls)
            s2.get_references("seed", cap=10)
            self.assertEqual(len(self.session.calls), n)

    def test_title_search(self):
        s2 = self.make(lambda path, params: FakeResponse(200, {"data": [s2_raw("a", "Alpha"), s2_raw("b", "Beta")]}))
        found = s2.search_title("alpha", 5)
        self.assertEqual([p["title"] for p in found], ["Alpha", "Beta"])

    def test_pacing_waits_between_calls(self):
        sleeps = []
        s2 = S2Client(cache=Cache(":memory:"), api_key=None, min_interval=1.0, sleep=sleeps.append,
                      clock=lambda: 100.0,
                      session=FakeSession(lambda p, q: FakeResponse(200, s2_raw("p", "T"))))
        s2.get_paper("a"); s2.get_paper("b")
        self.assertEqual(sleeps, [1.0])


class ArxivTests(unittest.TestCase):
    def test_parse_feed(self):
        paper = parse_arxiv_feed(ARXIV_FEED.format(aid="2305.12345"))
        self.assertEqual(paper["arxiv_id"], "2305.12345")
        self.assertEqual(paper["title"], "A Title On Two Lines")
        self.assertEqual(paper["abstract"], "The abstract text.")
        self.assertEqual((paper["year"], paper["authors"], paper["doi"]), (2023, ["Alan Turing"], "10.1000/xyz"))
        self.assertEqual(paper["key"], "arxiv:2305.12345")

    def test_error_and_empty_feeds_give_none(self):
        self.assertIsNone(parse_arxiv_feed(ARXIV_ERROR_FEED))
        self.assertIsNone(parse_arxiv_feed(ARXIV_EMPTY_FEED))
        self.assertIsNone(parse_arxiv_feed("not xml"))

    def test_client_caches(self):
        session = FakeSession(lambda path, params: FakeResponse(200, text=ARXIV_FEED.format(aid="2305.12345")))
        client = ArxivClient(cache=Cache(":memory:"), session=session, min_interval=0.0, sleep=lambda s: None)
        self.assertEqual(client.get_metadata("arXiv:2305.12345")["abstract"], "The abstract text.")
        client.get_metadata("2305.12345")
        self.assertEqual(len(session.calls), 1)
        self.assertIsNone(client.get_metadata("not an id"))


if __name__ == "__main__":
    unittest.main()
