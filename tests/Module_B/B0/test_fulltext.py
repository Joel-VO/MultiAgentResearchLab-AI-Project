import io
import shutil
import tempfile
import unittest
from pathlib import Path

from module_A_tooling.cache import Cache
from module_B_literature_kg.b0 import fulltext as ft
from module_B_literature_kg.b0.fulltext import (FulltextFetcher, PdfLibraryMissing, Thresholds, classify_html,
                                  classify_pdf, extract_html, extract_pdf_text, get_seed_fulltext,
                                  normalize_for_matching, order_blocks, pick_best, run_fulltext_step)
from module_B_literature_kg.b0.manifest import Manifest, atomic_write_json, read_json
from tests.Module_B.B0.fakes import FakeResponse, FakeSession

TITLE = "Attention Is All You Need"
SENT = "The quick brown fox jumps over the lazy dog while the model learns to attend. "
TH = Thresholds(min_chars=300, min_chars_per_page=50)          # small numbers so test pages can be short


def para(n=4):
    return f'<div class="ltx_para"><p class="ltx_p">{SENT * n}</p></div>'


def ar5iv_html(title=TITLE, abstract=True, intro=True, refs=True, math=True, table=True, empty_table=False,
               errors=0, hidden_text=False, extra_sections=2):
    parts = [f'<html><head><title>{title}</title></head><body><nav class="ltx_page_navbar">NAV</nav>'
             f'<article class="ltx_document"><h1 class="ltx_title ltx_title_document">{title}</h1>'
             '<div class="ltx_authors">Ada Lovelace, Alan Turing</div>']
    if abstract:
        parts.append('<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>'
                     f'<p class="ltx_p">{SENT * 3}</p></div>')
    if intro:
        formula = '<math alttext="x^2 + y^2"><semantics><mrow><mi>x</mi></mrow></semantics></math>' if math else ""
        parts.append('<section class="ltx_section"><h2 class="ltx_title ltx_title_section">'
                     f'<span class="ltx_tag">1 </span>Introduction</h2><div class="ltx_para"><p class="ltx_p">'
                     f'{SENT * 4} We study {formula} here.</p></div></section>')
    for i in range(extra_sections):
        parts.append(f'<section class="ltx_section"><h2 class="ltx_title ltx_title_section">{i + 2} Part {i}</h2>'
                     f'{para()}</section>')
    if table:
        cells = "" if empty_table else "<td>Model</td><td>BLEU</td>"
        row2 = "" if empty_table else "<tr><td>Ours</td><td>28.4</td></tr>"
        parts.append('<figure class="ltx_table"><figcaption class="ltx_caption">Table 1: Results.</figcaption>'
                     f'<table class="ltx_tabular"><tr>{cells}</tr>{row2}</table></figure>')
    for _ in range(errors):
        parts.append('<span class="ltx_ERROR undefined">\\badmacro</span>')
    if hidden_text:                                   # text in a wrapper the extractor does not understand
        parts.append(f'<div class="mystery">{SENT * 40}</div>')
    if refs:
        parts.append('<section class="ltx_bibliography"><h2 class="ltx_title ltx_title_bibliography">References</h2>'
                     '<ul class="ltx_biblist"><li class="ltx_bibitem">[1] Vaswani et al. 2017.</li></ul></section>')
    parts.append('</article><footer class="ltx_page_footer">FOOT</footer></body></html>')
    return "".join(parts)


def make_pdf(title=TITLE, pages=3, headers=True):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for p in range(pages):
        if headers:
            c.drawString(72, 800, "Journal of Tests 2017")
            c.drawString(300, 30, str(p + 1))
        y = 760
        if p == 0:
            c.drawString(72, y, title)
            y -= 30
            c.drawString(72, y, "Abstract")
            y -= 20
        for i in range(12):
            c.drawString(72, y, SENT[:78])
            y -= 16
        if p == 0:
            c.drawString(72, y, "1 Introduction")
        if p == 1:
            c.drawString(72, y, "2 Method")
        if p == pages - 1:
            c.drawString(72, y - 20, "References")
        c.showPage()
    c.save()
    return buf.getvalue()


def pdf_pages(title=TITLE, n=3, with_refs=True):
    """Page texts as a good reader would give them."""
    pages = []
    for p in range(n):
        lines = ["Journal of Tests 2017"]
        if p == 0:
            lines += [title, "", "Abstract", SENT * 3, "", "1 Introduction"]
        if p == 1:
            lines += ["2 Method"]
        lines += ["", SENT * 6, "", SENT * 4]
        if p == n - 1 and with_refs:
            lines += ["", "References"]
        lines += [str(p + 1)]
        pages.append("\n".join(lines))
    return pages


class HtmlTests(unittest.TestCase):
    def test_good_page_is_ok(self):
        ex = extract_html(ar5iv_html())
        status, reason, warnings = classify_html(ex, TITLE, TH)
        self.assertEqual((status, reason, warnings), ("ok", None, []))
        self.assertIn("$x^2 + y^2$", ex["text"])                 # formula kept as LaTeX
        self.assertIn("Model | BLEU", ex["text"])                # table rows kept
        self.assertIn("Table 1: Results.", ex["text"])
        self.assertNotIn("NAV", ex["text"])
        self.assertNotIn("Ada Lovelace", ex["text"])
        self.assertGreaterEqual(len(ex["sections"]), 4)

    def test_page_without_a_paper_fails(self):
        ex = extract_html("<html><head><title>Page not found</title></head><body>Sorry.</body></html>")
        self.assertEqual(classify_html(ex, TITLE, TH)[:2], ("failed", "ar5iv_error_page"))

    def test_short_text_is_never_success(self):
        html = ('<article class="ltx_document"><h1 class="ltx_title ltx_title_document">' + TITLE +
                '</h1><p class="ltx_p">Tiny.</p></article>')
        self.assertEqual(classify_html(extract_html(html), TITLE, TH)[:2], ("failed", "too_short"))

    def test_wrong_paper_fails(self):
        ex = extract_html(ar5iv_html(title="Deep Residual Learning For Images"))
        self.assertEqual(classify_html(ex, TITLE, TH)[:2], ("failed", "title_mismatch"))

    def test_one_minor_warning_stays_ok_two_make_degraded(self):
        status, _, w = classify_html(extract_html(ar5iv_html(refs=False)), TITLE, TH)
        self.assertEqual((status, [x["code"] for x in w]), ("ok", ["no_references_section"]))
        status, _, w = classify_html(extract_html(ar5iv_html(refs=False, empty_table=True)), TITLE, TH)
        self.assertEqual(status, "degraded")
        self.assertEqual({x["code"] for x in w}, {"no_references_section", "tables_lost"})

    def test_text_in_unknown_wrapper_lowers_coverage(self):
        status, _, w = classify_html(extract_html(ar5iv_html(hidden_text=True)), TITLE, TH)
        self.assertEqual(status, "degraded")
        self.assertIn("low_coverage", [x["code"] for x in w])

    def test_few_sections_is_critical(self):
        status, _, w = classify_html(extract_html(ar5iv_html(extra_sections=0, abstract=False, refs=False)), TITLE, TH)
        self.assertEqual(status, "degraded")
        self.assertIn("few_sections", [x["code"] for x in w])

    def test_neither_abstract_nor_intro_is_critical(self):
        _, _, w = classify_html(extract_html(ar5iv_html(abstract=False, intro=False)), TITLE, TH)
        self.assertIn("no_abstract_or_intro", {x["code"]: x["severity"] for x in w})
        self.assertEqual({x["code"]: x["severity"] for x in w}["no_abstract_or_intro"], "critical")

    def test_conversion_errors_minor_then_critical(self):
        _, _, few = classify_html(extract_html(ar5iv_html(errors=2)), TITLE, TH)
        self.assertEqual({x["code"]: x["severity"] for x in few}["conversion_errors"], "minor")
        status, _, many = classify_html(extract_html(ar5iv_html(errors=9)), TITLE, TH)
        self.assertEqual({x["code"]: x["severity"] for x in many}["conversion_errors"], "critical")
        self.assertEqual(status, "degraded")


class PdfTextTests(unittest.TestCase):
    def test_order_blocks_two_columns(self):
        W = 600
        title = (50, 10, 550, 40, "TITLE")
        l1, l2 = (50, 60, 280, 100, "L1"), (50, 110, 280, 150, "L2")
        r1, r2 = (320, 60, 550, 100, "R1"), (320, 110, 550, 150, "R2")
        got = [b[4] for b in order_blocks([r1, l2, title, r2, l1], W)]
        self.assertEqual(got, ["TITLE", "L1", "L2", "R1", "R2"])

    def test_order_blocks_one_column_is_top_to_bottom(self):
        a, b, c = (50, 10, 550, 40, "A"), (50, 50, 550, 80, "B"), (120, 90, 300, 120, "C")
        self.assertEqual([x[4] for x in order_blocks([c, a, b], 600)], ["A", "B", "C"])

    def test_cleaning(self):
        pages = ["Running Title 1\nAbstract\nThis is a trans-\nformer paper with a \ufb01ne \ufb02ow.\n1",
                 "Running Title 2\n1 Introduction\nMore text here.\n2",
                 "Running Title 3\nReferences\n[1] Someone.\n3"]
        ex = extract_pdf_text(pages, "pymupdf")
        self.assertNotIn("Running Title", ex["text"])
        self.assertIn("transformer paper with a fine flow.", ex["text"])      # hyphen joined, ligatures fixed
        self.assertEqual([s["title"] for s in ex["sections"]], ["Abstract", "1 Introduction", "References"])
        self.assertGreater(ex["stats"]["removed_header_footer_lines"], 0)

    def test_garbage_and_readable_ratios(self):
        self.assertGreater(ft.garbage_ratio("ab (cid:12)(cid:13) \ufffd\ufffd cd"), 0.4)
        self.assertEqual(ft.garbage_ratio("clean text only"), 0.0)
        self.assertGreater(ft.readable_word_ratio("The model reached 28.4 BLEU on WMT-14 = + ="), 0.95)
        self.assertLess(ft.readable_word_ratio("xq#z kd$$f lw@@e " * 20), 0.3)

    def test_good_pdf_is_ok(self):
        ex = extract_pdf_text(pdf_pages(), "pymupdf")
        status, reason, w = classify_pdf(ex, TITLE, TH)
        self.assertEqual((status, reason), ("ok", None))
        self.assertEqual({x["code"] for x in w}, {"header_footer_removed", "equations_unreliable"})
        self.assertTrue(all(x["severity"] == "info" for x in w))             # info notes never count

    def test_pdf_failures_and_warnings(self):
        self.assertEqual(classify_pdf(extract_pdf_text(["tiny"], "pymupdf"), TITLE, TH)[:2], ("failed", "too_short"))
        wrong = classify_pdf(extract_pdf_text(pdf_pages(title="Deep Residual Learning For Images"), "pymupdf"), TITLE, TH)
        self.assertEqual(wrong[:2], ("failed", "title_mismatch"))
        sparse = extract_pdf_text(pdf_pages(n=3), "pymupdf")
        low = classify_pdf(sparse, TITLE, Thresholds(min_chars=300, min_chars_per_page=10 ** 6))
        self.assertEqual(low[:2], ("failed", "low_text_per_page"))
        garbled = pdf_pages()
        garbled[1] = garbled[1] + "\n" + "(cid:77)" * 400
        status, _, w = classify_pdf(extract_pdf_text(garbled, "pymupdf"), TITLE, TH)
        self.assertEqual(status, "degraded")
        self.assertIn("high_garbage", [x["code"] for x in w])

    def test_fallback_reader_is_a_minor_warning(self):
        status, _, w = classify_pdf(extract_pdf_text(pdf_pages(), "pypdf"), TITLE, TH)
        self.assertIn("pdf_reader_fallback", [x["code"] for x in w])
        self.assertEqual(status, "ok")                                       # one minor alone is not enough

    def test_real_pdf_through_default_reader(self):
        pages, reader = ft.read_pdf_pages(make_pdf())
        ex = extract_pdf_text(pages, reader)
        status, reason, _ = classify_pdf(ex, TITLE, TH)
        self.assertNotEqual(status, "failed", reason)
        self.assertIn("Attention Is All You Need", ex["text"])
        self.assertNotIn("Journal of Tests 2017", ex["text"])                # running header removed


class PickBestTests(unittest.TestCase):
    def cand(self, source, status, warnings=()):
        return {"source": source, "status": status, "warnings": [{"code": c, "severity": s} for c, s in warnings]}

    def test_ranking(self):
        ok_pdf, deg_html = self.cand("arxiv_pdf", "ok"), self.cand("ar5iv", "degraded", [("x", "critical")])
        self.assertIs(pick_best([deg_html, ok_pdf]), ok_pdf)
        failed = self.cand("arxiv_pdf", "failed")
        self.assertIs(pick_best([failed, deg_html]), deg_html)
        mild, harsh = self.cand("s2_pdf", "degraded", [("a", "minor"), ("b", "minor")]), \
            self.cand("arxiv_pdf", "degraded", [("c", "critical")])
        self.assertIs(pick_best([harsh, mild]), mild)                         # fewer/milder warnings win
        tie_a, tie_b = self.cand("arxiv_pdf", "ok"), self.cand("ar5iv", "ok")
        self.assertIs(pick_best([tie_a, tie_b]), tie_b)                       # tie goes to ar5iv
        self.assertIsNone(pick_best([]))


class FlowTests(unittest.TestCase):
    AR5IV = ft.AR5IV_BASE_URL + "2305.12345"
    APDF = ft.ARXIV_PDF_BASE_URL + "2305.12345"
    S2PDF = "https://publisher.example/paper.pdf"
    paper = {"key": "s2:p1", "title": TITLE, "arxiv_id": "2305.12345", "open_access_pdf_url": S2PDF}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fetcher(self, routes, reader=None):
        def router(url, params):
            r = routes.get(url)
            return r() if callable(r) else (r or FakeResponse(404))
        self.session = FakeSession(router)
        return FulltextFetcher(cache=Cache(":memory:"), session=self.session, pdf_dir=self.tmp / "pdfs",
                               min_interval=0.0, sleep=lambda s: None, now=lambda: 1000.0,
                               pdf_reader=reader or (lambda b: (pdf_pages(), "pymupdf")))

    def pdf_ok(self):
        return FakeResponse(200, content=b"%PDF-1.4 fake pdf bytes")

    def go(self, routes, reader=None, paper=None):
        return get_seed_fulltext(paper or self.paper, self.fetcher(routes, reader), TH)

    def urls(self):
        return [c[0] for c in self.session.calls]

    def test_good_ar5iv_makes_no_pdf_request(self):
        res = self.go({self.AR5IV: FakeResponse(200, text=ar5iv_html())})
        self.assertEqual((res["has_full_text"], res["best"]["source"], res["best"]["status"]), (True, "ar5iv", "ok"))
        self.assertEqual(self.urls(), [self.AR5IV])
        self.assertEqual(res["best"]["provenance"]["extractor_version"], ft.EXTRACTOR_VERSION)
        self.assertEqual(res["best"]["provenance"]["fetched_at"], 1000.0)

    def test_ar5iv_missing_falls_back_to_arxiv_pdf(self):
        res = self.go({self.APDF: self.pdf_ok()})
        self.assertEqual((res["best"]["source"], res["best"]["status"]), ("arxiv_pdf", "ok"))
        self.assertEqual([a["status"] for a in res["attempts"]], ["failed", "ok"])
        self.assertEqual(res["attempts"][0]["reason"], "download_not_found")
        self.assertNotIn(self.S2PDF, self.urls())                             # S2 link only if needed

    def test_degraded_ar5iv_is_kept_when_pdf_fails(self):
        res = self.go({self.AR5IV: FakeResponse(200, text=ar5iv_html(hidden_text=True)),
                        self.APDF: FakeResponse(200, content=b"<html>paywall</html>"),
                        self.S2PDF: FakeResponse(404)})
        self.assertTrue(res["has_full_text"])
        self.assertEqual((res["best"]["source"], res["best"]["status"]), ("ar5iv", "degraded"))
        self.assertEqual(self.urls().count(self.S2PDF), 1)                    # S2 link was tried too

    def test_ok_pdf_beats_degraded_ar5iv(self):
        res = self.go({self.AR5IV: FakeResponse(200, text=ar5iv_html(hidden_text=True)), self.APDF: self.pdf_ok()})
        self.assertEqual((res["best"]["source"], res["best"]["status"]), ("arxiv_pdf", "ok"))
        self.assertNotIn(self.S2PDF, self.urls())

    def test_everything_failed_gives_abstract_only_with_attempt_log(self):
        res = self.go({self.AR5IV: FakeResponse(404), self.APDF: FakeResponse(404), self.S2PDF: FakeResponse(404)})
        self.assertFalse(res["has_full_text"])
        self.assertIsNone(res["best"])
        self.assertEqual([a["source"] for a in res["attempts"]], ["ar5iv", "arxiv_pdf", "s2_pdf"])

    def test_server_error_is_logged_not_raised(self):
        res = self.go({self.AR5IV: FakeResponse(500), self.APDF: self.pdf_ok()})
        self.assertEqual(res["attempts"][0]["reason"], "download_error")
        self.assertEqual(res["best"]["source"], "arxiv_pdf")

    def test_not_a_pdf_and_too_large(self):
        f = self.fetcher({self.APDF: FakeResponse(200, content=b"<html>login</html>")})
        self.assertEqual(f.fetch_pdf(self.APDF)["status"], "not_a_pdf")
        n = len(self.session.calls)
        self.assertEqual(f.fetch_pdf(self.APDF)["status"], "not_a_pdf")        # failure remembered briefly
        self.assertEqual(len(self.session.calls), n)

        f = self.fetcher({self.APDF: FakeResponse(200, content=b"%PDF-" + b"x" * 5000)})
        f.max_pdf_bytes = 1000
        self.assertEqual(f.fetch_pdf(self.APDF)["status"], "too_large")
        f = self.fetcher({self.APDF: FakeResponse(200, content=b"%PDF-", headers={"Content-Length": "999999"})})
        f.max_pdf_bytes = 1000
        self.assertEqual(f.fetch_pdf(self.APDF)["status"], "too_large")

    def test_pdf_is_downloaded_once_then_read_from_disk(self):
        f = self.fetcher({self.APDF: self.pdf_ok()})
        a, b = f.fetch_pdf(self.APDF), f.fetch_pdf(self.APDF)
        self.assertEqual((a["sha256"], len(self.session.calls)), (b["sha256"], 1))
        self.assertTrue(Path(a["path"]).exists())

    def test_unreadable_pdf_is_a_failed_candidate(self):
        def broken(_):
            raise ValueError("encrypted")
        res = self.go({self.AR5IV: FakeResponse(404), self.APDF: self.pdf_ok(), self.S2PDF: FakeResponse(404)}, reader=broken)
        self.assertEqual(res["attempts"][1]["reason"], "pdf_unreadable")
        self.assertFalse(res["has_full_text"])

    def test_missing_pdf_library_stops_loudly(self):
        def missing(_):
            raise PdfLibraryMissing("install it")
        with self.assertRaises(PdfLibraryMissing):
            self.go({self.AR5IV: FakeResponse(404), self.APDF: self.pdf_ok()}, reader=missing)

    def test_seed_without_arxiv_id_is_skipped(self):
        res = self.go({}, paper={"key": "s2:p9", "title": "X", "arxiv_id": None})
        self.assertEqual((res["has_full_text"], res["attempts"][0]["reason"]), (False, "no_arxiv_id"))
        self.assertEqual(self.session.calls, [])

    def test_extraction_is_cached_by_page_hash(self):
        f = self.fetcher({self.AR5IV: FakeResponse(200, text=ar5iv_html())})
        get_seed_fulltext(self.paper, f, TH)
        get_seed_fulltext(self.paper, f, TH)
        self.assertEqual(len(self.session.calls), 1)                          # second run: nothing downloaded


class StepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.m = Manifest(root=self.tmp)
        seeds = {"seeds": [
            {"paper": {"key": "s2:p1", "title": TITLE, "arxiv_id": "2305.12345", "open_access_pdf_url": None}},
            {"paper": {"key": "doi:10.1/x", "title": "Journal Paper", "arxiv_id": None, "open_access_pdf_url": None}}]}
        atomic_write_json(self.tmp / "seeds.json", seeds)
        routes = {ft.AR5IV_BASE_URL + "2305.12345": FakeResponse(200, text=ar5iv_html())}
        self.session = FakeSession(lambda url, params: routes.get(url, FakeResponse(404)))
        self.fetcher = FulltextFetcher(cache=Cache(":memory:"), session=self.session, pdf_dir=self.tmp / "pdfs",
                                       min_interval=0.0, sleep=lambda s: None, now=lambda: 1000.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_step_writes_files_and_skips_on_rerun(self):
        res = run_fulltext_step(self.m, self.fetcher, TH)
        self.assertEqual((res.status, res.notes["ok"], res.notes["unavailable"]), ("done", 1, 1))
        folder = self.tmp / "fulltext"
        index = read_json(folder / "index.json")
        self.assertTrue(index["s2:p1"]["has_full_text"])
        self.assertFalse(index["doi:10.1/x"]["has_full_text"])
        self.assertIn("Attention Is All You Need", (folder / "s2_p1.txt").read_text())
        self.assertFalse((folder / "doi_10.1_x.txt").exists())
        report = read_json(folder / "s2_p1.json")
        self.assertEqual((report["status"], report["source"]), ("ok", "ar5iv"))
        self.assertEqual(report["attempts"][0]["source"], "ar5iv")
        self.assertEqual(run_fulltext_step(self.m, self.fetcher, TH).status, "skipped")

    def test_threshold_change_reruns(self):
        run_fulltext_step(self.m, self.fetcher, TH)
        stricter = Thresholds(min_chars=300, min_chars_per_page=50, min_sections=99)
        self.assertEqual(run_fulltext_step(self.m, self.fetcher, stricter).status, "done")
        report = read_json(self.tmp / "fulltext" / "s2_p1.json")
        self.assertEqual(report["status"], "degraded")                         # same page, stricter rule

    def test_needs_seeds_first(self):
        (self.tmp / "seeds.json").unlink()
        from module_B_literature_kg.b0.manifest import MissingInputError
        with self.assertRaises(MissingInputError):
            run_fulltext_step(self.m, self.fetcher, TH)


class MatchingTests(unittest.TestCase):
    def test_normalize_for_matching(self):
        a = normalize_for_matching("We use a trans-\nformer and the \ufb01rst \u201cmodel\u201d.")
        b = normalize_for_matching('we use a transformer and the first "model".')
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
