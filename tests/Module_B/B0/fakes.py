"""Fake internet for the B0 tests: no network is ever used."""
import json
from urllib.parse import unquote

from module_A_tooling.config import S2_BASE_URL


class FakeResponse:
    def __init__(self, status=200, data=None, text=None, headers=None, content=None):
        self.status_code = status
        self._data = data
        self.content = content
        if text is not None:
            self.text = text
        elif content is not None:
            self.text = ""
        else:
            self.text = json.dumps(data if data is not None else {})
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size=1024):
        body = self.content or b""
        for i in range(0, len(body), chunk_size):
            yield body[i:i + chunk_size]

    def close(self):
        self.closed = True

    def json(self):
        return self._data if self._data is not None else json.loads(self.text)


class FakeSession:
    """session.get(...) answers come from `router(path, params)`; every call is recorded."""

    def __init__(self, router):
        self.router = router
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None, stream=False):
        self.calls.append((url, dict(params or {})))
        path = unquote(url[len(S2_BASE_URL):]) if url.startswith(S2_BASE_URL) else url
        return self.router(path, dict(params or {}))


def s2_raw(pid, title, abstract="An abstract.", arxiv=None, doi=None, year=2023):
    ext = {}
    if arxiv:
        ext["ArXiv"] = arxiv
    if doi:
        ext["DOI"] = doi
    return {"paperId": pid, "title": title, "abstract": abstract, "year": year, "venue": "V",
            "citationCount": 5, "externalIds": ext, "openAccessPdf": None,
            "authors": [{"name": "Ada Lovelace"}]}


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/{aid}v2</id>
    <published>2023-05-19T17:59:59Z</published>
    <title>  A Title
      On Two Lines </title>
    <summary>  The abstract
      text.  </summary>
    <author><name>Alan Turing</name></author>
    <arxiv:doi>10.1000/XYZ</arxiv:doi>
  </entry>
</feed>"""

ARXIV_ERROR_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/api/errors#incorrect_id_format</id><title>Error</title><summary>bad</summary></entry>
</feed>"""

ARXIV_EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
