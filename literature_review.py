#!/usr/bin/env python3
"""Literature review pipeline.

Features:
- Read boolean keyword expressions from a .txt file (AND/OR/parentheses).
- Search multiple sources (Semantic Scholar, arXiv, ACL Anthology, optional Google Scholar via SerpAPI).
- Build a unified dataset (title, abstract, year, IDs, source list).
- Deduplicate papers across sources.
- Rank relevance with one or more LLM models using a prompt template.
- Load configuration/API keys from .env.
- Export ranked results to JSON and CSV.

Environment variables (optional):
- OPENAI_API_KEY: required for LLM ranking.
- OPENAI_BASE_URL: defaults to https://api.openai.com/v1
- SEMANTIC_SCHOLAR_API_KEY: optional, improves rate limits for Semantic Scholar.
- SERPAPI_API_KEY: required for Google Scholar collection.
- ACL_ANTHOLOGY_DATA_DIR: optional local ACL anthology data directory for library mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class Paper:
    title: str
    abstract: str
    year: Optional[int] = None
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    paper_id: str = ""
    sources: Set[str] = field(default_factory=set)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    relevance_reasons: Dict[str, str] = field(default_factory=dict)
    relevance_details: Dict[str, Dict[str, object]] = field(default_factory=dict)
    avg_relevance_score: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "url": self.url,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "paper_id": self.paper_id,
            "sources": sorted(self.sources),
            "relevance_scores": self.relevance_scores,
            "relevance_reasons": self.relevance_reasons,
            "relevance_details": self.relevance_details,
            "avg_relevance_score": self.avg_relevance_score,
        }


class BooleanQueryParser:
    """Parse expressions with AND/OR/parentheses into DNF clauses."""

    OPS = {"AND", "OR", "(", ")"}

    def __init__(self, expression: str):
        self.tokens = self._tokenize(expression)
        self.pos = 0

    @staticmethod
    def _tokenize(expression: str) -> List[str]:
        # Supports quoted phrases, AND/OR operators, and parentheses.
        raw = re.findall(r'"[^"]+"|\(|\)|\bAND\b|\bOR\b|[^\s()]+', expression, flags=re.IGNORECASE)
        tokens: List[str] = []
        for token in raw:
            upper = token.upper()
            if upper in {"AND", "OR"}:
                tokens.append(upper)
            elif token in {"(", ")"}:
                tokens.append(token)
            else:
                cleaned = token.strip().strip('"').strip()
                if cleaned:
                    tokens.append(cleaned)

        # Insert implicit AND between adjacent terms/parentheses.
        with_implicit_and: List[str] = []
        for i, token in enumerate(tokens):
            with_implicit_and.append(token)
            if i == len(tokens) - 1:
                continue
            cur_is_term = token not in BooleanQueryParser.OPS
            nxt = tokens[i + 1]
            nxt_is_term_or_open = nxt not in BooleanQueryParser.OPS or nxt == "("
            if (cur_is_term or token == ")") and nxt_is_term_or_open:
                with_implicit_and.append("AND")
        return with_implicit_and

    def parse_to_dnf(self) -> List[List[str]]:
        if not self.tokens:
            raise ValueError("Keyword expression is empty.")
        ast = self._parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token near '{self.tokens[self.pos]}'.")
        return self._to_dnf(ast)

    def _parse_or(self):
        node = self._parse_and()
        while self._peek() == "OR":
            self._consume("OR")
            rhs = self._parse_and()
            node = ("OR", node, rhs)
        return node

    def _parse_and(self):
        node = self._parse_primary()
        while self._peek() == "AND":
            self._consume("AND")
            rhs = self._parse_primary()
            node = ("AND", node, rhs)
        return node

    def _parse_primary(self):
        token = self._peek()
        if token is None:
            raise ValueError("Incomplete boolean expression.")
        if token == "(":
            self._consume("(")
            node = self._parse_or()
            self._consume(")")
            return node
        if token in {"AND", "OR", ")"}:
            raise ValueError(f"Unexpected token '{token}'.")
        self.pos += 1
        return ("TERM", token)

    def _peek(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, expected: str) -> None:
        token = self._peek()
        if token != expected:
            raise ValueError(f"Expected '{expected}' but found '{token}'.")
        self.pos += 1

    @staticmethod
    def _to_dnf(node) -> List[List[str]]:
        op = node[0]
        if op == "TERM":
            return [[node[1]]]
        if op == "OR":
            return BooleanQueryParser._to_dnf(node[1]) + BooleanQueryParser._to_dnf(node[2])
        if op == "AND":
            left = BooleanQueryParser._to_dnf(node[1])
            right = BooleanQueryParser._to_dnf(node[2])
            combined = []
            for l, r in product(left, right):
                combined.append(l + r)
            return combined
        raise ValueError(f"Unknown AST node: {op}")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_title(title: str) -> str:
    title = clean_text(title).lower()
    return re.sub(r"[^a-z0-9]+", "", title)


def parse_year(text: str) -> Optional[int]:
    match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text or "")
    return int(match.group(1)) if match else None


def load_env_file(path: str = ".env", override: bool = False) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"[warn] Failed to read {path}: {exc}")


def normalize_keyword_expression(expression: str) -> str:
    # Handle smart quotes and mixed whitespace copied from docs/editors.
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\ufeff": "",
    }
    normalized = expression
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


def safe_get(url: str, session: requests.Session, **kwargs) -> requests.Response:
    resp = session.get(url, timeout=kwargs.pop("timeout", 30), **kwargs)
    resp.raise_for_status()
    return resp


def read_keyword_clauses(keyword_file: str, max_clauses: int = 64) -> List[List[str]]:
    with open(keyword_file, "r", encoding="utf-8") as f:
        content = normalize_keyword_expression(f.read())
    upper = content.upper()
    if "AND" in upper and "OR" in upper and "(" not in content and ")" not in content:
        print(
            "[warn] keywords.txt mixes AND/OR without parentheses. "
            "Parser precedence is AND before OR; wrap OR groups in parentheses for intended logic."
        )
    parser = BooleanQueryParser(content)
    clauses = parser.parse_to_dnf()
    if len(clauses) > max_clauses:
        print(
            f"[warn] Parsed {len(clauses)} boolean clauses (threshold={max_clauses}). "
            "Keeping all clauses for correctness; this may be slower."
        )
    return clauses


def search_semantic_scholar(
    clauses: Sequence[Sequence[str]],
    max_results: int,
    session: requests.Session,
    sleep_seconds: float = 0.2,
    request_timeout: int = 30,
) -> List[Paper]:
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"x-api-key": api_key} if api_key else {}
    fields = "paperId,title,abstract,year,url,externalIds"
    out: List[Paper] = []

    for clause in clauses:
        query = " ".join(clause)
        offset = 0
        while offset < max_results:
            limit = min(100, max_results - offset)
            params = {"query": query, "limit": limit, "offset": offset, "fields": fields}
            try:
                resp = safe_get(endpoint, session=session, params=params, headers=headers, timeout=request_timeout)
            except Exception as exc:
                print(f"[warn] Semantic Scholar failed for query '{query}': {exc}")
                break
            data = resp.json().get("data", [])
            if not data:
                break

            for item in data:
                ext = item.get("externalIds") or {}
                out.append(
                    Paper(
                        title=clean_text(item.get("title", "")),
                        abstract=clean_text(item.get("abstract", "")),
                        year=item.get("year"),
                        url=item.get("url", ""),
                        doi=(ext.get("DOI") or "").strip(),
                        arxiv_id=(ext.get("ArXiv") or "").strip(),
                        paper_id=(item.get("paperId") or "").strip(),
                        sources={"semantic_scholar"},
                    )
                )

            if len(data) < limit:
                break
            offset += limit
            time.sleep(sleep_seconds)

    return out


def _build_arxiv_query(clause: Sequence[str]) -> str:
    parts = []
    for token in clause:
        token = token.replace('"', "")
        parts.append(f'all:"{token}"')
    return " AND ".join(parts)


def search_arxiv(
    clauses: Sequence[Sequence[str]],
    max_results: int,
    session: requests.Session,
    sleep_seconds: float = 0.2,
    request_timeout: int = 30,
) -> List[Paper]:
    endpoint = "https://export.arxiv.org/api/query"
    out: List[Paper] = []

    for clause in clauses:
        query = _build_arxiv_query(clause)
        start = 0
        while start < max_results:
            batch = min(100, max_results - start)
            params = {
                "search_query": query,
                "start": start,
                "max_results": batch,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            try:
                resp = safe_get(endpoint, session=session, params=params, timeout=request_timeout)
                root = ElementTree.fromstring(resp.text)
            except Exception as exc:
                print(f"[warn] arXiv failed for query '{query}': {exc}")
                break

            entries = root.findall("atom:entry", namespaces=ATOM_NS)
            if not entries:
                break

            for entry in entries:
                title = clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
                abstract = clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
                url = clean_text(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
                published = clean_text(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
                year = parse_year(published)
                doi = clean_text(entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS))

                arxiv_id = ""
                if url:
                    parts = url.rstrip("/").split("/")
                    if parts:
                        arxiv_id = parts[-1]

                out.append(
                    Paper(
                        title=title,
                        abstract=abstract,
                        year=year,
                        url=url,
                        doi=doi,
                        arxiv_id=arxiv_id,
                        paper_id=f"arxiv:{arxiv_id}" if arxiv_id else "",
                        sources={"arxiv"},
                    )
                )

            if len(entries) < batch:
                break
            start += batch
            time.sleep(sleep_seconds)

    return out


def _clause_matches_text(text: str, clause: Sequence[str]) -> bool:
    haystack = text.lower()
    for term in clause:
        normalized = term.lower().strip()
        if not normalized:
            continue
        if "*" in normalized:
            # Wildcards are prefix/suffix token wildcards, not arbitrary substring matches.
            escaped = re.escape(normalized).replace(r"\*", r"[a-z0-9_-]*")
            pattern = rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"
            if re.search(pattern, haystack) is None:
                return False
        else:
            escaped = re.escape(normalized)
            pattern = rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"
            if re.search(pattern, haystack) is None:
                return False
    return True


def _matches_any_clause(
    title: str,
    abstract: str,
    clauses: Sequence[Sequence[str]],
    match_field: str = "title",
) -> bool:
    field = (match_field or "title").strip().lower()
    if field == "title":
        haystack = (title or "").strip()
    elif field == "title_abstract":
        haystack = f"{title} {abstract}".strip()
    else:
        raise ValueError(f"Unsupported keyword match field '{match_field}'. Use title or title_abstract.")
    if not haystack:
        return False
    return any(_clause_matches_text(haystack, clause) for clause in clauses)


def _maybe_call(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return value
    return value


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_to_text(v) for v in value)
    if isinstance(value, dict):
        if "text" in value:
            return _to_text(value.get("text"))
        return " ".join(_to_text(v) for v in value.values())
    if hasattr(value, "text"):
        return _to_text(getattr(value, "text"))
    return str(value)


def _get_item_field(item: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = None
        if isinstance(item, dict):
            if name in item:
                value = item.get(name)
        else:
            value = getattr(item, name, None)
            value = _maybe_call(value)
        if value not in (None, ""):
            return value
    return None


def _iter_from_container(container: Any) -> Optional[Iterator[Any]]:
    if container is None:
        return None
    if isinstance(container, dict):
        return iter(container.values())
    if isinstance(container, (list, tuple, set)):
        return iter(container)
    if hasattr(container, "values"):
        try:
            return iter(container.values())
        except Exception:
            pass
    if isinstance(container, str):
        return None
    try:
        return iter(container)
    except Exception:
        return None


def _iter_acl_objects(anthology: Any) -> Optional[Iterator[Any]]:
    attr_candidates = ("papers", "publications", "entries", "items")
    for attr in attr_candidates:
        if not hasattr(anthology, attr):
            continue
        container = _maybe_call(getattr(anthology, attr))
        iterator = _iter_from_container(container)
        if iterator is not None:
            return iterator

    method_candidates = ("get_papers", "iter_papers", "all_papers", "papers_iter")
    for method_name in method_candidates:
        method = getattr(anthology, method_name, None)
        if not callable(method):
            continue
        try:
            container = method()
        except Exception:
            continue
        iterator = _iter_from_container(container)
        if iterator is not None:
            return iterator

    iterator = _iter_from_container(anthology)
    if iterator is not None:
        return iterator
    return None


def _paper_from_acl_item(item: Any) -> Optional[Paper]:
    title = clean_text(_to_text(_get_item_field(item, ("title", "name"))))
    if not title:
        if hasattr(item, "as_dict") and callable(getattr(item, "as_dict")):
            try:
                data = item.as_dict()
                title = clean_text(_to_text(data.get("title")))
            except Exception:
                title = ""
    if not title:
        return None

    abstract = clean_text(_to_text(_get_item_field(item, ("abstract", "summary", "description"))))
    year_raw = _get_item_field(item, ("year", "date", "published", "citation"))
    year = None
    if isinstance(year_raw, int):
        year = year_raw
    elif year_raw is not None:
        year = parse_year(_to_text(year_raw))

    paper_id = clean_text(_to_text(_get_item_field(item, ("id", "paper_id", "anthology_id"))))
    url = clean_text(_to_text(_get_item_field(item, ("url", "link", "pdf"))))
    doi = clean_text(_to_text(_get_item_field(item, ("doi",))))

    if not url and paper_id and re.fullmatch(r"[A-Za-z0-9_.-]+", paper_id):
        url = f"https://aclanthology.org/{paper_id}/"
    if not year and paper_id:
        year = parse_year(paper_id)

    return Paper(
        title=title,
        abstract=abstract,
        year=year,
        url=url,
        doi=doi,
        paper_id=paper_id,
        sources={"acl_anthology"},
    )


def search_acl_anthology_library(
    clauses: Sequence[Sequence[str]],
    max_results: int,
    acl_data_dir: str = "",
    keyword_match_field: str = "title",
) -> Optional[List[Paper]]:
    try:
        from acl_anthology import Anthology  # type: ignore
    except Exception as exc:
        print(f"[info] acl-anthology library unavailable: {exc}")
        return None

    data_dir = (acl_data_dir or os.getenv("ACL_ANTHOLOGY_DATA_DIR", "")).strip()
    attempts: List[Tuple[str, Any]] = []

    if data_dir:
        attempts.extend(
            [
                ("Anthology(data_dir=...)", lambda: Anthology(data_dir=data_dir)),
                ("Anthology(...)", lambda: Anthology(data_dir)),
            ]
        )
        from_repo = getattr(Anthology, "from_repo", None)
        if callable(from_repo):
            attempts.extend(
                [
                    ("Anthology.from_repo(data_dir=...)", lambda: from_repo(data_dir=data_dir)),
                    ("Anthology.from_repo(...)", lambda: from_repo(data_dir)),
                ]
            )
    else:
        from_repo = getattr(Anthology, "from_repo", None)
        if callable(from_repo):
            attempts.append(("Anthology.from_repo()", lambda: from_repo()))
        attempts.append(("Anthology()", lambda: Anthology()))

    anthology = None
    errors: List[str] = []
    for label, loader in attempts:
        try:
            anthology = loader()
            if anthology is not None:
                break
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    if anthology is None:
        if errors:
            print(f"[warn] Failed to initialize acl-anthology library: {errors[0]}")
        return None

    iterator = _iter_acl_objects(anthology)
    if iterator is None:
        print("[warn] acl-anthology loaded but paper iterator was not found.")
        return None

    out: List[Paper] = []
    for item in iterator:
        paper = _paper_from_acl_item(item)
        if paper is None:
            continue
        if not _matches_any_clause(
            paper.title,
            paper.abstract,
            clauses,
            match_field=keyword_match_field,
        ):
            continue
        out.append(paper)
        if len(out) >= max_results:
            break

    return out


def _search_acl_with_bs4(query: str, html: str, max_results: int) -> List[Paper]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[Paper] = []

    # ACL Anthology layout can change; keep this broad and defensive.
    candidate_links = soup.select("a[href]")
    seen_urls: Set[str] = set()
    for link in candidate_links:
        href = link.get("href") or ""
        if "aclanthology.org" not in href and not href.startswith("/"):
            continue
        if "/search" in href:
            continue

        full_url = href if href.startswith("http") else f"https://aclanthology.org{href}"
        if not re.search(r"aclanthology\.org/.+", full_url):
            continue
        if full_url in seen_urls:
            continue

        title = clean_text(link.get_text(" ", strip=True))
        if not title or len(title) < 8:
            continue

        year = parse_year(full_url)
        context_text = clean_text(link.parent.get_text(" ", strip=True) if link.parent else "")
        abstract = ""

        # Heuristic: snippet text around the link may include abstract fragments.
        if context_text and context_text != title:
            abstract = context_text.replace(title, "").strip(" -:\u2013")

        out.append(
            Paper(
                title=title,
                abstract=abstract,
                year=year,
                url=full_url,
                sources={"acl_anthology"},
            )
        )
        seen_urls.add(full_url)
        if len(out) >= max_results:
            break

    return out


def _search_acl_with_regex(html: str, max_results: int) -> List[Paper]:
    out: List[Paper] = []
    seen_urls: Set[str] = set()
    pattern = re.compile(r'<a[^>]+href="([^"]*aclanthology\.org[^"]*|/[^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE)

    for href, raw_title in pattern.findall(html):
        full_url = href if href.startswith("http") else f"https://aclanthology.org{href}"
        title = clean_text(re.sub(r"<[^>]+>", "", raw_title))
        if not title or len(title) < 8:
            continue
        if full_url in seen_urls or "/search" in full_url:
            continue

        out.append(
            Paper(
                title=title,
                abstract="",
                year=parse_year(full_url),
                url=full_url,
                sources={"acl_anthology"},
            )
        )
        seen_urls.add(full_url)
        if len(out) >= max_results:
            break

    return out


def search_acl_anthology(
    clauses: Sequence[Sequence[str]],
    max_results: int,
    session: requests.Session,
    sleep_seconds: float = 0.2,
    request_timeout: int = 30,
    backend: str = "auto",
    acl_data_dir: str = "",
    keyword_match_field: str = "title",
) -> List[Paper]:
    backend = backend.strip().lower()
    if backend not in {"auto", "library", "web"}:
        raise ValueError(f"Unsupported ACL backend '{backend}'. Use auto, library, or web.")

    if backend in {"auto", "library"}:
        library_results = search_acl_anthology_library(
            clauses=clauses,
            max_results=max_results,
            acl_data_dir=acl_data_dir,
            keyword_match_field=keyword_match_field,
        )
        if library_results is not None:
            print(f"[info] ACL Anthology (library): {len(library_results)} collected")
            return library_results[:max_results]
        if backend == "library":
            return []
        print("[info] Falling back to ACL Anthology web search parser.")

    out: List[Paper] = []

    per_clause = max(1, max_results // max(1, len(clauses)))
    for clause in clauses:
        query = " ".join(clause)
        url = f"https://aclanthology.org/search/?q={quote_plus(query)}"
        try:
            resp = safe_get(url, session=session, timeout=request_timeout)
            html = resp.text
        except Exception as exc:
            print(f"[warn] ACL Anthology failed for query '{query}': {exc}")
            continue

        if BeautifulSoup is not None:
            found = _search_acl_with_bs4(query, html, per_clause)
        else:
            found = _search_acl_with_regex(html, per_clause)

        if not found:
            print(f"[warn] ACL Anthology returned no parseable results for query '{query}'.")
        out.extend(found)
        time.sleep(sleep_seconds)

    return out[:max_results]


def search_google_scholar_serpapi(
    clauses: Sequence[Sequence[str]],
    max_results: int,
    session: requests.Session,
    sleep_seconds: float = 0.2,
    request_timeout: int = 30,
) -> List[Paper]:
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        print("[info] SERPAPI_API_KEY not set; skipping Google Scholar collection.")
        return []

    endpoint = "https://serpapi.com/search.json"
    out: List[Paper] = []
    per_clause = max(1, max_results // max(1, len(clauses)))

    for clause in clauses:
        query = " AND ".join(clause)
        start = 0
        collected = 0
        while collected < per_clause:
            num = min(20, per_clause - collected)
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": api_key,
                "start": start,
                "num": num,
                "hl": "en",
            }
            try:
                resp = safe_get(endpoint, session=session, params=params, timeout=request_timeout)
                payload = resp.json()
            except Exception as exc:
                print(f"[warn] Google Scholar (SerpAPI) failed for query '{query}': {exc}")
                break

            results = payload.get("organic_results", [])
            if not results:
                break

            for item in results:
                title = clean_text(item.get("title", ""))
                if not title:
                    continue
                snippet = clean_text(item.get("snippet", ""))
                year = None
                publication_info = item.get("publication_info", {})
                if isinstance(publication_info, dict):
                    year = parse_year(publication_info.get("summary", ""))
                out.append(
                    Paper(
                        title=title,
                        abstract=snippet,
                        year=year,
                        url=item.get("link", ""),
                        paper_id=str(item.get("result_id", "")),
                        sources={"google_scholar"},
                    )
                )
                collected += 1
                if collected >= per_clause:
                    break

            if len(results) < num:
                break
            start += num
            time.sleep(sleep_seconds)

    return out[:max_results]


def merge_papers(base: Paper, incoming: Paper) -> Paper:
    base.sources.update(incoming.sources)

    if len(incoming.abstract or "") > len(base.abstract or ""):
        base.abstract = incoming.abstract

    if not base.year and incoming.year:
        base.year = incoming.year

    if not base.url and incoming.url:
        base.url = incoming.url

    if not base.doi and incoming.doi:
        base.doi = incoming.doi

    if not base.arxiv_id and incoming.arxiv_id:
        base.arxiv_id = incoming.arxiv_id

    if not base.paper_id and incoming.paper_id:
        base.paper_id = incoming.paper_id

    return base


def paper_key(p: Paper) -> Tuple[str, str]:
    if p.doi:
        return ("doi", p.doi.strip().lower())
    if p.arxiv_id:
        return ("arxiv", p.arxiv_id.strip().lower())
    title_key = normalize_title(p.title)
    year_key = str(p.year) if p.year else ""
    return ("title_year", f"{title_key}:{year_key}")


def deduplicate_papers(papers: Iterable[Paper]) -> List[Paper]:
    by_key: Dict[Tuple[str, str], Paper] = {}
    title_fallback: Dict[str, Tuple[str, str]] = {}

    for p in papers:
        if not clean_text(p.title):
            continue

        key = paper_key(p)
        norm_title = normalize_title(p.title)

        chosen_key = key
        # If a title already exists, merge even when year differs/missing.
        if norm_title in title_fallback:
            chosen_key = title_fallback[norm_title]
        else:
            title_fallback[norm_title] = key

        if chosen_key in by_key:
            by_key[chosen_key] = merge_papers(by_key[chosen_key], p)
        else:
            by_key[chosen_key] = p

    return list(by_key.values())


def render_prompt(template: str, paper: Paper) -> str:
    fields = {
        "title": paper.title,
        "abstract": paper.abstract or "",
        "year": paper.year or "",
        "url": paper.url,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "paper_json": json.dumps(
            {
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.year,
                "url": paper.url,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "sources": sorted(paper.sources),
            },
            ensure_ascii=False,
        ),
    }
    rendered = template
    replaced = False
    for key, value in fields.items():
        placeholder = "{" + key + "}"
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, str(value))
            replaced = True

    # If template has no supported placeholders, append standardized paper context.
    if not replaced:
        rendered = (
            f"{template.strip()}\n\n"
            f"Paper metadata:\n"
            f"Title: {paper.title}\n"
            f"Abstract: {paper.abstract or 'N/A'}\n"
            f"Year: {paper.year or 'N/A'}\n"
            f"URL: {paper.url or 'N/A'}\n"
            f"DOI: {paper.doi or 'N/A'}\n"
            f"arXiv ID: {paper.arxiv_id or 'N/A'}\n"
            "Return strict JSON with at least: {\"score\": number, \"reason\": \"short rationale\"}."
        )
    return rendered


def extract_response_text(payload: Dict[str, object]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"]:
        return str(payload["output_text"])

    output = payload.get("output", [])
    chunks: List[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    txt = content.get("text", "")
                    if txt:
                        chunks.append(str(txt))
    return "\n".join(chunks).strip()


def parse_llm_score(text: str) -> Tuple[Optional[float], str, Optional[Dict[str, object]]]:
    text = text.strip()
    if not text:
        return None, "", None

    # Remove code fences if present.
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    try:
        obj = json.loads(fenced)
        if isinstance(obj, dict):
            score = obj.get("score")
            if score is None:
                score = obj.get("relevance_score")
            reason = obj.get("reason") or obj.get("rationale") or obj.get("reasoning") or ""
            if isinstance(score, (int, float, str)):
                score_val = float(score)
                return score_val, clean_text(str(reason)), obj
    except Exception:
        pass

    score_match = re.search(r"(?:relevance[_ ]?score|score)\s*[:=]\s*(-?\d+(?:\.\d+)?)", fenced, flags=re.IGNORECASE)
    if score_match is None:
        score_match = re.search(r"(-?\d+(?:\.\d+)?)", fenced)
    score = float(score_match.group(1)) if score_match else None
    return score, clean_text(fenced), None


def call_openai_responses_api(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    session: requests.Session,
    temperature: float = 0.0,
    timeout: int = 90,
) -> str:
    url = base_url.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
    }

    resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return extract_response_text(resp.json())


def rank_papers_with_llm(
    papers: List[Paper],
    prompt_template: str,
    models: Sequence[str],
    api_key: str,
    base_url: str,
    session: requests.Session,
    sleep_seconds: float = 0.1,
    request_timeout: int = 90,
) -> None:
    for idx, paper in enumerate(papers, start=1):
        for model in models:
            prompt = render_prompt(prompt_template, paper)
            try:
                raw = call_openai_responses_api(
                    prompt=prompt,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    session=session,
                    timeout=request_timeout,
                )
                score, reason, details = parse_llm_score(raw)
            except Exception as exc:
                print(f"[warn] LLM scoring failed for '{paper.title[:80]}' with {model}: {exc}")
                score, reason, details = None, "", None

            if score is not None:
                paper.relevance_scores[model] = float(score)
            if reason:
                paper.relevance_reasons[model] = reason
            if details:
                paper.relevance_details[model] = details
            time.sleep(sleep_seconds)

        if paper.relevance_scores:
            paper.avg_relevance_score = sum(paper.relevance_scores.values()) / len(paper.relevance_scores)

        if idx % 20 == 0 or idx == len(papers):
            print(f"[info] Scored {idx}/{len(papers)} papers")


def export_json(path: str, papers: Sequence[Paper]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in papers], f, indent=2, ensure_ascii=False)


def export_csv(path: str, papers: Sequence[Paper]) -> None:
    fields = [
        "title",
        "abstract",
        "year",
        "url",
        "doi",
        "arxiv_id",
        "sources",
        "avg_relevance_score",
        "relevance_scores",
        "relevance_reasons",
        "relevance_details",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in papers:
            writer.writerow(
                {
                    "title": p.title,
                    "abstract": p.abstract,
                    "year": p.year,
                    "url": p.url,
                    "doi": p.doi,
                    "arxiv_id": p.arxiv_id,
                    "sources": ",".join(sorted(p.sources)),
                    "avg_relevance_score": p.avg_relevance_score,
                    "relevance_scores": json.dumps(p.relevance_scores, ensure_ascii=False),
                    "relevance_reasons": json.dumps(p.relevance_reasons, ensure_ascii=False),
                    "relevance_details": json.dumps(p.relevance_details, ensure_ascii=False),
                }
            )


def load_papers_from_json(path: str) -> List[Paper]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of papers in {path}")

    papers: List[Paper] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        sources = item.get("sources", [])
        source_set = {str(s).strip() for s in sources} if isinstance(sources, list) else set()

        paper = Paper(
            title=clean_text(str(item.get("title", ""))),
            abstract=clean_text(str(item.get("abstract", ""))),
            year=item.get("year") if isinstance(item.get("year"), int) else parse_year(str(item.get("year", ""))),
            url=clean_text(str(item.get("url", ""))),
            doi=clean_text(str(item.get("doi", ""))),
            arxiv_id=clean_text(str(item.get("arxiv_id", ""))),
            paper_id=clean_text(str(item.get("paper_id", ""))),
            sources=source_set,
        )

        scores = item.get("relevance_scores", {})
        if isinstance(scores, dict):
            for k, v in scores.items():
                try:
                    paper.relevance_scores[str(k)] = float(v)
                except Exception:
                    continue

        reasons = item.get("relevance_reasons", {})
        if isinstance(reasons, dict):
            paper.relevance_reasons = {str(k): str(v) for k, v in reasons.items()}

        details = item.get("relevance_details", {})
        if isinstance(details, dict):
            paper.relevance_details = {str(k): v for k, v in details.items() if isinstance(v, dict)}

        avg_score = item.get("avg_relevance_score")
        if isinstance(avg_score, (int, float)):
            paper.avg_relevance_score = float(avg_score)

        if paper.title:
            papers.append(paper)

    return papers


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_optional_float(name: str) -> Optional[float]:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Literature review pipeline")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument(
        "--keywords-file",
        default=os.getenv("KEYWORDS_FILE", "keywords.txt"),
        help="Path to .txt file with boolean keyword expression",
    )
    parser.add_argument(
        "--prompt-file",
        default=os.getenv("PROMPT_FILE", "prompt.txt"),
        help="Prompt template .txt for LLM ranking",
    )
    parser.add_argument(
        "--input-json",
        default=os.getenv("INPUT_JSON", ""),
        help="Load papers from an existing JSON file instead of collecting from sources",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect and export papers (skip LLM ranking)",
    )
    parser.add_argument(
        "--sources",
        default=os.getenv("SOURCES", "acl_anthology"),
        help="Comma-separated list: semantic_scholar, arxiv, acl_anthology, google_scholar",
    )
    parser.add_argument(
        "--acl-backend",
        default=os.getenv("ACL_BACKEND", "auto").lower(),
        choices=["auto", "library", "web"],
        help="ACL source backend: auto (try library then web), library, or web",
    )
    parser.add_argument(
        "--acl-data-dir",
        default=os.getenv("ACL_ANTHOLOGY_DATA_DIR", ""),
        help="Optional local ACL anthology data directory for library mode",
    )
    parser.add_argument(
        "--keyword-match-field",
        default=os.getenv("KEYWORD_MATCH_FIELD", "title").lower(),
        choices=["title", "title_abstract"],
        help="Where to enforce keyword matching after retrieval",
    )
    parser.add_argument(
        "--max-results-per-source",
        type=int,
        default=env_int("MAX_RESULTS_PER_SOURCE", 100),
        help="Max papers per source",
    )
    parser.add_argument(
        "--max-clauses",
        type=int,
        default=env_int("MAX_CLAUSES", 64),
        help="Max boolean DNF clauses before fallback",
    )
    parser.add_argument(
        "--models",
        default=os.getenv("MODELS", "gpt-4.1-mini"),
        help="Comma-separated LLM models (OpenAI Responses API compatible)",
    )
    parser.add_argument("--skip-ranking", action="store_true", help="Skip LLM relevance ranking")
    parser.add_argument(
        "--min-score",
        type=float,
        default=env_optional_float("MIN_SCORE"),
        help="Filter final list by avg relevance score",
    )
    parser.add_argument(
        "--output-json",
        default=os.getenv("OUTPUT_JSON", "ranked_papers.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-csv",
        default=os.getenv("OUTPUT_CSV", "ranked_papers.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=env_float("SLEEP_SECONDS", 0.2),
        help="Sleep between API requests",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=env_int("REQUEST_TIMEOUT", 30),
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env-file", default=".env")
    bootstrap_args, _ = bootstrap.parse_known_args()
    load_env_file(bootstrap_args.env_file)

    args = parse_args()
    if args.collect_only:
        args.skip_ranking = True
    if args.skip_ranking and args.min_score is not None:
        print("[info] Ignoring min score filter because ranking is skipped.")
        args.min_score = None

    session = requests.Session()
    session.headers.update({"User-Agent": "literature-review-pipeline/1.0"})

    all_papers: List[Paper] = []
    if args.input_json:
        all_papers = load_papers_from_json(args.input_json)
        print(f"[info] Loaded {len(all_papers)} papers from {args.input_json}")
    else:
        clauses = read_keyword_clauses(args.keywords_file, max_clauses=args.max_clauses)
        print(f"[info] Parsed {len(clauses)} query clause(s): {clauses}")

        selected_sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}

        if "semantic_scholar" in selected_sources:
            papers = search_semantic_scholar(
                clauses=clauses,
                max_results=args.max_results_per_source,
                session=session,
                sleep_seconds=args.sleep_seconds,
                request_timeout=args.request_timeout,
            )
            all_papers.extend(papers)
            print(f"[info] Semantic Scholar: {len(papers)} collected")

        if "arxiv" in selected_sources:
            papers = search_arxiv(
                clauses=clauses,
                max_results=args.max_results_per_source,
                session=session,
                sleep_seconds=args.sleep_seconds,
                request_timeout=args.request_timeout,
            )
            all_papers.extend(papers)
            print(f"[info] arXiv: {len(papers)} collected")

        if "acl_anthology" in selected_sources:
            papers = search_acl_anthology(
                clauses=clauses,
                max_results=args.max_results_per_source,
                session=session,
                sleep_seconds=args.sleep_seconds,
                request_timeout=args.request_timeout,
                backend=args.acl_backend,
                acl_data_dir=args.acl_data_dir,
                keyword_match_field=args.keyword_match_field,
            )
            all_papers.extend(papers)
            print(f"[info] ACL Anthology: {len(papers)} collected")

        if "google_scholar" in selected_sources:
            papers = search_google_scholar_serpapi(
                clauses=clauses,
                max_results=args.max_results_per_source,
                session=session,
                sleep_seconds=args.sleep_seconds,
                request_timeout=args.request_timeout,
            )
            all_papers.extend(papers)
            print(f"[info] Google Scholar (SerpAPI): {len(papers)} collected")

        pre_filter_count = len(all_papers)
        all_papers = [
            p
            for p in all_papers
            if _matches_any_clause(
                p.title,
                p.abstract,
                clauses,
                match_field=args.keyword_match_field,
            )
        ]
        print(
            f"[info] Keyword enforcement ({args.keyword_match_field}): "
            f"{len(all_papers)}/{pre_filter_count} kept"
        )

    print(f"[info] Total raw papers: {len(all_papers)}")

    unique = deduplicate_papers(all_papers)
    print(f"[info] Deduplicated papers: {len(unique)}")

    if not args.skip_ranking:
        if not args.prompt_file:
            raise ValueError("--prompt-file is required unless --skip-ranking is used.")
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read().strip()
        if not prompt_template:
            raise ValueError("Prompt file is empty.")

        models = [m.strip() for m in args.models.split(",") if m.strip()]
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for ranking.")

        rank_papers_with_llm(
            papers=unique,
            prompt_template=prompt_template,
            models=models,
            api_key=api_key,
            base_url=base_url,
            session=session,
            sleep_seconds=max(0.05, args.sleep_seconds),
            request_timeout=max(30, args.request_timeout),
        )

        unique.sort(key=lambda p: p.avg_relevance_score if p.avg_relevance_score is not None else float("-inf"), reverse=True)

    if args.min_score is not None:
        unique = [p for p in unique if p.avg_relevance_score is not None and p.avg_relevance_score >= args.min_score]
        print(f"[info] Filtered by min score ({args.min_score}): {len(unique)} papers")

    export_json(args.output_json, unique)
    export_csv(args.output_csv, unique)
    print(f"[done] Wrote {len(unique)} papers to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
