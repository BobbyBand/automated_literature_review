# Literature Review Pipeline

Python pipeline for systematic literature review workflows over scholarly sources. It can:

- parse boolean keyword queries from a text file
- collect papers from Semantic Scholar, arXiv, ACL Anthology, and optional Google Scholar via SerpAPI
- deduplicate records across sources
- rank relevance with one or more OpenAI-compatible models
- export results to JSON and CSV

The current prompt and keyword templates are geared toward finding explainability and attribution work for LLM-based multi-document text generation, but the pipeline itself is general enough to reuse for other review topics.

## Features

- Boolean query parsing with `AND`, `OR`, parentheses, and quoted phrases
- Multi-source retrieval with a unified paper schema
- Post-retrieval keyword enforcement on title or title+abstract
- Optional local ACL Anthology data directory for faster lookup
- LLM-based scoring with per-model reasoning and averaged relevance scores
- Re-ranking from a previously exported JSON file

## Repository Contents

```text
.
├── literature_review.py
├── keywords.txt
├── prompt.txt
├── .env.example
├── requirements.txt
├── outputs/                  # recommended local output directory, ignored by git
└── acl_anthology_data/       # optional local ACL data mirror, ignored by git
```

## Requirements

- Python 3.10+
- `requests`
- `beautifulsoup4` for HTML parsing in web-backed flows
- Optional API access depending on sources and ranking mode:
  - OpenAI-compatible API for LLM ranking
  - Semantic Scholar API key for higher rate limits
  - SerpAPI key for Google Scholar collection

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in only the settings you need:

```bash
cp .env.example .env
```

Important variables:

- `OPENAI_API_KEY`: required for LLM ranking
- `OPENAI_BASE_URL`: defaults to `https://api.openai.com/v1`
- `SEMANTIC_SCHOLAR_API_KEY`: optional
- `SERPAPI_API_KEY`: required for Google Scholar collection
- `ACL_ANTHOLOGY_DATA_DIR`: optional local ACL data directory
- `KEYWORDS_FILE`: boolean query file
- `PROMPT_FILE`: prompt template used for scoring
- `INPUT_JSON`: optional previously exported JSON file for re-ranking
- `SOURCES`: comma-separated list of `semantic_scholar`, `arxiv`, `acl_anthology`, `google_scholar`
- `MODELS`: comma-separated OpenAI-compatible model names
- `OUTPUT_JSON`, `OUTPUT_CSV`: output paths

The public repo should include [`.env.example`](/Users/amir/Desktop/Literature review/.env.example), not `.env`.

## Query File

[`keywords.txt`](/Users/amir/Desktop/Literature review/keywords.txt) contains a boolean expression. Example:

```text
("attribute*" OR "XAI" OR "explain*" OR "cite*" OR "highlight*" OR "quote*")
AND
("retrieval" OR "augment*" OR "context" OR "RAG")
```

Use parentheses when mixing `AND` and `OR`. The parser treats `AND` with higher precedence if grouping is omitted.

## Prompt File

[`prompt.txt`](/Users/amir/Desktop/Literature review/prompt.txt) is a template used to score each paper. The script injects:

- `{title}`
- `{abstract}`

The model is expected to return JSON with a relevance score and structured reasoning.

## Usage

Show CLI options:

```bash
python literature_review.py --help
```

Run the full pipeline:

```bash
python literature_review.py --env-file .env
```

Collect papers without LLM ranking:

```bash
python literature_review.py \
  --env-file .env \
  --collect-only \
  --sources semantic_scholar,arxiv,acl_anthology \
  --output-json outputs/candidates.json \
  --output-csv outputs/candidates.csv
```

Re-rank from an existing JSON export:

```bash
python literature_review.py \
  --env-file .env \
  --input-json outputs/candidates.json \
  --models gpt-4.1-mini
```

Use multiple models and keep only high-scoring papers:

```bash
python literature_review.py \
  --env-file .env \
  --models gpt-4.1-mini,gpt-4.1 \
  --min-score 4
```

Use ACL Anthology only with a local data directory:

```bash
python literature_review.py \
  --env-file .env \
  --sources acl_anthology \
  --acl-backend library \
  --acl-data-dir ./acl_anthology_data
```

## Main CLI Options

- `--env-file`: path to `.env`
- `--keywords-file`: boolean query file
- `--prompt-file`: scoring prompt template
- `--input-json`: load an existing JSON file instead of collecting from APIs
- `--collect-only`: collect and export without ranking
- `--sources`: retrieval sources
- `--acl-backend`: `auto`, `library`, or `web`
- `--acl-data-dir`: local ACL data directory
- `--keyword-match-field`: `title` or `title_abstract`
- `--max-results-per-source`: retrieval cap per source
- `--models`: comma-separated model list
- `--skip-ranking`: skip LLM scoring
- `--min-score`: filter by averaged relevance score
- `--output-json`, `--output-csv`: output paths

## Output Format

The JSON export is a list of paper objects with fields such as:

- `title`
- `abstract`
- `year`
- `url`
- `doi`
- `arxiv_id`
- `paper_id`
- `sources`
- `relevance_scores`
- `relevance_reasons`
- `relevance_details`
- `avg_relevance_score`

The CSV export includes:

- bibliographic fields
- source list
- average relevance score
- JSON-serialized scoring details

## Typical Workflow

1. Define your search scope in [`keywords.txt`](/Users/amir/Desktop/Literature review/keywords.txt).
2. Write or revise the screening rubric in [`prompt.txt`](/Users/amir/Desktop/Literature review/prompt.txt).
3. Run collection only to inspect the candidate pool.
4. Re-run with ranking enabled.
5. Manually review high-scoring papers and borderline cases.

## Privacy and Publication Safety

This repository is intended to be publishable without leaking local secrets or machine-specific state. Before pushing to GitHub:

- never commit `.env`
- never commit `.venv/`
- avoid committing raw result exports unless you intentionally want them public
- inspect prompt and keyword files for unpublished project details
- keep output files under an ignored directory such as `outputs/`

The current ignore policy is defined in [`.gitignore`](/Users/amir/Desktop/Literature review/.gitignore).

## Notes

- Google Scholar retrieval depends on SerpAPI rather than direct scraping.
- If ranking is skipped, `--min-score` is ignored.
- If `INPUT_JSON` is provided, collection is skipped and the pipeline starts from the saved records.
- The script loads `.env` itself, so `python-dotenv` is not required.
