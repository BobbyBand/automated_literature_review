# Literature Review Pipeline

Python pipeline for academic literature review workflows:

- query scholarly sources from boolean keyword expressions
- collect and deduplicate candidate papers
- filter and rank papers with LLM prompts
- export scored results to JSON and CSV

## Privacy-first publication notes

This public repository excludes local secrets, virtual environments, cached data, and generated result exports.

Before publishing:

1. keep real credentials only in `.env`
2. commit `.env.example`, not `.env`
3. inspect prompts, keyword files, and sample data for confidential project details
4. avoid committing raw paper exports unless you intentionally want them public

## Repository layout

```text
.
├── literature_review.py
├── keywords.txt
├── prompt.txt
├── .env.example
├── requirements.txt
├── outputs/                # ignored generated files
└── acl_anthology_data/     # ignored local data directory
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with local credentials and paths.

## Example run

```bash
python literature_review.py --env-file .env
```

Generated outputs should go to `./outputs/` and remain untracked.

## Recommended public release contents

- source code
- prompt template if it is safe to disclose
- keyword template if it is safe to disclose
- dependency file
- usage documentation

Do not publish:

- `.env`
- `.venv/`
- local caches
- raw exports with sensitive or unnecessary metadata
- machine-specific paths or personal notes
