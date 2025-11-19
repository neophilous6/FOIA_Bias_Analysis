# FOIA Bias Analysis

Automated, end-to-end pipeline for building a labeled corpus of FOIA disclosures and
quantifying potential partisan bias in what gets released. The repository packages a
modular ingestion stack, preprocessing utilities, LLM-based document classifiers, and
analysis helpers that operationalize both "wrongdoing" and "favorability" hypotheses
for Democratic and Republican administrations.

## Repository layout

```
foia_bias/
  analysis/             # Aggregations + statistical models
  data_sources/         # API + scraper clients for FOIA feeds
  llm/                  # Prompt templates and classification wrappers
  processing/           # Text extraction, admin mapping, filters, dedupe
  utils/                # Shared helpers (logging, concurrency, caching)
sample_data/            # Tiny CSV fixtures used for smoke-testing agency logs
config.yaml             # Primary experiment configuration
config.json             # JSON mirror of the YAML config
main.py                 # Typer-based CLI entry point
pyproject.toml          # Project metadata + dependencies
requirements.txt        # Convenience dependency list
```

## Quick start

1. **Install dependencies** (ideally inside a virtual environment):

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm  # required for NER filter
   ```

2. **Configure credentials** via environment variables:

   ```bash
   export OPENAI_API_KEY="sk-..."
   # Optional but faster: token from https://www.muckrock.com/api/
   export MUCKROCK_API_TOKEN="mr-..."
   ```

   > The pipeline can read public MuckRock documents without a token as long as the
   > client throttles itself to ~1 request/second. Provide a token if you have one to
   > avoid the built-in rate limit and speed up ingestion.

3. **Run the CLI** to process sources and build labeled datasets:

   ```bash
   python main.py run --config config.yaml
   ```

   The CLI supports sub-commands for individual sources, for example:

   ```bash
   python main.py ingest muckrock --config config.yaml
   python main.py ingest agency-logs --config config.yaml
   python main.py analyze wrongdoing --config config.yaml
   ```

   > **Sample data for smoke tests:** The default configuration keeps the agency-log
   > sources pointed at `sample_data/agency_logs/*.csv` so every fresh clone can run
   > end-to-end without contacting placeholder domains such as `example.gov`. Update
   > the `sources.agency_logs.agencies` block (or the JSON equivalent) with real
   > URLs once you are ready to ingest full-sized FOIA logs.

4. **Review outputs** in `data/processed/` (per-source labeled parquet files) and
   inspect logs under `logs/`.

### Troubleshooting

- **`ImportError: cannot import name 'MuckRockClient'`** – this repository ships its
  own lightweight `SimpleMuckRockClient` and no longer depends on the legacy
  `python-muckrock` package. If you previously installed that package globally, pip may
  still try to import it before the local module updates. Fix it by reinstalling this
  project (`pip install -e .` or `pip install -r requirements.txt`) and, if necessary,
  uninstalling the stray dependency:

  ```bash
  pip uninstall python-muckrock
  ```

  After reinstalling, re-run `python main.py --help` to confirm the CLI imports cleanly
  with the bundled HTTP client.

- **`ERROR: Could not find a version that satisfies the requirement python-muckrock>=0.5.0`** –
  this message means pip is reading an **old requirements file** (or cached wheel) that
  still referenced the now-removed `python-muckrock` dependency. Make sure you are on the
  latest commit (pull or reclone), then clear the cached requirement by uninstalling it
  and reinstalling from the fresh repo state:

  ```bash
  pip uninstall python-muckrock  # safe even if it is not currently installed
  pip install -r requirements.txt
  ```

  If the error persists, wipe pip's cache (`pip cache purge`) or create a brand-new
  virtual environment so pip cannot reuse the stale dependency metadata.

- **Codespaces / limited disk environments** – the ingestion step caches every PDF and
  CSV it downloads (often multiple gigabytes). If you are running inside GitHub
  Codespaces or any environment with <30 GB free space, consider disabling sources you
  do not need (`sources.*.enabled`), reducing `sources.muckrock.max_requests`, or
  pointing the `download_dir` paths to a mounted volume with more space before running
  `python main.py run`. The bundled sample agency logs reside in `sample_data/` and are
  only a few kilobytes, so smoke tests stay lightweight even in constrained dev
  containers.

## Core pipeline stages

> **Temporal scope:** The default configuration only processes FOIA material through the
> end of the Biden administration (January 20, 2025). Requests or disclosures after that
> date are ignored so the analysis focuses on administrations with complete data.

1. **Ingestion**: Pull FOIA data from MuckRock, agency logs, reading rooms, and FOIA.gov
   annual stats. Each source module handles pagination, download directories, and
   metadata normalization.

2. **Preprocessing**: Convert PDFs to text with `pdfplumber` + OCR fallback, map decision
   dates to administrations, deduplicate near-identical texts, and run cheap keyword/NER
   prefilters to avoid unnecessary LLM calls.

   The partisan dictionary for that prefilter is assembled automatically at startup by
   downloading the `congress-legislators` dataset from GitHub (every Senator and
   Representative since 1789, filtered down to 1993–2025) and caching the merged JSON at
   `data/cache/congress_legislators.json`. The loader also injects the presidents and vice
   presidents from the Clinton → Biden administrations so executive-branch principals are
   always recognized. If outbound HTTP is blocked, the loader falls back to a tiny static
   list and logs a warning; override the URLs or cache location with the
   `CONGRESS_LEGISLATORS_CURRENT_URL`, `CONGRESS_LEGISLATORS_HISTORICAL_URL`, and
   `CONGRESS_LEGISLATORS_CACHE` environment variables when needed.

3. **LLM labeling**: Send political candidates to a JSON-structured classifier prompt that
   scores political relevance, wrongdoing probability, and favorability for each party. A
   separate embedding-based classifier supports a cheap "is this political" triage step.

4. **Analysis**: Build derived variables (party targets, same-party indicators, favorability
   differences) and run regression models (logit/OLS with fixed effects) to evaluate the
   cross-party exposure/disadvantage and same-party suppression/advantage hypotheses.

## Testing hypotheses

Two framing variants are encoded in the analysis helpers:

- **Wrongdoing framing**: focuses on whether documents alleging wrongdoing target the
  incumbent party or the opposition, aligned with the cross-party exposure and same-party
  suppression hypotheses.
- **Favorability framing**: evaluates continuous favorability scores to capture
  cross-party disadvantage and same-party advantage signals.

See `foia_bias/analysis/models.py` for regression formulas and options to include agency
and year fixed effects plus clustered standard errors.

## Extending the project

- Add new ingestion endpoints by implementing `BaseIngestor` subclasses under
  `foia_bias/data_sources/` and registering them in `main.py`.
- Improve the political prefilter by expanding the keyword list, adding more named
  entities, or training a better embedding classifier with human-labeled data.
- Swap in alternative LLM providers by adapting `foia_bias/llm/client.py` while keeping
  prompt + schema contracts intact.
- Export results to a relational database by enabling the storage config block in
  `config.yaml` and implementing a SQL uploader.

## Reference FOIA endpoints

| Agency | Endpoint URL | Contents |
| --- | --- | --- |
| Office of Information Policy (Department of Justice) | https://www.justice.gov/oip/available-documents-oip | FOIA logs (monthly CSV/PDF) for OIP component |
| DOJ (older logs) | https://www.justice.gov/oip/older-foia-logs | Historical FOIA logs (2012-2024) |
| Department of Health & Human Services (HHS) | https://www.hhs.gov/foia/electronic-reading-room/foia-logs/index.html | FOIA logs (2017-2024) |
| National Archives and Records Administration (NARA) | https://www.archives.gov/foia/electronic-reading-room | Frequently requested + reading-room disclosures |
| Defense Intelligence Agency (DIA) | https://www.dia.mil/FOIA/FOIA-Electronic-Reading-Room/ | Reading room + request logs |
| United States Secret Service (USSS) | https://www.secretservice.gov/foia/foia-reading-library | FOIA logs + reading library |
| Environmental Protection Agency (EPA) | https://foiapublicaccessportal.epa.gov/app/ReadingRoom.aspx | Public access reading room with released documents |
| Department of Commerce | https://www.commerce.gov/opog/foia/electronic-foia-library | FOIA library + logs for Commerce |

## License

MIT License. See `LICENSE` (add your own terms as needed).
