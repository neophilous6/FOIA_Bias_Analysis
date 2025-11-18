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
   # Choose ONE of the following authentication paths for MuckRock
   export MUCKROCK_API_TOKEN="mr-..."                 # token-based REST client
   # or provide username/password for the python-muckrock SDK
   # export MUCKROCK_USERNAME="you@example.com"
   # export MUCKROCK_PASSWORD="super-secret"
   ```

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

4. **Review outputs** in `data/processed/` (per-source labeled parquet files) and
   inspect logs under `logs/`.

## Core pipeline stages

1. **Ingestion**: Pull FOIA data from MuckRock, agency logs, reading rooms, and FOIA.gov
   annual stats. Each source module handles pagination, download directories, and
   metadata normalization.

2. **Preprocessing**: Convert PDFs to text with `pdfplumber` + OCR fallback, map decision
   dates to administrations, deduplicate near-identical texts, and run cheap keyword/NER
   prefilters to avoid unnecessary LLM calls.

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
| DOJ (older logs) | https://www.justice.gov/oip/older-foia-logs | Historical FOIA logs (2012-2023) |
| Department of Health & Human Services (HHS) | https://www.hhs.gov/foia/electronic-reading-room/foia-logs/index.html | FOIA logs (2017-2023) |
| National Archives and Records Administration (NARA) | https://www.archives.gov/foia/electronic-reading-room | Frequently requested + reading-room disclosures |
| Defense Intelligence Agency (DIA) | https://www.dia.mil/FOIA/FOIA-Electronic-Reading-Room/ | Reading room + request logs |
| United States Secret Service (USSS) | https://www.secretservice.gov/foia/foia-reading-library | FOIA logs + reading library |
| Environmental Protection Agency (EPA) | https://foiapublicaccessportal.epa.gov/app/ReadingRoom.aspx | Public access reading room with released documents |
| Department of Commerce | https://www.commerce.gov/opog/foia/electronic-foia-library | FOIA library + logs for Commerce |

## License

MIT License. See `LICENSE` (add your own terms as needed).
