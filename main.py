"""Typer-based CLI for running the FOIA bias pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from foia_bias.pipeline import Pipeline
from foia_bias.utils.config_loader import load_config

app = typer.Typer(help="FOIA partisan bias pipeline")
ingest_app = typer.Typer(help="Ingestion subcommands")
analysis_app = typer.Typer(help="Analysis subcommands")
app.add_typer(ingest_app, name="ingest")
app.add_typer(analysis_app, name="analyze")


def load_pipeline(config_path: str) -> Pipeline:
    """Load config from disk and build a ready-to-run pipeline."""
    config = load_config(config_path)
    return Pipeline(config)


@app.command()
def run(config: str = typer.Option("config.yaml", help="Path to YAML/JSON config")):
    """Run all enabled sources end-to-end."""
    pipeline = load_pipeline(config)
    pipeline.run_all()


@ingest_app.command("muckrock")
def ingest_muckrock(config: str = typer.Option("config.yaml")):
    pipeline = load_pipeline(config)
    pipeline.process_muckrock()


@ingest_app.command("agency-logs")
def ingest_agency_logs(config: str = typer.Option("config.yaml")):
    pipeline = load_pipeline(config)
    pipeline.process_agency_logs()


@ingest_app.command("reading-rooms")
def ingest_reading_rooms(config: str = typer.Option("config.yaml")):
    pipeline = load_pipeline(config)
    pipeline.process_reading_rooms()


@ingest_app.command("foia-gov")
def ingest_foia_gov(config: str = typer.Option("config.yaml")):
    pipeline = load_pipeline(config)
    pipeline.process_foia_gov_annual()


@analysis_app.command("wrongdoing")
def analyze_wrongdoing(
    config: str = typer.Option("config.yaml"),
    source: Optional[str] = typer.Option(None, help="Optional source filter (e.g., muckrock)"),
):
    pipeline = load_pipeline(config)
    typer.echo(pipeline.analyze_wrongdoing(source))


@analysis_app.command("favorability")
def analyze_favorability(
    config: str = typer.Option("config.yaml"),
    source: Optional[str] = typer.Option(None, help="Optional source filter"),
):
    pipeline = load_pipeline(config)
    typer.echo(pipeline.analyze_favorability(source))


if __name__ == "__main__":
    app()
