"""Shared helpers for the pipeline steps."""
from __future__ import annotations

from pathlib import Path


def analysis_dir(song, analysis=None) -> Path:
    """Where a song's analysis lives (stems, transcript, beats, aligned lyrics, prompts): --analysis if given, else
    <song>-analysis next to the song."""
    song = Path(song)
    return Path(analysis) if analysis else song.parent / f"{song.stem}-analysis"
