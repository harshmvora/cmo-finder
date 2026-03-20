"""
Persistent storage for CMO Finder results and search state.

Saves to ~/.cmo_finder/state.json — the home directory is writable on
Streamlit Cloud (appuser home) and locally (Windows/Linux/macOS).

Falls back to /tmp/cmo_finder/ if the home directory is not writable.

Data structure on disk:
{
  "results":      [ ...manufacturer dicts... ],
  "seen_by_key":  { "Tablets": ["url1","url2",...], "Tablets|metformin": [...] },
  "batch_by_key": { "Tablets": 7, "Tablets|metformin": 2 },
  "last_saved":   "2026-03-20 14:35"
}
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _find_writable_dir() -> Path:
    """Return the first directory we can actually write to."""
    candidates = [
        Path.home() / ".cmo_finder",   # ~/.cmo_finder  — persists across reruns
        Path("/tmp") / "cmo_finder",    # /tmp/cmo_finder — always writable, ephemeral
    ]
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return d
        except Exception:
            continue
    return candidates[-1]  # best effort


DATA_DIR   = _find_writable_dir()
STATE_FILE = DATA_DIR / "state.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty() -> dict:
    return {
        "results":      [],
        "seen_by_key":  {},   # search_key -> list[str] of URLs already tried
        "batch_by_key": {},   # search_key -> int  (which batch to start from next)
        "last_saved":   "",
    }


def load() -> dict:
    """Read persisted state from disk.  Returns empty state on any error."""
    try:
        if STATE_FILE.exists():
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            base = _empty()
            base.update(raw)            # fill in any missing keys (backwards compat)
            return base
    except Exception:
        pass
    return _empty()


def save(state: dict) -> str:
    """
    Write state to disk.
    Returns "" on success, or an error message string on failure.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        state["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ""           # success
    except Exception as e:
        return str(e)       # caller can surface this in the UI


def clear() -> None:
    """Delete the saved state file (used by 'Clear All')."""
    try:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    except Exception:
        pass


def data_dir_path() -> str:
    """Return the path being used for storage (for debug display)."""
    return str(STATE_FILE)


# ── Search-key helpers ─────────────────────────────────────────────────────────

def make_key(dosage_forms: list[str], product_name: str) -> str:
    """
    Stable string key that identifies a unique search context.
    Examples:
      dosage_forms=["Tablets"]            → "Tablets"
      dosage_forms=["Tablets","Capsules"] → "Capsules|Tablets"
      product_name="Metformin 500mg"      → "Tablets :: metformin 500mg"
    """
    forms = "|".join(sorted(dosage_forms))
    if product_name and product_name.strip():
        return f"{forms} :: {product_name.strip().lower()}"
    return forms


def key_label(search_key: str) -> str:
    """Human-readable version of make_key output for display."""
    if " :: " in search_key:
        forms_part, product = search_key.split(" :: ", 1)
        return f"{forms_part.replace('|', ', ')} — {product.title()}"
    return search_key.replace("|", ", ")
