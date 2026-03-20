"""
Persistent storage for CMO Finder results and search state.

Saves a single JSON file so data survives across page refreshes,
browser tab closes, and app restarts (within the same container).

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

# Stored next to app.py → cmo-finder/data/state.json
DATA_DIR   = Path(__file__).parent.parent / "data"
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
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            base = _empty()
            base.update(raw)            # fill in any missing keys (backwards compat)
            return base
    except Exception:
        pass
    return _empty()


def save(state: dict) -> None:
    """Write state to disk.  Silently swallows errors so the UI never crashes."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        state["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def clear() -> None:
    """Delete the saved state file (used by 'Clear All')."""
    try:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    except Exception:
        pass


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
