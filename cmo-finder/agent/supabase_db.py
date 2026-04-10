"""
Supabase persistence layer for CMO Finder.

All operations fall back gracefully — if Supabase is not configured the
caller gets empty data / a non-empty error string and the app continues
with local-disk persistence.

Tables required (run in Supabase SQL editor):

    CREATE TABLE manufacturers (
      company_name         TEXT PRIMARY KEY,
      city                 TEXT,
      state                TEXT,
      phone                TEXT,
      email                TEXT,
      website              TEXT,
      plant_address        TEXT,
      gst                  TEXT,
      certifications       JSONB  DEFAULT '[]',
      dosage_forms         JSONB  DEFAULT '[]',
      capacity             TEXT,
      min_order            TEXT,
      description          TEXT,
      source_url           TEXT,
      contact_person       TEXT,
      all_phones           JSONB  DEFAULT '[]',
      searched_dosage_form TEXT,
      search_key           TEXT,
      found_at             TEXT,
      grade                TEXT   DEFAULT 'ungraded',
      status               TEXT   DEFAULT 'new',
      grade_notes          TEXT,
      graded_at            TEXT,
      extra                JSONB  DEFAULT '{}'
    );

    CREATE TABLE search_state (
      search_key  TEXT PRIMARY KEY,
      seen_urls   JSONB   DEFAULT '[]',
      batch_count INTEGER DEFAULT 0
    );

    -- Disable RLS for both tables so the anon/service key can read/write:
    ALTER TABLE manufacturers DISABLE ROW LEVEL SECURITY;
    ALTER TABLE search_state  DISABLE ROW LEVEL SECURITY;
"""

from __future__ import annotations

import os

# Known top-level columns in the manufacturers table.
# Everything else is packed into the JSONB `extra` column.
_COLS = {
    "company_name", "city", "state", "phone", "email", "website",
    "plant_address", "gst", "certifications", "dosage_forms", "capacity",
    "min_order", "description", "source_url", "contact_person", "all_phones",
    "searched_dosage_form", "search_key", "found_at",
    "grade", "status", "grade_notes", "graded_at",
}

# Module-level singleton — created once per process lifetime.
_client = None
_init_done = False


def _get_client():
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True

    url = key = ""
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", "") or ""
        key = st.secrets.get("SUPABASE_KEY", "") or ""
    except Exception:
        pass
    if not url:
        url = os.environ.get("SUPABASE_URL", "")
    if not key:
        key = os.environ.get("SUPABASE_KEY", "")

    if url and key:
        try:
            from supabase import create_client
            _client = create_client(url, key)
        except Exception:
            _client = None
    return _client


def is_configured() -> bool:
    """Return True if Supabase credentials are available."""
    return _get_client() is not None


# ── Row packing / unpacking ───────────────────────────────────────────────────

def _pack(record: dict) -> dict:
    """Split a manufacturer dict into known columns + extra JSONB."""
    row: dict = {}
    extra: dict = {}
    for k, v in record.items():
        if k in _COLS:
            row[k] = v
        else:
            extra[k] = v
    row["extra"] = extra
    return row


def _unpack(row: dict) -> dict:
    """Merge extra JSONB back into the flat manufacturer dict."""
    r = {k: v for k, v in row.items() if k != "extra" and v is not None}
    r.update(row.get("extra") or {})
    return r


# ── Public API ────────────────────────────────────────────────────────────────

def load_manufacturers() -> list[dict]:
    """Load all manufacturers from Supabase. Returns [] on error."""
    c = _get_client()
    if not c:
        return []
    try:
        resp = c.table("manufacturers").select("*").execute()
        return [_unpack(row) for row in (resp.data or [])]
    except Exception:
        return []


def bulk_upsert(records: list[dict]) -> str:
    """Upsert a list of manufacturer records. Returns '' on success."""
    c = _get_client()
    if not c:
        return "Supabase not configured"
    if not records:
        return ""
    try:
        rows = [_pack(r) for r in records]
        c.table("manufacturers").upsert(rows, on_conflict="company_name").execute()
        return ""
    except Exception as e:
        return str(e)


def upsert_one(record: dict) -> str:
    """Upsert a single manufacturer. Returns '' on success."""
    c = _get_client()
    if not c:
        return "Supabase not configured"
    try:
        c.table("manufacturers").upsert(_pack(record), on_conflict="company_name").execute()
        return ""
    except Exception as e:
        return str(e)


def update_grade(company_name: str, grade: str, status: str, notes: str) -> str:
    """Update grade/status/notes for one manufacturer. Returns '' on success."""
    c = _get_client()
    if not c:
        return "Supabase not configured"
    try:
        from datetime import datetime
        c.table("manufacturers").update({
            "grade":       grade,
            "status":      status,
            "grade_notes": notes,
            "graded_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        }).eq("company_name", company_name).execute()
        return ""
    except Exception as e:
        return str(e)


def load_search_state() -> dict:
    """
    Returns {search_key: {"seen_urls": set, "batch_count": int}, ...}.
    Returns {} on error or if Supabase not configured.
    """
    c = _get_client()
    if not c:
        return {}
    try:
        resp = c.table("search_state").select("*").execute()
        result = {}
        for row in (resp.data or []):
            result[row["search_key"]] = {
                "seen_urls":   set(row.get("seen_urls") or []),
                "batch_count": row.get("batch_count", 0),
            }
        return result
    except Exception:
        return {}


def save_search_state(search_key: str, seen_urls: set | list, batch_count: int) -> str:
    """Upsert search progress for one key. Returns '' on success."""
    c = _get_client()
    if not c:
        return "Supabase not configured"
    try:
        c.table("search_state").upsert({
            "search_key":  search_key,
            "seen_urls":   list(seen_urls),
            "batch_count": batch_count,
        }, on_conflict="search_key").execute()
        return ""
    except Exception as e:
        return str(e)
