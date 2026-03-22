"""
CMO / CDMO Finder — India
AI-powered dashboard to discover Indian contract pharma manufacturers.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from ddgs import DDGS

from agent.extractor import extract_from_rich, extract_from_snippet, get_last_error
from agent.scraper import scrape_rich
from agent.searcher import DOSAGE_FORM_KEYWORDS, HUB_GROUPS, search_cmos, search_company_contacts
from agent import persistence

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CMO Finder India",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Modern CSS with Inter font ─────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
      rel="stylesheet">
<style>
  html, body, [class*="css"], .stMarkdown, .stText, .stButton>button,
  input, select, textarea, label, p, h1, h2, h3, h4, h5, h6,
  .stTabs [data-baseweb="tab"], [data-testid="stMetricLabel"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  }

  /* ── App header ── */
  .app-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 55%, #0f766e 100%);
    color: white; padding: 28px 36px; border-radius: 16px; margin-bottom: 20px;
  }
  .app-header h1 { margin: 0; font-size: 27px; font-weight: 700; letter-spacing: -0.6px; }
  .app-header p  { margin: 5px 0 0; opacity: 0.75; font-size: 14px; }
  .stat-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.18); border-radius: 20px;
    padding: 4px 14px; font-size: 13px; font-weight: 500; margin-top: 10px; margin-right: 8px;
  }

  /* ── Pills ── */
  .pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin: 2px 3px 2px 0; letter-spacing: 0.1px;
  }
  .pill-green  { background: #dcfce7; color: #15803d; }
  .pill-blue   { background: #dbeafe; color: #1d4ed8; }
  .pill-gray   { background: #f1f5f9; color: #475569; }

  /* ── Section labels ── */
  .sect {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; color: #94a3b8; margin: 14px 0 4px;
  }

  /* ── Background status card ── */
  .bg-on {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #86efac; border-radius: 12px; padding: 14px 18px; margin-top: 12px;
  }
  .bg-off {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 14px 18px; margin-top: 12px;
  }

  /* ── Misc ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stProgress > div > div {
    background: linear-gradient(90deg, #312e81, #0f766e) !important;
  }
  .stTabs [data-baseweb="tab"]                   { font-weight: 500; font-size: 14px; }
  .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #312e81; font-weight: 600; }
  .stButton > button                              { border-radius: 8px; font-weight: 500; }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #312e81, #0f766e) !important;
    border: none !important; color: white !important;
  }
  .stExpander [data-testid="stExpanderToggleIcon"] { color: #64748b; }
</style>
""", unsafe_allow_html=True)


# ── Config helpers ─────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / ".cmo_config.json"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    except Exception:
        return {}


def _save_config(data: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(data))
    except Exception:
        pass


def _default_api_key() -> str:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return _load_config().get("api_key", "")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED IN-MEMORY DATABASE  (cache_resource)
# Survives page refreshes / new tabs. Resets only on container restart.
# Background thread writes to this same dict safely (it's a mutable object).
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _db() -> dict:
    disk = persistence.load()
    return {
        "results":      disk.get("results", []),
        "seen_by_key":  {k: set(v) for k, v in disk.get("seen_by_key", {}).items()},
        "batch_by_key": dict(disk.get("batch_by_key", {})),
        "last_saved":   disk.get("last_saved", ""),
        "save_error":   "",
    }


@st.cache_resource
def _bg() -> dict:
    """Background worker state — shared across all browser sessions."""
    return {
        "running": False,
        "thread":  None,
        "params":  {},
        "log":     [],   # list[str] — most-recent first, capped at 20
    }


def _persist() -> None:
    """Write db to disk. Updates db['save_error'] and db['last_saved']."""
    db = _db()
    err = persistence.save({
        "results":      db["results"],
        "seen_by_key":  {k: list(v) for k, v in db["seen_by_key"].items()},
        "batch_by_key": db["batch_by_key"],
    })
    db["save_error"] = err
    if not err:
        db["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M")


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND SEARCH WORKER
# Runs in a daemon thread — searches continuously without a browser session.
# Writes results directly into the shared db dict.
# ══════════════════════════════════════════════════════════════════════════════

def _bg_log(msg: str) -> None:
    bg = _bg()
    bg["log"] = ([msg] + bg["log"])[:20]


def _background_worker(params: dict) -> None:
    """Runs indefinitely until bg['running'] is set to False."""
    db = _db()
    bg = _bg()

    try:
        client = anthropic.Anthropic(api_key=params["api_key"])
    except Exception as e:
        _bg_log(f"❌ Client error: {e}")
        bg["running"] = False
        return

    skey  = params["search_key"]
    batch = db["batch_by_key"].get(skey, 0)

    while bg["running"]:
        try:
            seen = set(db["seen_by_key"].get(skey, set()))
            hub_n = (batch % len(HUB_GROUPS)) + 1
            _bg_log(f"🔎 Batch #{batch + 1} · hub {hub_n}/{len(HUB_GROUPS)}")

            hits = search_cmos(
                params["dosage_forms"],
                params.get("product_name", ""),
                params.get("requirements", ""),
                params.get("max_results", 20),
                batch=batch,
                already_seen=seen,
            )

            db["seen_by_key"][skey]  = seen | {h["url"] for h in hits}
            db["batch_by_key"][skey] = batch + 1

            if not hits:
                _bg_log(f"ℹ️  Batch #{batch + 1} — no new URLs, rotating hub…")
                batch += 1
                time.sleep(5)
                continue

            _bg_log(f"📄 Batch #{batch + 1} — {len(hits)} URLs, extracting…")
            existing = {r.get("company_name", "").strip().lower() for r in db["results"]}
            added = 0

            for hit in hits:
                if not bg["running"]:
                    break
                dosage_ctx = hit["dosage_form"] + (
                    f" — {params['product_name']}" if params.get("product_name") else ""
                )
                extracted = None

                scraped = scrape_rich(hit["url"], follow_contact=True)
                if scraped:
                    extracted, _ = extract_from_rich(scraped, hit["url"], dosage_ctx, client)
                if not extracted and hit.get("snippet"):
                    extracted, _ = extract_from_snippet(
                        hit["title"], hit["snippet"], hit["url"], dosage_ctx, client
                    )
                if extracted and extracted.get("company_name"):
                    key = extracted["company_name"].strip().lower()
                    if key not in existing:
                        extracted.update({
                            "searched_dosage_form": hit["dosage_form"],
                            "search_key":           skey,
                            "found_at":             datetime.now().strftime("%Y-%m-%d %H:%M"),
                        })
                        db["results"].append(extracted)
                        existing.add(key)
                        added += 1

            if added:
                _bg_log(
                    f"✅ Batch #{batch + 1} — added {added} "
                    f"manufacturer{'s' if added != 1 else ''} "
                    f"({len(db['results'])} total)"
                )

            # Save after every batch
            persistence.save({
                "results":      db["results"],
                "seen_by_key":  {k: list(v) for k, v in db["seen_by_key"].items()},
                "batch_by_key": db["batch_by_key"],
            })
            db["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            batch += 1
            time.sleep(3)

        except Exception as e:
            _bg_log(f"⚠️  Error in batch #{batch + 1}: {e}")
            time.sleep(30)

    _bg_log("⏹️  Background search stopped.")


def _start_bg(params: dict) -> None:
    bg = _bg()
    if bg["running"] and bg["thread"] and bg["thread"].is_alive():
        return  # already running
    bg["params"]  = params
    bg["running"] = True
    t = threading.Thread(target=_background_worker, args=(params,), daemon=True)
    t.start()
    bg["thread"] = t


def _stop_bg() -> None:
    _bg()["running"] = False


# ── Session state (UI only — not database) ────────────────────────────────────
for _k, _v in [
    ("pending_lookup", None),
    ("search_batch",   0),
    ("seen_urls",      set()),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

db = _db()
bg = _bg()

# ── Sidebar ────────────────────────────────────────────────────────────────────
_config = _load_config()

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_key = st.text_input(
        "Claude API Key",
        type="password",
        placeholder="sk-ant-...",
        value=_default_api_key(),
        help="Get yours at console.anthropic.com",
    )
    if api_key and api_key != _config.get("api_key", ""):
        _save_config({**_config, "api_key": api_key})

    st.divider()
    max_results = st.slider("URLs per batch", 5, 50, 20, 5)
    deep_scrape = st.checkbox("Deep scrape (follow contact pages)", value=True)

    st.divider()
    if db["last_saved"]:
        st.caption(f"💾 Saved {db['last_saved']}")
    if db["save_error"]:
        st.warning(f"Save failed: {db['save_error']}")

    if db["results"]:
        st.divider()
        if st.button("🗑️ Clear Database", use_container_width=True):
            db.update({
                "results": [], "seen_by_key": {}, "batch_by_key": {},
                "last_saved": "", "save_error": "",
            })
            st.session_state.seen_urls      = set()
            st.session_state.pending_lookup = None
            st.session_state.search_batch   = 0
            _stop_bg()
            persistence.clear()
            st.rerun()

    st.markdown("---")
    st.caption("Powered by Claude Haiku · DuckDuckGo")

# ── Header ─────────────────────────────────────────────────────────────────────
_is_bg_alive = bg["running"] and bg["thread"] and bg["thread"].is_alive()
_n = len(db["results"])

_badges = ""
if _n:
    _badges += f'<span class="stat-badge">🗄️ {_n} manufacturers</span>'
if _is_bg_alive:
    _badges += '<span class="stat-badge">🔄 Background search active</span>'

st.markdown(f"""
<div class="app-header">
  <h1>🏭 CMO &amp; CDMO Finder — India</h1>
  <p>AI-powered discovery of Indian contract &amp; third-party manufacturers</p>
  {'<div style="margin-top:10px">' + _badges + '</div>' if _badges else ''}
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_search, tab_db = st.tabs(["🔍 Search", "🗄️ Database"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:

    if not api_key:
        st.info("👈 Enter your Claude API key in the sidebar to get started.")

    # ── Search form ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        dosage_forms = st.multiselect(
            "💊 Dosage Form(s)",
            list(DOSAGE_FORM_KEYWORDS.keys()),
            default=["Nutraceuticals / Supplements"],
        )
    with c2:
        product_name = st.text_input(
            "🧪 Product Name *(optional)*",
            placeholder="e.g. Whey Protein, Metformin 500mg…",
        )
    with c3:
        requirements = st.text_area(
            "📋 Other Requirements *(optional)*",
            placeholder="e.g. WHO-GMP certified, min 10M units/month…",
            height=70,
        )

    # Resume hint
    if dosage_forms:
        _skey  = persistence.make_key(dosage_forms, product_name)
        _done  = db["batch_by_key"].get(_skey, 0)
        _tried = len(db["seen_by_key"].get(_skey, set()))
        if _done:
            st.caption(
                f"▶  Resuming — {_done} batch{'es' if _done != 1 else ''} done "
                f"· {_tried} URLs tried · continuing from batch #{_done + 1}"
            )

    # ── Action buttons ─────────────────────────────────────────────────────────
    btn1, btn2, _ = st.columns([2, 2, 2])

    with btn1:
        search_clicked = st.button(
            "🔍 Search Once",
            use_container_width=True,
            type="primary",
            disabled=not api_key or not dosage_forms,
            help="Run one batch and show results immediately",
        )

    with btn2:
        if _is_bg_alive:
            if st.button("⏹️ Stop Background", use_container_width=True):
                _stop_bg()
                st.rerun()
        else:
            bg_start = st.button(
                "🔄 Start Background Search",
                use_container_width=True,
                disabled=not api_key or not dosage_forms,
                help="Searches continuously — keeps going even when you close this tab",
            )
            if bg_start:
                _skey = persistence.make_key(dosage_forms, product_name)
                _start_bg({
                    "api_key":      api_key,
                    "dosage_forms": dosage_forms,
                    "product_name": product_name,
                    "requirements": requirements,
                    "max_results":  max_results,
                    "search_key":   _skey,
                })
                st.rerun()

    # ── Background status (auto-refreshes every 10 s) ─────────────────────────
    @st.fragment(run_every=10)
    def _bg_panel() -> None:
        bg  = _bg()
        db  = _db()
        alive = bg["running"] and bg["thread"] and bg["thread"].is_alive()
        if not alive and not bg["log"]:
            return
        css = "bg-on" if alive else "bg-off"
        icon = "🔄" if alive else "⏹️"
        status = "running" if alive else "paused"
        st.markdown(
            f'<div class="{css}">'
            f'<strong>{icon} Background search {status}</strong>'
            f' — <span style="color:#475569">{len(db["results"])} manufacturers found</span>',
            unsafe_allow_html=True,
        )
        for msg in bg["log"][:6]:
            st.caption(msg)
        st.markdown("</div>", unsafe_allow_html=True)

    _bg_panel()

    # ── One-shot search execution ──────────────────────────────────────────────
    if search_clicked and api_key and dosage_forms:
        client = anthropic.Anthropic(api_key=api_key)
        skey   = persistence.make_key(dosage_forms, product_name)
        batch  = db["batch_by_key"].get(skey, 0)
        seen   = set(db["seen_by_key"].get(skey, set()))

        prog = st.progress(0)
        stat = st.empty()
        stat.markdown(f"🔎 **Batch #{batch + 1}** — searching…")
        prog.progress(8)

        try:
            hits = search_cmos(
                dosage_forms, product_name, requirements, max_results,
                batch=batch, already_seen=seen,
            )
        except Exception as e:
            st.error(f"Search error: {e}")
            st.stop()

        db["seen_by_key"][skey]  = seen | {h["url"] for h in hits}
        db["batch_by_key"][skey] = batch + 1

        if not hits:
            _persist()
            stat.info("No new URLs found this batch — click again to try the next hub.")
            st.stop()

        stat.markdown(f"📄 **{len(hits)} URLs** found — extracting with Claude…")
        prog.progress(18)

        new_results: list[dict] = []
        stat_scraped = stat_ok = stat_not_tpm = stat_err = stat_fail = 0
        total = len(hits)

        for i, hit in enumerate(hits):
            prog.progress(18 + int((i / total) * 78))
            stat.markdown(f"🤖 **{i+1}/{total}** — *{hit['title'][:65]}*")

            dosage_ctx = hit["dosage_form"] + (f" — {product_name}" if product_name else "")
            extracted  = None
            reason     = ""

            if deep_scrape:
                scraped = scrape_rich(hit["url"], follow_contact=True)
                if scraped:
                    stat_scraped += 1
                    extracted, reason = extract_from_rich(scraped, hit["url"], dosage_ctx, client)
                else:
                    stat_fail += 1

            if not extracted and hit.get("snippet"):
                extracted, reason = extract_from_snippet(
                    hit["title"], hit["snippet"], hit["url"], dosage_ctx, client
                )

            if reason == "not_tpm":
                stat_not_tpm += 1
            elif reason == "api_error":
                stat_err += 1
            elif extracted and extracted.get("company_name"):
                stat_ok += 1
                extracted.update({
                    "searched_dosage_form": hit["dosage_form"],
                    "search_key":           skey,
                    "found_at":             datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                new_results.append(extracted)

        prog.progress(100)

        api_err_msg = get_last_error()
        if api_err_msg:
            st.error(f"Claude API error: **{api_err_msg}**\n\nCheck your API key in the sidebar.")
        else:
            stat.markdown(
                f"✅ **Done** — {total} tried · {stat_ok} passed · "
                f"{stat_not_tpm} not TPM"
                + (f" · {stat_err} errors" if stat_err else "")
            )

        existing = {r.get("company_name", "").strip().lower() for r in db["results"]}
        added = 0
        for r in new_results:
            k = r.get("company_name", "").strip().lower()
            if k and k not in existing:
                db["results"].append(r)
                existing.add(k)
                added += 1

        _persist()

        if added:
            st.success(
                f"✅ Added **{added}** new manufacturer(s) — "
                f"**{len(db['results'])} total**. Open the **Database** tab to view all."
            )
        else:
            st.info("No new unique manufacturers found this batch.")

    # ── Recent finds preview ───────────────────────────────────────────────────
    if db["results"]:
        st.divider()
        st.markdown(f"**Recent finds** *(last 5 of {len(db['results'])} total)*")
        for r in db["results"][-5:][::-1]:
            loc = ", ".join(filter(None, [r.get("city", ""), r.get("state", "")])) or "India"
            ph  = f" · 📱 {r['phone']}" if r.get("phone") else ""
            st.markdown(f"• **{r.get('company_name', '?')}** — {loc}{ph}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_db:

    # ── Contact lookup (triggered from card buttons) ──────────────────────────
    if st.session_state.pending_lookup is not None and api_key:
        idx = st.session_state.pending_lookup
        st.session_state.pending_lookup = None
        r = db["results"][idx]
        with st.spinner(f"Finding contacts for **{r.get('company_name', '')}**…"):
            try:
                with DDGS() as ddgs:
                    hits = search_company_contacts(r.get("company_name", ""), ddgs)
                for hit in hits:
                    sc = scrape_rich(hit["url"], follow_contact=True)
                    if not sc:
                        continue
                    if sc.get("phones") and not r.get("phone"):
                        r["phone"] = sc["phones"][0]
                        r["all_phones"] = sc["phones"]
                    elif sc.get("phones"):
                        r.setdefault("all_phones", sc["phones"])
                    if sc.get("emails") and not r.get("email"):
                        r["email"] = sc["emails"][0]
                    if sc.get("address_hints") and not r.get("plant_address"):
                        r["plant_address"] = " | ".join(sc["address_hints"][:2])
                    if sc.get("gst") and not r.get("gst"):
                        r["gst"] = sc["gst"]
                    if sc.get("phones") or sc.get("emails"):
                        break
                db["results"][idx] = r
                _persist()
                st.success("✅ Contacts updated!")
            except Exception as e:
                st.warning(f"Lookup failed: {e}")

    # ── Empty state ────────────────────────────────────────────────────────────
    if not db["results"]:
        st.markdown("""
<div style="text-align:center;padding:80px 20px;color:#94a3b8">
  <div style="font-size:64px">🏭</div>
  <h3 style="color:#64748b;margin-top:16px;font-family:Inter,sans-serif">No manufacturers yet</h3>
  <p style="font-family:Inter,sans-serif">
    Go to <strong>🔍 Search</strong> and click <em>Search Once</em>
    or <em>Start Background Search</em>.
  </p>
</div>
""", unsafe_allow_html=True)

    else:
        # ── Metrics ───────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",        len(db["results"]))
        m2.metric("With Contact", sum(1 for r in db["results"] if r.get("phone") or r.get("email")))
        m3.metric("With Address", sum(1 for r in db["results"] if r.get("plant_address")))
        m4.metric("WHO-GMP",      sum(
            1 for r in db["results"]
            if any("gmp" in c.lower() for c in (r.get("certifications") or []))
        ))

        # ── Export row ────────────────────────────────────────────────────────
        df_all = pd.DataFrame(db["results"])
        ex1, ex2, ex3, _ = st.columns([1, 1, 1, 2])

        with ex1:
            st.download_button(
                "📥 CSV",
                df_all.to_csv(index=False).encode("utf-8"),
                f"cmo_india_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv", use_container_width=True,
            )
        with ex2:
            _buf = BytesIO()
            with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
                df_all.to_excel(_w, index=False, sheet_name="CMO Results")
            st.download_button(
                "📥 Excel", _buf.getvalue(),
                f"cmo_india_{datetime.now().strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with ex3:
            _snap = {
                "results":      db["results"],
                "seen_by_key":  {k: list(v) for k, v in db["seen_by_key"].items()},
                "batch_by_key": db["batch_by_key"],
                "exported_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.download_button(
                "📦 Backup JSON",
                data=json.dumps(_snap, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"cmo_backup_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )

        # ── Filters + view toggle ─────────────────────────────────────────────
        st.divider()
        fc1, fc2, fc3, fc4 = st.columns([1, 1, 2, 1])
        with fc1:
            f_contact = st.checkbox("Has contact")
        with fc2:
            f_address = st.checkbox("Has address")
        with fc3:
            f_certs = st.multiselect("Certifications", ["WHO-GMP", "USFDA", "EU GMP", "ISO 9001"])
        with fc4:
            view_mode = st.radio("View", ["Cards", "Table"], horizontal=True)

        # Apply filters
        display = db["results"].copy()
        if f_contact:
            display = [r for r in display if r.get("phone") or r.get("email")]
        if f_address:
            display = [r for r in display if r.get("plant_address")]
        if f_certs:
            display = [
                r for r in display
                if all(c.upper() in " ".join(r.get("certifications") or []).upper() for c in f_certs)
            ]

        if not display:
            st.warning("No results match your filters.")

        elif view_mode == "Table":
            df_d = pd.DataFrame(display)
            COLS = ["company_name", "city", "state", "phone", "email",
                    "certifications", "dosage_forms", "found_at"]
            avail = [c for c in COLS if c in df_d.columns]
            tbl = df_d[avail].copy()
            tbl.columns = [c.replace("_", " ").title() for c in avail]
            for col in ["Certifications", "Dosage Forms"]:
                if col in tbl.columns:
                    tbl[col] = tbl[col].apply(
                        lambda v: ", ".join(v) if isinstance(v, list) else (v or "")
                    )
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        else:  # Cards
            for ci, r in enumerate(display):
                true_idx = next(
                    (j for j, x in enumerate(db["results"])
                     if x.get("company_name") == r.get("company_name")), ci
                )
                name  = r.get("company_name", "Unknown")
                loc   = ", ".join(filter(None, [r.get("city", ""), r.get("state", "")])) or "India"
                certs = r.get("certifications") or []
                forms = r.get("dosage_forms") or []

                with st.expander(f"🏭  {name}  ·  {loc}", expanded=False):
                    col_l, col_r = st.columns(2)

                    with col_l:
                        if r.get("plant_address"):
                            st.markdown('<div class="sect">📍 Address</div>', unsafe_allow_html=True)
                            st.write(r["plant_address"])
                        if r.get("gst"):
                            st.caption(f"GST: {r['gst']}")

                        st.markdown('<div class="sect">📞 Contact</div>', unsafe_allow_html=True)
                        lines = []
                        if r.get("contact_person"): lines.append(f"👤 {r['contact_person']}")
                        if r.get("phone"):          lines.append(f"📱 {r['phone']}")
                        for x in (r.get("all_phones") or [])[1:3]:
                            lines.append(f"📱 {x}")
                        if r.get("email"):          lines.append(f"📧 {r['email']}")
                        if r.get("website"):        lines.append(f"🌐 [{r['website']}]({r['website']})")
                        st.markdown("\n\n".join(lines) if lines else "*Not found*")

                        _lbl = "🔍 Find Contacts" if not (r.get("phone") or r.get("email")) else "🔄 Update"
                        if st.button(_lbl, key=f"lk_{true_idx}", use_container_width=True):
                            st.session_state.pending_lookup = true_idx
                            st.rerun()

                    with col_r:
                        if forms:
                            st.markdown('<div class="sect">💊 Dosage Forms</div>', unsafe_allow_html=True)
                            st.markdown(
                                "".join(f'<span class="pill pill-green">{f}</span>' for f in forms),
                                unsafe_allow_html=True,
                            )
                        if certs:
                            st.markdown('<div class="sect">🏆 Certifications</div>', unsafe_allow_html=True)
                            st.markdown(
                                "".join(f'<span class="pill pill-blue">{c}</span>' for c in certs),
                                unsafe_allow_html=True,
                            )
                        if r.get("capacity"):
                            st.markdown('<div class="sect">⚙️ Capacity</div>', unsafe_allow_html=True)
                            st.write(r["capacity"])
                        if r.get("specialisation"):
                            st.markdown('<div class="sect">🔬 Specialisation</div>', unsafe_allow_html=True)
                            st.write(r["specialisation"])

                    if r.get("description"):
                        st.markdown('<div class="sect">📝 About</div>', unsafe_allow_html=True)
                        st.write(r["description"])
                    if r.get("source_url"):
                        st.markdown(
                            f'<a href="{r["source_url"]}" target="_blank" '
                            f'style="font-size:12px;color:#64748b">🔗 Source</a>',
                            unsafe_allow_html=True,
                        )

        # ── Import backup (collapsible) ────────────────────────────────────────
        st.divider()
        with st.expander("📥 Restore from backup"):
            st.markdown(
                "Upload a previously downloaded **Backup JSON** to restore manufacturers.\n\n"
                "⚠️ **Important:** Always download a backup before the app goes to sleep "
                "(Streamlit Cloud free tier resets data on restart). "
                "Click **📦 Backup JSON** above regularly to save your data."
            )
            uploaded = st.file_uploader("Choose backup JSON", type=["json"], key="import_up")
            if uploaded:
                try:
                    imp     = json.loads(uploaded.read().decode("utf-8"))
                    imp_res = imp.get("results", [])
                    existing_names = {r.get("company_name", "").strip().lower()
                                      for r in db["results"]}
                    added = 0
                    for r in imp_res:
                        n = r.get("company_name", "").strip().lower()
                        if n and n not in existing_names:
                            db["results"].append(r)
                            existing_names.add(n)
                            added += 1
                    for k, urls in imp.get("seen_by_key", {}).items():
                        db["seen_by_key"][k] = db["seen_by_key"].get(k, set()) | set(urls)
                    for k, cnt in imp.get("batch_by_key", {}).items():
                        db["batch_by_key"][k] = max(db["batch_by_key"].get(k, 0), cnt)
                    _persist()
                    st.success(
                        f"✅ Imported **{added}** manufacturers. "
                        f"Database now has **{len(db['results'])} total**."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
