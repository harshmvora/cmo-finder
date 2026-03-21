"""
CMO / CDMO Finder — India
AI-powered dashboard to discover Indian contract pharma manufacturers.
"""

import json
from io import BytesIO
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from ddgs import DDGS

from agent.extractor import extract_from_rich, extract_from_snippet, get_last_error, REASON_NOT_TPM, REASON_API_ERR
from agent.scraper import scrape_rich
from agent.searcher import DOSAGE_FORM_KEYWORDS, HUB_GROUPS, search_cmos, search_company_contacts
from agent import persistence

# ── Config persistence (API key) ──────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / ".cmo_config.json"

def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    except Exception:
        return {}

def _save_config(data: dict):
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
# SHARED IN-MEMORY DATABASE  (st.cache_resource)
#
# Unlike st.session_state (which is per-browser-tab and resets on refresh),
# cache_resource lives for the entire server-process lifetime.
# Every tab, every refresh, every rerun sees the SAME db object.
# Disk is used as a backup: loaded on first boot, written after every change.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _db() -> dict:
    """
    Singleton shared database.  Initialised once from disk (or empty).
    Survives page refreshes, new browser tabs, and Streamlit reruns.
    Only resets when the server process restarts (app sleep/redeploy).
    """
    disk = persistence.load()
    return {
        "results":      disk.get("results", []),
        "seen_by_key":  {k: set(v) for k, v in disk.get("seen_by_key", {}).items()},
        "batch_by_key": dict(disk.get("batch_by_key", {})),
        "last_saved":   disk.get("last_saved", ""),
        "save_error":   "",
    }


def _persist() -> None:
    """Write the shared db to disk and update save status."""
    db = _db()
    err = persistence.save({
        "results":      db["results"],
        "seen_by_key":  {k: list(v) for k, v in db["seen_by_key"].items()},
        "batch_by_key": db["batch_by_key"],
    })
    db["save_error"] = err
    if not err:
        db["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M")


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMO / CDMO Finder | India",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.header-banner {
    background: linear-gradient(135deg, #0f3460 0%, #0d7a4e 100%);
    color: white; padding: 22px 32px; border-radius: 14px; margin-bottom: 18px;
}
.header-banner h1 { margin: 0; font-size: 26px; letter-spacing: -0.5px; }
.header-banner p  { margin: 5px 0 0; opacity: 0.82; font-size: 14px; }
.pill {
    display: inline-block; padding: 2px 11px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin: 2px 3px 2px 0;
}
.pill-green { background: #dcfce7; color: #166534; }
.pill-blue  { background: #dbeafe; color: #1e40af; }
.pill-gray  { background: #f1f5f9; color: #475569; }
.section-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: #64748b; margin: 10px 0 3px;
}
.stProgress > div > div { background-color: #0d7a4e !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state (UI-only — not database) ─────────────────────────────────────
for _k, _v in [
    ("pending_lookup", None),
    ("search_batch",   0),
    ("auto_continue",  False),
    ("auto_params",    {}),
    ("seen_urls",      set()),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Convenience shortcut ───────────────────────────────────────────────────────
db = _db()   # shared database dict — use db["results"] etc. everywhere

# ── Header ─────────────────────────────────────────────────────────────────────
total_db   = len(db["results"])
last_saved = db["last_saved"]

st.markdown(f"""
<div class="header-banner">
  <h1>🏭 CMO &amp; CDMO Finder — India</h1>
  <p>AI-powered search · Scrapes plant address &amp; contacts · Deep contact-page hunting
  {'&nbsp;&nbsp;|&nbsp;&nbsp;<strong>🗄️ ' + str(total_db) + ' manufacturers in database</strong>' if total_db else ''}
  {'&nbsp;&nbsp;|&nbsp;&nbsp;last saved ' + last_saved if last_saved else ''}
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
_config = _load_config()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Claude API Key", type="password", placeholder="sk-ant-...",
        value=_default_api_key(),
    )
    if api_key and api_key != _config.get("api_key", ""):
        _save_config({**_config, "api_key": api_key})

    st.divider()
    st.markdown("### 🔎 Search Settings")
    max_results = st.slider("New URLs per batch", 5, 50, 25, 5)
    deep_scrape = st.checkbox("Deep scrape + contact-page hunting", value=True,
        help="Follows /contact-us pages and extracts phones/emails via regex.")

    st.divider()
    st.markdown("### 🔬 Filter Results")
    filter_contact = st.checkbox("Only show results with phone / email")
    filter_address = st.checkbox("Only show results with plant address")
    cert_filter = st.multiselect(
        "Must have certification",
        ["WHO-GMP", "USFDA", "EU GMP", "ISO 9001", "CGMP", "ISO 14001"],
    )

    # ── Search history ─────────────────────────────────────────────────────────
    if db["batch_by_key"]:
        st.divider()
        st.markdown("### 📊 Search History")
        for skey, batches in sorted(db["batch_by_key"].items(), key=lambda x: -x[1]):
            seen_count = len(db["seen_by_key"].get(skey, set()))
            label = persistence.key_label(skey)
            st.caption(f"**{label}**  \n{batches} batch{'es' if batches != 1 else ''} · {seen_count} URLs tried")

    st.divider()
    st.markdown("### 🔄 Auto-Continue")
    auto_continue = st.toggle("Keep searching non-stop",
        value=st.session_state.auto_continue,
        help="Runs a new batch automatically, rotating through different pharma hubs.")
    st.session_state.auto_continue = auto_continue
    if auto_continue and st.session_state.auto_params:
        hub_idx = st.session_state.search_batch % len(HUB_GROUPS)
        st.caption(f"Batch #{st.session_state.search_batch + 1} · Hub: {HUB_GROUPS[hub_idx]}")

    st.divider()
    if st.button("🗑️ Clear All Results", use_container_width=True):
        db["results"]      = []
        db["seen_by_key"]  = {}
        db["batch_by_key"] = {}
        db["last_saved"]   = ""
        db["save_error"]   = ""
        st.session_state.seen_urls      = set()
        st.session_state.pending_lookup = None
        st.session_state.search_batch   = 0
        st.session_state.auto_continue  = False
        st.session_state.auto_params    = {}
        persistence.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"💾 `{persistence.data_dir_path()}`")
    if db["save_error"]:
        st.warning(f"⚠️ Save failed: {db['save_error']}")
    elif db["last_saved"]:
        st.caption(f"✅ Saved {db['last_saved']}")
    st.caption("Powered by Claude Haiku · DuckDuckGo")

# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_search, tab_db, tab_io = st.tabs(["🔍 Search", "🗄️ Database", "📤 Import / Export"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:

    col_form, col_prod, col_req = st.columns([2, 2, 2])
    with col_form:
        st.markdown("#### 💊 Dosage Form(s)")
        dosage_forms = st.multiselect(
            "Dosage forms", list(DOSAGE_FORM_KEYWORDS.keys()),
            default=["Tablets"], label_visibility="collapsed",
        )
    with col_prod:
        st.markdown("#### 🧪 Product Name *(optional)*")
        product_name = st.text_input(
            "product_name",
            placeholder="e.g. Metformin 500mg, Amoxicillin...",
            label_visibility="collapsed",
        )
    with col_req:
        st.markdown("#### 📋 Other Requirements *(optional)*")
        requirements = st.text_area(
            "requirements",
            placeholder="e.g. WHO-GMP, min 10M units/month, Gujarat...",
            height=68, label_visibility="collapsed",
        )

    # Resume hint
    if dosage_forms:
        _preview_key  = persistence.make_key(dosage_forms, product_name)
        _done_batches = db["batch_by_key"].get(_preview_key, 0)
        _done_urls    = len(db["seen_by_key"].get(_preview_key, set()))
        if _done_batches:
            st.info(
                f"📂 **Resuming** — {_done_batches} batch{'es' if _done_batches != 1 else ''} "
                f"done for *{persistence.key_label(_preview_key)}* · "
                f"{_done_urls} URLs already tried. "
                f"Continuing from batch #{_done_batches + 1}."
            )

    _, btn_col, stop_col, _ = st.columns([1, 2, 1, 1])
    with btn_col:
        search_clicked = st.button(
            "🔍  Find Manufacturers", use_container_width=True, type="primary",
            disabled=(not api_key) or (not dosage_forms),
        )
    with stop_col:
        if st.button("⏹ Stop", use_container_width=True,
                     disabled=not st.session_state.auto_continue):
            st.session_state.auto_continue = False
            st.rerun()

    if not api_key:
        st.info("👈  Enter your Claude API key in the sidebar to get started.")

    _auto_trigger = (
        st.session_state.auto_continue
        and st.session_state.auto_params
        and not search_clicked
    )

    # ── Search execution ───────────────────────────────────────────────────────
    if (search_clicked or _auto_trigger) and api_key and dosage_forms:
        client = anthropic.Anthropic(api_key=api_key)

        if search_clicked:
            skey = persistence.make_key(dosage_forms, product_name)
            # Restore batch/seen state from shared db for this key
            st.session_state.search_batch = db["batch_by_key"].get(skey, 0)
            st.session_state.seen_urls    = set(db["seen_by_key"].get(skey, set()))
            st.session_state.auto_params  = {
                "dosage_forms": dosage_forms,
                "product_name": product_name,
                "requirements": requirements,
                "max_results":  max_results,
                "search_key":   skey,
            }

        params = st.session_state.auto_params
        batch  = st.session_state.search_batch
        skey   = params["search_key"]

        progress_bar = st.progress(0)
        status       = st.empty()

        hub_idx = batch % len(HUB_GROUPS)
        status.markdown(
            f"🔎 **Batch #{batch + 1}** · Hub: *{HUB_GROUPS[hub_idx]}* · "
            f"Search: *{persistence.key_label(skey)}*"
        )
        progress_bar.progress(8)

        try:
            search_hits = search_cmos(
                params["dosage_forms"], params["product_name"],
                params["requirements"], params["max_results"],
                batch=batch,
                already_seen=set(st.session_state.seen_urls),
            )
        except Exception as search_err:
            st.error(f"Search error: {search_err}")
            st.stop()

        new_seen = set(st.session_state.seen_urls) | {h["url"] for h in search_hits}
        st.session_state.seen_urls = new_seen

        # Always update counters + save, even on empty batch
        db["seen_by_key"][skey]  = new_seen
        db["batch_by_key"][skey] = batch + 1
        st.session_state.search_batch = batch + 1

        if not search_hits:
            _persist()
            if st.session_state.auto_continue:
                import time as _t; _t.sleep(1)
                st.rerun()
            else:
                st.warning("No new URLs found this batch. Click Search again to continue with the next hub.")
                st.stop()

        status.markdown(f"📄 **{len(search_hits)} candidate pages.** Extracting with Claude…")
        progress_bar.progress(18)

        new_results: list[dict] = []
        stat_scraped = stat_claude_ok = stat_not_tpm = stat_api_err = stat_scrape_fail = 0
        total = len(search_hits)

        for i, hit in enumerate(search_hits):
            progress_bar.progress(18 + int((i / total) * 78))
            status.markdown(f"🤖 **{i+1}/{total}** — *{hit['title'][:70]}*")

            extracted  = None
            reason     = ""
            dosage_ctx = hit["dosage_form"] + (
                f" — {params['product_name']}" if params.get("product_name") else ""
            )

            if deep_scrape:
                scraped = scrape_rich(hit["url"], follow_contact=True)
                if scraped:
                    stat_scraped += 1
                    extracted, reason = extract_from_rich(scraped, hit["url"], dosage_ctx, client)
                else:
                    stat_scrape_fail += 1

            if not extracted and hit.get("snippet"):
                extracted, reason = extract_from_snippet(
                    hit["title"], hit["snippet"], hit["url"], dosage_ctx, client
                )

            if reason == REASON_NOT_TPM:
                stat_not_tpm += 1
            elif reason == REASON_API_ERR:
                stat_api_err += 1
            elif extracted and extracted.get("company_name"):
                stat_claude_ok += 1
                extracted["searched_dosage_form"] = hit["dosage_form"]
                extracted["search_key"]           = skey
                extracted["found_at"]             = datetime.now().strftime("%Y-%m-%d %H:%M")
                new_results.append(extracted)

        progress_bar.progress(100)

        # ── Diagnostic summary ────────────────────────────────────────────────
        api_err_msg = get_last_error()
        if api_err_msg:
            status.markdown(f"⚠️ **Claude API error:** `{api_err_msg}`")
            st.error(
                f"Claude API is returning errors: **{api_err_msg}**\n\n"
                "Check your API key in the sidebar — it may have expired or be invalid."
            )
        else:
            diag_parts = [
                f"**{total}** URLs tried",
                f"**{stat_scraped}** scraped",
                f"**{stat_claude_ok}** passed AI filter",
                f"**{stat_not_tpm}** rejected (not TPM/contract mfr)",
            ]
            if stat_api_err:  diag_parts.append(f"**{stat_api_err}** Claude errors")
            if stat_scrape_fail: diag_parts.append(f"**{stat_scrape_fail}** scrape failures")
            status.markdown(f"✅ **Batch done:** " + " · ".join(diag_parts))

        # Deduplicate and accumulate into shared db
        existing = {r.get("company_name", "").strip().lower() for r in db["results"]}
        added = 0
        for r in new_results:
            key = r.get("company_name", "").strip().lower()
            if key and key not in existing:
                db["results"].append(r)
                existing.add(key)
                added += 1

        _persist()

        if added:
            st.success(
                f"✅ Added **{added}** new manufacturer(s) — "
                f"**{len(db['results'])} total** in database. "
                f"Switch to the **🗄️ Database** tab to view all."
            )
        else:
            st.info("No new unique manufacturers found this batch (all already in database).")

        if st.session_state.auto_continue:
            import time as _t; _t.sleep(2)
            st.rerun()

    # Recent results preview
    if db["results"]:
        st.divider()
        st.markdown(
            f"#### 🏭 Recent Results  "
            f"<span style='font-size:13px;color:#64748b'>({len(db['results'])} total — "
            f"open the **🗄️ Database** tab for the full list)</span>",
            unsafe_allow_html=True,
        )
        for r in db["results"][-5:][::-1]:
            name  = r.get("company_name", "?")
            loc   = ", ".join(filter(None, [r.get("city",""), r.get("state","")])) or "India"
            phone = r.get("phone", "")
            st.markdown(f"**{name}** — {loc}" + (f" · 📱 {phone}" if phone else ""))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_db:

    # Handle pending contact lookup
    if st.session_state.pending_lookup is not None and api_key:
        idx = st.session_state.pending_lookup
        st.session_state.pending_lookup = None
        r = db["results"][idx]

        with st.spinner(f"Finding contacts for **{r.get('company_name', '')}**…"):
            try:
                with DDGS() as ddgs:
                    hits = search_company_contacts(r.get("company_name", ""), ddgs)
                for hit in hits:
                    scraped = scrape_rich(hit["url"], follow_contact=True)
                    if not scraped:
                        continue
                    if scraped.get("phones") and not r.get("phone"):
                        r["phone"]      = scraped["phones"][0]
                        r["all_phones"] = scraped["phones"]
                    elif scraped.get("phones"):
                        r.setdefault("all_phones", scraped["phones"])
                    if scraped.get("emails") and not r.get("email"):
                        r["email"] = scraped["emails"][0]
                    if scraped.get("address_hints") and not r.get("plant_address"):
                        r["plant_address"] = " | ".join(scraped["address_hints"][:2])
                    if scraped.get("gst") and not r.get("gst"):
                        r["gst"] = scraped["gst"]
                    if scraped.get("phones") or scraped.get("emails"):
                        break
                db["results"][idx] = r
                _persist()
                st.success("Contact details updated!")
            except Exception as e:
                st.warning(f"Contact lookup failed: {e}")

    if not db["results"]:
        st.markdown("""
<div style="text-align:center; padding:60px 20px; color:#94a3b8;">
  <div style="font-size:56px">🗄️</div>
  <h3 style="color:#64748b; margin-top:12px;">Database is empty</h3>
  <p>Go to the <strong>🔍 Search</strong> tab and run a search.<br>
  Every manufacturer found is automatically added here.</p>
  <p style="font-size:12px;margin-top:16px;">
    You can also restore a previous database using the <strong>📤 Import / Export</strong> tab.
  </p>
</div>
""", unsafe_allow_html=True)
    else:
        # Filters
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            db_fc = st.checkbox("Has phone / email", key="db_fc")
        with fc2:
            db_fa = st.checkbox("Has address", key="db_fa")
        with fc3:
            db_cert = st.multiselect("Certifications",
                ["WHO-GMP", "USFDA", "EU GMP", "ISO 9001", "CGMP"], key="db_cert")

        all_keys = sorted({r.get("search_key","") for r in db["results"] if r.get("search_key")})
        if len(all_keys) > 1:
            sel_label = st.selectbox(
                "Filter by search",
                ["All searches"] + [persistence.key_label(k) for k in all_keys],
                key="db_key_filter",
            )
        else:
            sel_label = "All searches"

        display = db["results"].copy()
        if db_fc:   display = [r for r in display if r.get("phone") or r.get("email")]
        if db_fa:   display = [r for r in display if r.get("plant_address") or r.get("city")]
        if db_cert: display = [r for r in display if all(
            c.upper() in " ".join(r.get("certifications") or []).upper() for c in db_cert)]
        if sel_label != "All searches":
            sel_key = next((k for k in all_keys if persistence.key_label(k) == sel_label), None)
            if sel_key:
                display = [r for r in display if r.get("search_key") == sel_key]

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total in Database", len(db["results"]))
        m2.metric("With Contact",  sum(1 for r in db["results"] if r.get("phone") or r.get("email")))
        m3.metric("With Address",  sum(1 for r in db["results"] if r.get("plant_address")))
        m4.metric("Showing",       len(display))

        # Export
        df_all = pd.DataFrame(db["results"])
        ex1, ex2, _ = st.columns([1, 1, 3])
        with ex1:
            st.download_button(
                "📥 Export CSV",
                df_all.to_csv(index=False).encode("utf-8"),
                f"cmo_india_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True,
            )
        with ex2:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df_all.to_excel(w, index=False, sheet_name="CMO Results")
            st.download_button(
                "📥 Export Excel", buf.getvalue(),
                f"cmo_india_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.divider()
        view = st.radio("View mode", ["Table", "Cards"], horizontal=True, key="db_view")

        if not display:
            st.warning("No results match the current filters.")

        elif view == "Table":
            df_disp = pd.DataFrame(display)
            TABLE_COLS = ["company_name", "city", "state", "phone", "email",
                          "certifications", "dosage_forms", "website", "found_at"]
            avail = [c for c in TABLE_COLS if c in df_disp.columns]
            tbl   = df_disp[avail].copy()
            tbl.columns = [c.replace("_", " ").title() for c in avail]
            for col in ["Certifications", "Dosage Forms"]:
                if col in tbl.columns:
                    tbl[col] = tbl[col].apply(
                        lambda v: ", ".join(v) if isinstance(v, list) else (v or ""))
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        else:
            for card_i, r in enumerate(display):
                true_idx = next(
                    (j for j, x in enumerate(db["results"])
                     if x.get("company_name") == r.get("company_name")), card_i)
                name  = r.get("company_name", "Unknown")
                loc   = ", ".join(filter(None, [r.get("city",""), r.get("state","")])) or "India"
                filled = sum(bool(r.get(f)) for f in
                             ["phone","email","plant_address","certifications","dosage_forms"])
                score  = f"{'●'*filled}{'○'*(5-filled)}"

                with st.expander(f"🏭  {name}  —  {loc}  |  {score}", expanded=False):
                    left, right = st.columns(2)
                    with left:
                        st.markdown('<div class="section-label">📍 Address</div>', unsafe_allow_html=True)
                        st.write(r.get("plant_address") or loc or "—")
                        if r.get("gst"): st.caption(f"GST: {r['gst']}")

                        st.markdown('<div class="section-label">📞 Contact</div>', unsafe_allow_html=True)
                        lines = []
                        if r.get("contact_person"): lines.append(f"👤 {r['contact_person']}")
                        if r.get("phone"):          lines.append(f"📱 {r['phone']}")
                        for x in (r.get("all_phones") or [])[1:3]: lines.append(f"📱 {x}")
                        if r.get("email"):          lines.append(f"📧 {r['email']}")
                        if r.get("website"):        lines.append(f"🌐 [{r['website']}]({r['website']})")
                        st.markdown("\n\n".join(lines) if lines else "—")

                        btn_lbl = "🔍 Find Contacts" if not (r.get("phone") or r.get("email")) else "🔄 Refresh"
                        if st.button(btn_lbl, key=f"lookup_{true_idx}", use_container_width=True):
                            st.session_state.pending_lookup = true_idx
                            st.rerun()

                    with right:
                        st.markdown('<div class="section-label">💊 Dosage Forms</div>', unsafe_allow_html=True)
                        forms = r.get("dosage_forms") or []
                        st.markdown(
                            "".join(f'<span class="pill pill-green">{f}</span>' for f in forms)
                            if forms else "—", unsafe_allow_html=True)

                        st.markdown('<div class="section-label">🏆 Certifications</div>', unsafe_allow_html=True)
                        certs = r.get("certifications") or []
                        st.markdown(
                            "".join(f'<span class="pill pill-blue">{c}</span>' for c in certs)
                            if certs else "—", unsafe_allow_html=True)

                        for lbl, field in [("⚙️ Capacity","capacity"),("📦 Min Order","min_order")]:
                            if r.get(field):
                                st.markdown(f'<div class="section-label">{lbl}</div>', unsafe_allow_html=True)
                                st.write(r[field])

                        if r.get("search_key"):
                            st.markdown('<div class="section-label">🔎 Found via</div>', unsafe_allow_html=True)
                            st.markdown(f'<span class="pill pill-gray">{persistence.key_label(r["search_key"])}</span>', unsafe_allow_html=True)

                    if r.get("description"):
                        st.markdown('<div class="section-label">📝 About</div>', unsafe_allow_html=True)
                        st.write(r["description"])
                    if r.get("source_url"):
                        st.markdown(f'<a href="{r["source_url"]}" target="_blank">🔗 Source</a>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMPORT / EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_io:
    st.markdown("### 📤 Export Database")
    st.markdown(
        "Download the full database + search history as a JSON backup. "
        "Re-upload to restore everything after the app restarts."
    )
    _snap = {
        "results":      db["results"],
        "seen_by_key":  {k: list(v) for k, v in db["seen_by_key"].items()},
        "batch_by_key": db["batch_by_key"],
        "last_saved":   db["last_saved"],
        "exported_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.download_button(
        f"📥 Download backup  ({len(db['results'])} manufacturers)",
        data=json.dumps(_snap, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"cmo_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        disabled=(not db["results"]),
    )

    st.divider()
    st.markdown("### 📥 Import / Restore Database")
    st.markdown("Upload a backup JSON to restore. Existing results are **merged** — duplicates skipped.")

    uploaded = st.file_uploader("Choose a backup JSON file", type=["json"])
    if uploaded:
        try:
            imported     = json.loads(uploaded.read().decode("utf-8"))
            imp_results  = imported.get("results", [])
            imp_seen     = imported.get("seen_by_key", {})
            imp_batches  = imported.get("batch_by_key", {})

            existing_names = {r.get("company_name","").strip().lower() for r in db["results"]}
            added = 0
            for r in imp_results:
                n = r.get("company_name","").strip().lower()
                if n and n not in existing_names:
                    db["results"].append(r)
                    existing_names.add(n)
                    added += 1

            for k, urls in imp_seen.items():
                db["seen_by_key"][k] = db["seen_by_key"].get(k, set()) | set(urls)

            for k, cnt in imp_batches.items():
                db["batch_by_key"][k] = max(db["batch_by_key"].get(k, 0), cnt)

            _persist()
            st.success(
                f"✅ Import complete — added **{added}** new manufacturers. "
                f"Database now has **{len(db['results'])} total**."
            )
            st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")

    st.divider()
    st.markdown("### ℹ️ Storage Info")
    st.markdown(f"- **File:** `{persistence.data_dir_path()}`")
    st.markdown(f"- **Manufacturers:** {len(db['results'])}")
    st.markdown(f"- **Search contexts:** {len(db['batch_by_key'])}")
    if db["last_saved"]:  st.markdown(f"- **Last saved:** {db['last_saved']}")
    if db["save_error"]:
        st.error(f"**Save error:** {db['save_error']}")
        st.info("Your data is safe in this server session. Use Export above to download a backup.")
