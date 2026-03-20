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

from agent.extractor import extract_from_rich, extract_from_snippet
from agent.scraper import scrape_rich
from agent.searcher import DOSAGE_FORM_KEYWORDS, HUB_GROUPS, search_cmos, search_company_contacts
from agent import persistence

# ── Config persistence (saves API key locally so user doesn't re-enter it) ───
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

# ── Helper: save state and surface any error ──────────────────────────────────
def _persist():
    """Save current session state to disk; show a sidebar warning if it fails."""
    err = persistence.save({
        "results":      st.session_state.results,
        "seen_by_key":  {k: list(v) for k, v in st.session_state._seen_by_key.items()},
        "batch_by_key": st.session_state._batch_by_key,
    })
    if err:
        st.session_state._save_error = err
    else:
        st.session_state._save_error = ""
        st.session_state._last_saved = datetime.now().strftime("%Y-%m-%d %H:%M")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMO / CDMO Finder | India",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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

# ── Boot: load persistent state once per session ──────────────────────────────
if "_pers_loaded" not in st.session_state:
    _disk = persistence.load()
    st.session_state.results        = _disk["results"]
    st.session_state._seen_by_key   = {k: set(v) for k, v in _disk["seen_by_key"].items()}
    st.session_state._batch_by_key  = dict(_disk["batch_by_key"])
    st.session_state._last_saved    = _disk.get("last_saved", "")
    st.session_state._save_error    = ""
    st.session_state._pers_loaded   = True

# Standard session state defaults
for _k, _v in [
    ("pending_lookup", None),
    ("search_batch",   0),
    ("auto_continue",  False),
    ("auto_params",    {}),
    ("seen_urls",      set()),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Header ────────────────────────────────────────────────────────────────────
total_db    = len(st.session_state.results)
last_saved  = st.session_state._last_saved

st.markdown(f"""
<div class="header-banner">
  <h1>🏭 CMO &amp; CDMO Finder — India</h1>
  <p>AI-powered search · Scrapes plant address &amp; contacts · Deep contact-page hunting
  {'&nbsp;&nbsp;|&nbsp;&nbsp;<strong>🗄️ ' + str(total_db) + ' manufacturers in database</strong>' if total_db else ''}
  {'&nbsp;&nbsp;|&nbsp;&nbsp;last saved ' + last_saved if last_saved else ''}
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
_config = _load_config()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Claude API Key", type="password", placeholder="sk-ant-...",
        value=_default_api_key(),
        help="Saved locally — you won't need to re-enter it next time",
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

    # ── Search history ────────────────────────────────────────────────────────
    if st.session_state._batch_by_key:
        st.divider()
        st.markdown("### 📊 Search History")
        for skey, batches in sorted(st.session_state._batch_by_key.items(), key=lambda x: -x[1]):
            seen_count = len(st.session_state._seen_by_key.get(skey, set()))
            label = persistence.key_label(skey)
            st.caption(f"**{label}**  \n{batches} batch{'es' if batches != 1 else ''} · {seen_count} URLs tried")

    st.divider()
    st.markdown("### 🔄 Auto-Continue")
    auto_continue = st.toggle("Keep searching non-stop",
        value=st.session_state.auto_continue,
        help="Runs a new batch automatically after each one, rotating pharma hubs.")
    st.session_state.auto_continue = auto_continue
    if auto_continue and st.session_state.auto_params:
        hub_idx = st.session_state.search_batch % len(HUB_GROUPS)
        st.caption(f"Batch #{st.session_state.search_batch + 1} · Hub: {HUB_GROUPS[hub_idx]}")

    st.divider()
    if st.button("🗑️ Clear All Results", use_container_width=True):
        st.session_state.results        = []
        st.session_state._seen_by_key   = {}
        st.session_state._batch_by_key  = {}
        st.session_state.seen_urls      = set()
        st.session_state.pending_lookup = None
        st.session_state.search_batch   = 0
        st.session_state.auto_continue  = False
        st.session_state.auto_params    = {}
        st.session_state._last_saved    = ""
        st.session_state._save_error    = ""
        persistence.clear()
        st.rerun()

    # ── Storage status ────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"💾 Storage: `{persistence.data_dir_path()}`")
    if st.session_state._save_error:
        st.warning(f"⚠️ Save failed: {st.session_state._save_error}")
    elif st.session_state._last_saved:
        st.caption(f"✅ Last saved: {st.session_state._last_saved}")
    st.caption("Powered by Claude Haiku · DuckDuckGo")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_search, tab_db, tab_io = st.tabs(["🔍 Search", "🗄️ Database", "📤 Import / Export"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:

    # ── Input form ────────────────────────────────────────────────────────────
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

    # ── Resume hint ───────────────────────────────────────────────────────────
    if dosage_forms:
        _preview_key   = persistence.make_key(dosage_forms, product_name)
        _done_batches  = st.session_state._batch_by_key.get(_preview_key, 0)
        _done_urls     = len(st.session_state._seen_by_key.get(_preview_key, set()))
        if _done_batches:
            st.info(
                f"📂 **Resuming** — {_done_batches} batch{'es' if _done_batches != 1 else ''} "
                f"already done for *{persistence.key_label(_preview_key)}* · "
                f"{_done_urls} URLs already tried. "
                f"New search continues from batch #{_done_batches + 1}."
            )

    # ── Search / Stop buttons ─────────────────────────────────────────────────
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

    # Auto-continue trigger
    _auto_trigger = (
        st.session_state.auto_continue
        and st.session_state.auto_params
        and not search_clicked
    )

    # ── Search execution ──────────────────────────────────────────────────────
    if (search_clicked or _auto_trigger) and api_key and dosage_forms:
        client = anthropic.Anthropic(api_key=api_key)

        if search_clicked:
            skey = persistence.make_key(dosage_forms, product_name)
            # Restore batch counter + seen URLs for this key → resume, not restart
            st.session_state.search_batch = st.session_state._batch_by_key.get(skey, 0)
            st.session_state.seen_urls    = set(st.session_state._seen_by_key.get(skey, set()))
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

        # Update seen URLs (no in-place mutation on Streamlit proxies)
        new_seen = set(st.session_state.seen_urls) | {h["url"] for h in search_hits}
        st.session_state.seen_urls = new_seen

        if not search_hits:
            st.session_state._batch_by_key[skey] = batch + 1
            st.session_state._seen_by_key[skey]  = new_seen
            st.session_state.search_batch        = batch + 1
            _persist()
            if st.session_state.auto_continue:
                import time as _t; _t.sleep(1)
                st.rerun()
            else:
                st.warning("No new URLs found for this batch. Try different dosage forms or click Search again to continue.")
                st.stop()

        status.markdown(f"📄 **{len(search_hits)} candidate pages found.** Extracting with Claude…")
        progress_bar.progress(18)

        new_results: list[dict] = []
        total = len(search_hits)
        for i, hit in enumerate(search_hits):
            progress_bar.progress(18 + int((i / total) * 78))
            status.markdown(f"🤖 **{i+1}/{total}** — *{hit['title'][:70]}*")

            extracted    = None
            dosage_ctx   = hit["dosage_form"] + (
                f" — {params['product_name']}" if params.get("product_name") else ""
            )

            if deep_scrape:
                scraped = scrape_rich(hit["url"], follow_contact=True)
                if scraped:
                    extracted = extract_from_rich(scraped, hit["url"], dosage_ctx, client)

            if not extracted and hit.get("snippet"):
                extracted = extract_from_snippet(
                    hit["title"], hit["snippet"], hit["url"], dosage_ctx, client
                )

            if extracted and extracted.get("company_name"):
                extracted["searched_dosage_form"] = hit["dosage_form"]
                extracted["search_key"]           = skey
                extracted["found_at"]             = datetime.now().strftime("%Y-%m-%d %H:%M")
                new_results.append(extracted)

        progress_bar.progress(100)
        status.markdown(f"✅ **Done!** Extracted **{len(new_results)}** manufacturers this batch.")

        # ── Deduplicate + accumulate ──────────────────────────────────────────
        existing = {r.get("company_name", "").strip().lower() for r in st.session_state.results}
        added = 0
        for r in new_results:
            key = r.get("company_name", "").strip().lower()
            if key and key not in existing:
                st.session_state.results.append(r)
                existing.add(key)
                added += 1

        if added:
            st.success(
                f"✅ Added **{added}** new manufacturer(s) — "
                f"**{len(st.session_state.results)} total** in database. "
                f"Switch to the **🗄️ Database** tab to view all."
            )
        else:
            st.info("No new unique manufacturers found this batch (all already in database).")

        # ── Update per-key counters + save ────────────────────────────────────
        st.session_state._seen_by_key[skey]  = new_seen
        st.session_state._batch_by_key[skey] = batch + 1
        st.session_state.search_batch        = batch + 1
        _persist()

        if st.session_state.auto_continue:
            import time as _t; _t.sleep(2)
            st.rerun()

    # ── Recent results (last batch found, compact view) ───────────────────────
    if st.session_state.results:
        st.divider()
        st.markdown(
            f"#### 🏭 Recent Results  "
            f"<span style='font-size:13px;color:#64748b'>({len(st.session_state.results)} total — "
            f"open the **🗄️ Database** tab for the full list)</span>",
            unsafe_allow_html=True,
        )
        # Show last 5 found
        recent = st.session_state.results[-5:][::-1]
        for r in recent:
            name  = r.get("company_name", "?")
            city  = r.get("city", "")
            state = r.get("state", "")
            loc   = ", ".join(filter(None, [city, state])) or "India"
            phone = r.get("phone", "")
            st.markdown(
                f"**{name}** — {loc}"
                + (f" · 📱 {phone}" if phone else ""),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATABASE (full cumulative list, always visible)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_db:

    # ── Handle pending contact lookup ─────────────────────────────────────────
    if st.session_state.pending_lookup is not None and api_key:
        idx = st.session_state.pending_lookup
        st.session_state.pending_lookup = None
        r   = st.session_state.results[idx]

        with st.spinner(f"Finding contacts for **{r.get('company_name', '')}**…"):
            try:
                with DDGS() as ddgs:
                    hits = search_company_contacts(r.get("company_name", ""), ddgs)
                for hit in hits:
                    scraped = scrape_rich(hit["url"], follow_contact=True)
                    if not scraped:
                        continue
                    if scraped.get("phones") and not r.get("phone"):
                        r["phone"]     = scraped["phones"][0]
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
                st.session_state.results[idx] = r
                _persist()
                st.success("Contact details updated!")
            except Exception as e:
                st.warning(f"Contact lookup failed: {e}")

    if not st.session_state.results:
        st.markdown("""
<div style="text-align:center; padding:60px 20px; color:#94a3b8;">
  <div style="font-size:56px">🗄️</div>
  <h3 style="color:#64748b; margin-top:12px;">Database is empty</h3>
  <p>Go to the <strong>🔍 Search</strong> tab and run a search.<br>
  Every manufacturer found is automatically saved here.</p>
  <p style="font-size:12px;margin-top:16px;">
    You can also restore a previous database using the <strong>📤 Import / Export</strong> tab.
  </p>
</div>
""", unsafe_allow_html=True)
    else:
        # ── Filters ───────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            db_filter_contact = st.checkbox("Has phone / email", key="db_fc")
        with fc2:
            db_filter_address = st.checkbox("Has address", key="db_fa")
        with fc3:
            db_cert_filter = st.multiselect("Certifications", ["WHO-GMP", "USFDA", "EU GMP", "ISO 9001", "CGMP"], key="db_cert")

        all_keys = sorted({r.get("search_key", "") for r in st.session_state.results if r.get("search_key")})
        if len(all_keys) > 1:
            key_options   = ["All searches"] + [persistence.key_label(k) for k in all_keys]
            selected_label = st.selectbox("Filter by search", key_options, key="db_key_filter")
        else:
            selected_label = "All searches"

        display = st.session_state.results.copy()
        if db_filter_contact:
            display = [r for r in display if r.get("phone") or r.get("email")]
        if db_filter_address:
            display = [r for r in display if r.get("plant_address") or r.get("city")]
        if db_cert_filter:
            display = [r for r in display if all(
                c.upper() in " ".join(r.get("certifications") or []).upper()
                for c in db_cert_filter
            )]
        if selected_label != "All searches":
            sel_key = next((k for k in all_keys if persistence.key_label(k) == selected_label), None)
            if sel_key:
                display = [r for r in display if r.get("search_key") == sel_key]

        # ── Metrics ───────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total in Database", len(st.session_state.results))
        m2.metric("With Contact",      sum(1 for r in st.session_state.results if r.get("phone") or r.get("email")))
        m3.metric("With Address",      sum(1 for r in st.session_state.results if r.get("plant_address")))
        m4.metric("Showing",           len(display))

        # ── Export buttons ────────────────────────────────────────────────────
        df_all  = pd.DataFrame(st.session_state.results)
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
                        lambda v: ", ".join(v) if isinstance(v, list) else (v or "")
                    )
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        else:
            # ── Card view ─────────────────────────────────────────────────────
            for card_i, r in enumerate(display):
                true_idx = next(
                    (j for j, x in enumerate(st.session_state.results)
                     if x.get("company_name") == r.get("company_name")),
                    card_i,
                )
                name  = r.get("company_name", "Unknown")
                city  = r.get("city", "")
                state = r.get("state", "")
                loc   = ", ".join(filter(None, [city, state])) or "India"
                filled = sum(bool(r.get(f)) for f in
                             ["phone", "email", "plant_address", "certifications", "dosage_forms"])
                score_label = f"{'●' * filled}{'○' * (5 - filled)}"

                with st.expander(f"🏭  {name}  —  {loc}  |  {score_label}", expanded=False):
                    left, right = st.columns(2)
                    with left:
                        st.markdown('<div class="section-label">📍 Address</div>', unsafe_allow_html=True)
                        st.write(r.get("plant_address") or loc or "—")
                        if r.get("gst"):
                            st.caption(f"GST: {r['gst']}")

                        st.markdown('<div class="section-label">📞 Contact</div>', unsafe_allow_html=True)
                        lines = []
                        if r.get("contact_person"): lines.append(f"👤 {r['contact_person']}")
                        if r.get("phone"):          lines.append(f"📱 {r['phone']}")
                        for extra in (r.get("all_phones") or [])[1:3]:
                                                    lines.append(f"📱 {extra}")
                        if r.get("email"):          lines.append(f"📧 {r['email']}")
                        if r.get("website"):        lines.append(f"🌐 [{r['website']}]({r['website']})")
                        st.markdown("\n\n".join(lines) if lines else "—")

                        btn_label = "🔍 Find Contacts" if not (r.get("phone") or r.get("email")) else "🔄 Refresh Contacts"
                        if st.button(btn_label, key=f"lookup_{true_idx}", use_container_width=True):
                            st.session_state.pending_lookup = true_idx
                            st.rerun()

                    with right:
                        st.markdown('<div class="section-label">💊 Dosage Forms</div>', unsafe_allow_html=True)
                        forms = r.get("dosage_forms") or []
                        if forms:
                            st.markdown("".join(f'<span class="pill pill-green">{f}</span>' for f in forms), unsafe_allow_html=True)
                        else:
                            st.write("—")

                        st.markdown('<div class="section-label">🏆 Certifications</div>', unsafe_allow_html=True)
                        certs = r.get("certifications") or []
                        if certs:
                            st.markdown("".join(f'<span class="pill pill-blue">{c}</span>' for c in certs), unsafe_allow_html=True)
                        else:
                            st.write("—")

                        if r.get("capacity"):
                            st.markdown('<div class="section-label">⚙️ Capacity</div>', unsafe_allow_html=True)
                            st.write(r["capacity"])
                        if r.get("min_order"):
                            st.markdown('<div class="section-label">📦 Min Order</div>', unsafe_allow_html=True)
                            st.write(r["min_order"])
                        if r.get("search_key"):
                            st.markdown('<div class="section-label">🔎 Found via</div>', unsafe_allow_html=True)
                            st.markdown(f'<span class="pill pill-gray">{persistence.key_label(r["search_key"])}</span>', unsafe_allow_html=True)

                    if r.get("description"):
                        st.markdown('<div class="section-label">📝 About</div>', unsafe_allow_html=True)
                        st.write(r["description"])
                    if r.get("source_url"):
                        st.markdown(f'<a href="{r["source_url"]}" target="_blank">🔗 View Source</a>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMPORT / EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_io:
    st.markdown("### 📤 Export Database")
    st.markdown(
        "Download the **full database + search history** as a JSON backup. "
        "You can re-upload this file later to restore all data (even after the app restarts)."
    )

    # Build the full state snapshot for download
    _snap = {
        "results":      st.session_state.results,
        "seen_by_key":  {k: list(v) for k, v in st.session_state._seen_by_key.items()},
        "batch_by_key": st.session_state._batch_by_key,
        "last_saved":   st.session_state._last_saved,
        "exported_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.download_button(
        f"📥 Download backup JSON  ({len(st.session_state.results)} manufacturers)",
        data=json.dumps(_snap, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"cmo_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=False,
        disabled=(not st.session_state.results),
    )

    st.divider()
    st.markdown("### 📥 Import / Restore Database")
    st.markdown(
        "Upload a previously exported JSON backup to restore manufacturers and search history. "
        "**Existing results will be merged** — duplicates (by company name) are skipped."
    )

    uploaded = st.file_uploader("Choose a backup JSON file", type=["json"])
    if uploaded:
        try:
            imported = json.loads(uploaded.read().decode("utf-8"))
            imp_results  = imported.get("results", [])
            imp_seen     = imported.get("seen_by_key", {})
            imp_batches  = imported.get("batch_by_key", {})

            # Merge results (skip duplicates)
            existing_names = {r.get("company_name", "").strip().lower()
                              for r in st.session_state.results}
            added = 0
            for r in imp_results:
                n = r.get("company_name", "").strip().lower()
                if n and n not in existing_names:
                    st.session_state.results.append(r)
                    existing_names.add(n)
                    added += 1

            # Merge seen URLs per key
            for k, urls in imp_seen.items():
                existing_set = st.session_state._seen_by_key.get(k, set())
                st.session_state._seen_by_key[k] = existing_set | set(urls)

            # Merge batch counters (take the max)
            for k, cnt in imp_batches.items():
                st.session_state._batch_by_key[k] = max(
                    st.session_state._batch_by_key.get(k, 0), cnt
                )

            _persist()
            st.success(
                f"✅ Import complete — added **{added}** new manufacturers. "
                f"Database now has **{len(st.session_state.results)}** total."
            )
            st.rerun()
        except Exception as e:
            st.error(f"Failed to import: {e}")

    st.divider()
    st.markdown("### ℹ️ Storage Info")
    st.markdown(f"- **File path:** `{persistence.data_dir_path()}`")
    st.markdown(f"- **Manufacturers stored:** {len(st.session_state.results)}")
    st.markdown(f"- **Search contexts tracked:** {len(st.session_state._batch_by_key)}")
    if st.session_state._last_saved:
        st.markdown(f"- **Last saved:** {st.session_state._last_saved}")
    if st.session_state._save_error:
        st.error(f"**Save error:** {st.session_state._save_error}")
        st.markdown(
            "The disk write is failing. Your data **is safe in this browser session** "
            "but will be lost if you close the tab. Use the **Export** button above "
            "to download a backup now."
        )
