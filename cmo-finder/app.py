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
    """Read API key from st.secrets (Streamlit Cloud) or local config file."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return _load_config().get("api_key", "")

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
    color: white; padding: 22px 32px; border-radius: 14px; margin-bottom: 28px;
}
.header-banner h1 { margin: 0; font-size: 26px; letter-spacing: -0.5px; }
.header-banner p  { margin: 5px 0 0; opacity: 0.82; font-size: 14px; }
.pill {
    display: inline-block; padding: 2px 11px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin: 2px 3px 2px 0;
}
.pill-green { background: #dcfce7; color: #166534; }
.pill-blue  { background: #dbeafe; color: #1e40af; }
.section-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: #64748b; margin: 10px 0 3px;
}
.stProgress > div > div { background-color: #0d7a4e !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results: list[dict] = []
if "pending_lookup" not in st.session_state:
    st.session_state.pending_lookup = None
if "search_batch" not in st.session_state:
    st.session_state.search_batch = 0        # increments each auto-continue run
if "auto_continue" not in st.session_state:
    st.session_state.auto_continue = False
if "auto_params" not in st.session_state:
    st.session_state.auto_params = {}        # stores last search params for auto-continue
if "seen_urls" not in st.session_state:
    st.session_state.seen_urls: set[str] = set()   # URLs already processed (cross-batch)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🏭 CMO &amp; CDMO Finder — India</h1>
  <p>AI-powered search · Scrapes plant address &amp; contacts from multiple sources · Deep contact-page hunting</p>
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
    # Save whenever the key changes (local dev only — on Cloud it comes from st.secrets)
    if api_key and api_key != _config.get("api_key", ""):
        _save_config({**_config, "api_key": api_key})

    st.divider()
    st.markdown("### 🔎 Search Settings")
    max_results = st.slider("New URLs per batch", 5, 50, 25, 5,
        help="Each batch fetches this many fresh URLs (already-seen URLs are skipped automatically)")
    deep_scrape = st.checkbox(
        "Deep scrape + contact-page hunting",
        value=True,
        help="Follows /contact-us pages and extracts phones/emails via regex. Slower but much higher quality.",
    )

    st.divider()
    st.markdown("### 🔬 Filter Results")
    filter_contact = st.checkbox("Only show results with phone / email")
    filter_address = st.checkbox("Only show results with plant address")
    cert_filter = st.multiselect(
        "Must have certification",
        ["WHO-GMP", "USFDA", "EU GMP", "ISO 9001", "CGMP", "ISO 14001"],
    )

    st.divider()
    st.markdown("### 🔄 Auto-Continue")
    auto_continue = st.toggle(
        "Keep searching non-stop",
        value=st.session_state.auto_continue,
        help="Automatically runs a new search batch after each one completes, rotating through different Indian pharma hubs.",
    )
    st.session_state.auto_continue = auto_continue
    if auto_continue:
        hub_idx = st.session_state.search_batch % len(HUB_GROUPS)
        st.caption(f"Batch #{st.session_state.search_batch + 1} · Hub: {HUB_GROUPS[hub_idx]}")

    st.divider()
    if st.button("🗑️ Clear All Results", use_container_width=True):
        st.session_state.results = []
        st.session_state.pending_lookup = None
        st.session_state.search_batch = 0
        st.session_state.auto_continue = False
        st.session_state.seen_urls = set()
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Claude Haiku · DuckDuckGo")
    st.caption("Regex phone/email · Contact-page hunting")

# ── Input form ────────────────────────────────────────────────────────────────
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
        placeholder="e.g. Metformin 500mg, Amoxicillin, Pantoprazole...",
        label_visibility="collapsed",
        help="Searches specifically for manufacturers of this product. Overrides generic dosage-form keywords.",
    )
with col_req:
    st.markdown("#### 📋 Other Requirements *(optional)*")
    requirements = st.text_area(
        "requirements",
        placeholder="e.g. WHO-GMP, min 10M units/month, Gujarat or Maharashtra...",
        height=68, label_visibility="collapsed",
    )

# ── Search button ─────────────────────────────────────────────────────────────
_, btn_col, stop_col, _ = st.columns([1, 2, 1, 1])
with btn_col:
    search_clicked = st.button(
        "🔍  Find Manufacturers", use_container_width=True, type="primary",
        disabled=(not api_key) or (not dosage_forms),
    )
with stop_col:
    if st.button("⏹ Stop", use_container_width=True, disabled=not st.session_state.auto_continue):
        st.session_state.auto_continue = False
        st.rerun()

if not api_key:
    st.info("👈  Enter your Claude API key in the sidebar to get started.")

# Auto-continue: treat it like a search click if params are stored
_auto_trigger = (
    st.session_state.auto_continue
    and st.session_state.auto_params
    and not search_clicked
)

# ── Search execution ──────────────────────────────────────────────────────────
if (search_clicked or _auto_trigger) and api_key and dosage_forms:
    client = anthropic.Anthropic(api_key=api_key)

    # Save params for auto-continue to reuse
    if search_clicked:
        st.session_state.auto_params = {
            "dosage_forms": dosage_forms,
            "product_name": product_name,
            "requirements": requirements,
            "max_results":  max_results,
        }
        st.session_state.search_batch = 0
        st.session_state.seen_urls = set()   # reset seen URLs on a fresh search

    params = st.session_state.auto_params
    batch  = st.session_state.search_batch

    progress_bar = st.progress(0)
    status = st.empty()
    new_results: list[dict] = []

    hub_idx = batch % len(HUB_GROUPS)
    status.markdown(
        f"🔎 **Batch #{batch + 1}** · Searching hub: *{HUB_GROUPS[hub_idx]}*…"
    )
    progress_bar.progress(8)

    search_hits = search_cmos(
        params["dosage_forms"], params["product_name"],
        params["requirements"], params["max_results"],
        batch=batch,
        already_seen=st.session_state.seen_urls,
    )

    # Record all URLs we just tried so future batches skip them
    for h in search_hits:
        st.session_state.seen_urls.add(h["url"])

    if not search_hits:
        if st.session_state.auto_continue:
            # No new URLs this batch — advance and try next hub
            st.session_state.search_batch = batch + 1
            import time as _time; _time.sleep(1)
            st.rerun()
        else:
            st.warning("No new results found. Try different dosage forms or requirements.")
            st.stop()

    status.markdown(f"📄 **Found {len(search_hits)} candidate pages.** Extracting with Claude…")
    progress_bar.progress(18)

    total = len(search_hits)
    for i, hit in enumerate(search_hits):
        progress_bar.progress(18 + int((i / total) * 78))
        status.markdown(f"🤖 **{i+1}/{total}** — *{hit['title'][:70]}*")

        extracted = None

        dosage_context = f"{hit['dosage_form']}" + (f" — {params['product_name']}" if params.get('product_name') else "")

        if deep_scrape:
            scraped = scrape_rich(hit["url"], follow_contact=True)
            if scraped:
                extracted = extract_from_rich(scraped, hit["url"], dosage_context, client)

        if not extracted and hit.get("snippet"):
            extracted = extract_from_snippet(
                hit["title"], hit["snippet"], hit["url"], dosage_context, client
            )

        if extracted and extracted.get("company_name"):
            extracted["searched_dosage_form"] = hit["dosage_form"]
            extracted["found_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_results.append(extracted)

    progress_bar.progress(100)
    status.markdown(f"✅ **Done!** Identified **{len(new_results)}** manufacturers.")

    # Deduplicate and add
    existing = {r.get("company_name", "").strip().lower() for r in st.session_state.results}
    added = 0
    for r in new_results:
        key = r.get("company_name", "").strip().lower()
        if key and key not in existing:
            st.session_state.results.append(r)
            existing.add(key)
            added += 1

    if added:
        st.success(f"Added {added} new manufacturer(s) to results (Batch #{batch + 1}).")

    # Advance batch counter and auto-rerun if enabled
    st.session_state.search_batch = batch + 1
    if st.session_state.auto_continue:
        import time as _time
        _time.sleep(2)   # brief pause between batches
        st.rerun()

# ── Results section ───────────────────────────────────────────────────────────
if st.session_state.results:

    # ── Handle pending contact lookup (triggered by per-card button) ──────────
    if st.session_state.pending_lookup is not None and api_key:
        idx = st.session_state.pending_lookup
        st.session_state.pending_lookup = None
        r = st.session_state.results[idx]

        with st.spinner(f"Searching for contact details: **{r.get('company_name', '')}**…"):
            try:
                with DDGS() as ddgs:
                    hits = search_company_contacts(r.get("company_name", ""), ddgs)
                for hit in hits:
                    scraped = scrape_rich(hit["url"], follow_contact=True)
                    if not scraped:
                        continue
                    if scraped.get("phones") and not r.get("phone"):
                        r["phone"] = scraped["phones"][0]
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
                        break   # one good hit is enough

                st.session_state.results[idx] = r
                st.success("Contact details updated!")
            except Exception as e:
                st.warning(f"Contact lookup failed: {e}")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    display = st.session_state.results.copy()
    if filter_contact:
        display = [r for r in display if r.get("phone") or r.get("email")]
    if filter_address:
        display = [r for r in display if r.get("plant_address") or r.get("city")]
    if cert_filter:
        def _has_certs(r):
            s = " ".join(r.get("certifications") or []).upper()
            return all(c.upper() in s for c in cert_filter)
        display = [r for r in display if _has_certs(r)]

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Found", len(st.session_state.results))
    m2.metric("With Contact", sum(1 for r in st.session_state.results if r.get("phone") or r.get("email")))
    m3.metric("With Address", sum(1 for r in st.session_state.results if r.get("plant_address")))
    m4.metric("Showing", len(display))

    # ── Export ────────────────────────────────────────────────────────────────
    df_all = pd.DataFrame(st.session_state.results)
    exp1, exp2, _ = st.columns([1, 1, 3])
    with exp1:
        st.download_button(
            "📥 Export CSV",
            df_all.to_csv(index=False).encode("utf-8"),
            f"cmo_india_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv", use_container_width=True,
        )
    with exp2:
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
    view = st.radio("View mode", ["Cards", "Table"], horizontal=True)

    if not display:
        st.warning("No results match the current filters.")

    elif view == "Table":
        df_disp = pd.DataFrame(display)
        TABLE_COLS = ["company_name", "city", "state", "phone", "email",
                      "certifications", "dosage_forms", "website", "source_url"]
        avail = [c for c in TABLE_COLS if c in df_disp.columns]
        tbl = df_disp[avail].copy()
        tbl.columns = [c.replace("_", " ").title() for c in avail]
        for col in ["Certifications", "Dosage Forms"]:
            if col in tbl.columns:
                tbl[col] = tbl[col].apply(lambda v: ", ".join(v) if isinstance(v, list) else (v or ""))
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    else:
        # ── Card view ─────────────────────────────────────────────────────────
        for card_i, r in enumerate(display):
            # Find the true index in session_state.results for contact lookup
            true_idx = next(
                (j for j, x in enumerate(st.session_state.results)
                 if x.get("company_name") == r.get("company_name")),
                card_i,
            )

            name = r.get("company_name", "Unknown")
            city  = r.get("city", "")
            state = r.get("state", "")
            location_short = ", ".join(filter(None, [city, state])) or "India"

            # Completeness score
            filled = sum(bool(r.get(f)) for f in ["phone", "email", "plant_address", "certifications", "dosage_forms"])
            score_color = "#22c55e" if filled >= 4 else "#f59e0b" if filled >= 2 else "#ef4444"
            score_label = f"{'●' * filled}{'○' * (5 - filled)}"

            with st.expander(
                f"🏭  {name}  —  {location_short}  |  {score_label}",
                expanded=False,
            ):
                left, right = st.columns(2)

                with left:
                    st.markdown('<div class="section-label">📍 Plant / Facility Address</div>', unsafe_allow_html=True)
                    addr = r.get("plant_address") or location_short or "—"
                    st.write(addr)
                    if r.get("gst"):
                        st.caption(f"GST: {r['gst']}")

                    st.markdown('<div class="section-label">📞 Contact Details</div>', unsafe_allow_html=True)
                    lines = []
                    if r.get("contact_person"):
                        lines.append(f"👤 {r['contact_person']}")
                    if r.get("phone"):
                        lines.append(f"📱 {r['phone']}")
                    # Show additional phones if found
                    for extra in (r.get("all_phones") or [])[1:3]:
                        lines.append(f"📱 {extra}")
                    if r.get("email"):
                        lines.append(f"📧 {r['email']}")
                    if r.get("website"):
                        lines.append(f"🌐 [{r['website']}]({r['website']})")
                    st.markdown("\n\n".join(lines) if lines else "—")

                    # Contact lookup button
                    missing = not r.get("phone") and not r.get("email")
                    btn_label = "🔍 Find Contacts" if missing else "🔄 Refresh Contacts"
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

                    if r.get("specialisation"):
                        st.markdown('<div class="section-label">🔬 Specialisation</div>', unsafe_allow_html=True)
                        st.write(r["specialisation"])

                if r.get("description"):
                    st.markdown('<div class="section-label">📝 About</div>', unsafe_allow_html=True)
                    st.write(r["description"])

                src = r.get("source_url", "")
                if src:
                    st.markdown(f'<a href="{src}" target="_blank">🔗 View Source</a>', unsafe_allow_html=True)

else:
    st.markdown("""
<div style="text-align:center; padding: 64px 20px; color:#94a3b8;">
  <div style="font-size:64px">🏭</div>
  <h3 style="color:#64748b; margin-top:12px;">No results yet</h3>
  <p>Select dosage forms above and click <strong>Find Manufacturers</strong>.</p>
  <p style="font-size:13px; margin-top:8px;">
    The agent hunts across DuckDuckGo, IndiaMART, PharmaBiz and company websites,
    follows /contact-us pages, and extracts phone numbers and plant addresses automatically.
  </p>
</div>
""", unsafe_allow_html=True)
