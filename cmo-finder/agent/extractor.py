"""
Claude-powered extraction of structured CMO/CDMO information.
Pre-populates phones/emails/address from scraper data so Claude
focuses only on fields that require language understanding.
"""

from __future__ import annotations  # makes all type hints lazy strings — fixes Python 3.9 compat

import json
import re
from typing import Optional

import anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM = (
    "You are an expert researcher specialising in Indian pharmaceutical and nutraceutical "
    "contract manufacturers. You extract structured data from web content about CMOs, CDMOs, "
    "and private-label / white-label manufacturers. "
    "Return ONLY valid JSON — no markdown fences, no prose."
)

PROMPT_TEMPLATE = """\
Extract information about an Indian CONTRACT / THIRD-PARTY manufacturer from the content below.

We are looking for companies that manufacture FOR other brands. This includes:
  • Pharmaceutical CMOs / CDMOs / TPM (third-party manufacturing / loan licence)
  • Nutraceutical / supplement CONTRACT or PRIVATE-LABEL / WHITE-LABEL manufacturers
  • Any Indian SME that makes products under another company's brand name

We do NOT want large branded pharma companies (Sun Pharma, Cipla, Lupin, Dr. Reddy's,
Aurobindo, Cadila/Zydus, Torrent, Alkem, Mankind, Glenmark, Abbott, Pfizer, Novartis,
GSK, Sanofi, IPCA, Wockhardt, Emcure, Biocon, Jubilant, Piramal, Intas, Laurus, Divi's, Granules).

ALREADY EXTRACTED (trust these, do not ignore them):
{pre_extracted}

Return a single JSON object (use null for missing fields):

{{
  "company_name":      "string — official company name",
  "plant_address":     "string — complete manufacturing plant/facility address",
  "city":              "string",
  "state":             "string — Indian state",
  "contact_person":    "string — name + designation of sales/BD contact if found",
  "phone":             "string — use pre-extracted if available",
  "email":             "string — use pre-extracted if available",
  "website":           "string",
  "dosage_forms":      ["list of dosage forms / product types they manufacture for others"],
  "certifications":    ["WHO-GMP", "FSSAI", "USFDA", "ISO 9001", "GMP", etc.],
  "capacity":          "string — production capacity if mentioned",
  "min_order":         "string — minimum order quantity if mentioned",
  "specialisation":    "string — e.g. softgels, protein supplements, herbal extracts",
  "description":       "string — 2–3 sentences on their contract/private-label capabilities",
  "is_tpm":            true
}}

Set "is_tpm" to TRUE if ANY of these apply:
  ✓ Offers third-party manufacturing / loan licence manufacturing
  ✓ Offers contract manufacturing for pharma or nutraceuticals
  ✓ Offers private-label or white-label manufacturing
  ✓ Makes products under other brands' names
  ✓ Is a CMO / CDMO / toll manufacturer

Set "is_tpm" to FALSE only if ALL of these apply:
  ✗ Is a large listed branded pharma company (see blacklist above)
  ✗ OR is purely a trading/distribution company with no manufacturing
  ✗ OR is a news article, job board, regulatory database, or empty directory page

If the company does BOTH its own branded products AND contract manufacturing for others, set is_tpm = TRUE.

Source URL: {url}
Dosage form context: {dosage_form}

--- CONTENT START ---
{content}
--- CONTENT END ---
"""

# Module-level error tracker (single-user app — no thread-safety needed)
_last_error: str = ""


def get_last_error() -> str:
    """Return the last Claude API error (empty string if last call succeeded)."""
    return _last_error


def _build_pre_extracted(scraped: dict) -> str:
    lines = []
    if scraped.get("phones"):
        lines.append(f"Phone numbers (from tel: links / regex): {', '.join(scraped['phones'])}")
    if scraped.get("emails"):
        lines.append(f"Email addresses (from mailto: links / regex): {', '.join(scraped['emails'])}")
    if scraped.get("gst"):
        lines.append(f"GST number (confirms Indian registration): {scraped['gst']}")
    if scraped.get("address_hints"):
        lines.append("Address fragments found in page text:")
        for hint in scraped["address_hints"]:
            lines.append(f"  • {hint}")
    return "\n".join(lines) if lines else "None extracted yet."


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# Rejection reason constants (returned instead of None so callers can count them)
REASON_NOT_TPM   = "not_tpm"
REASON_NO_NAME   = "no_name"
REASON_PARSE_ERR = "parse_error"
REASON_API_ERR   = "api_error"


def _call_claude(
    content: str,
    pre_extracted: str,
    url: str,
    dosage_form: str,
    client: anthropic.Anthropic,
) -> tuple[Optional[dict], str]:
    """
    Returns (result_dict, reason).
    result_dict is None on failure; reason is "" on success or one of REASON_* constants.
    """
    global _last_error
    prompt = PROMPT_TEMPLATE.format(
        pre_extracted=pre_extracted,
        url=url,
        dosage_form=dosage_form,
        content=content[:7000],
    )
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        _last_error = ""
        data = _parse_json(msg.content[0].text)
        if not data:
            return None, REASON_PARSE_ERR
        if not data.get("is_tpm", True):
            return None, REASON_NOT_TPM
        if not data.get("company_name"):
            return None, REASON_NO_NAME
        return data, ""
    except Exception as e:
        _last_error = str(e)
        return None, REASON_API_ERR


def _merge_scraped_into_result(result: dict, scraped: dict) -> dict:
    if scraped.get("phones") and not result.get("phone"):
        result["phone"] = scraped["phones"][0]
        result["all_phones"] = scraped["phones"]
    elif scraped.get("phones"):
        result["all_phones"] = scraped["phones"]
    if scraped.get("emails") and not result.get("email"):
        result["email"] = scraped["emails"][0]
    if scraped.get("gst") and not result.get("gst"):
        result["gst"] = scraped["gst"]
    if scraped.get("address_hints") and not result.get("plant_address"):
        result["plant_address"] = " | ".join(scraped["address_hints"][:2])
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def extract_from_rich(
    scraped: dict,
    url: str,
    dosage_form: str,
    client: anthropic.Anthropic,
) -> tuple[Optional[dict], str]:
    """Returns (result, reason). reason="" on success."""
    pre_extracted = _build_pre_extracted(scraped)
    full_text = scraped.get("text", "")
    if scraped.get("contact_page_text"):
        full_text += "\n\n[FROM CONTACT PAGE]\n" + scraped["contact_page_text"]

    result, reason = _call_claude(full_text, pre_extracted, url, dosage_form, client)
    if not result:
        return None, reason

    result = _merge_scraped_into_result(result, scraped)
    result["source_url"] = url
    return result, ""


def extract_from_snippet(
    title: str,
    snippet: str,
    url: str,
    dosage_form: str,
    client: anthropic.Anthropic,
) -> tuple[Optional[dict], str]:
    """Returns (result, reason). reason="" on success."""
    content = f"Page title: {title}\n\nSearch snippet: {snippet}"
    result, reason = _call_claude(content, "None extracted yet.", url, dosage_form, client)
    if result:
        result["source_url"] = url
    return result, reason


def extract_from_text(
    text: str,
    url: str,
    dosage_form: str,
    client: anthropic.Anthropic,
) -> tuple[Optional[dict], str]:
    """Backward-compatible plain-text extraction."""
    result, reason = _call_claude(text, "None extracted yet.", url, dosage_form, client)
    if result:
        result["source_url"] = url
    return result, reason
