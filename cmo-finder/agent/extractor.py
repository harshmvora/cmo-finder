"""
Claude-powered extraction of structured CMO/CDMO information.
Pre-populates phones/emails/address from scraper data so Claude
focuses only on fields that require language understanding.
"""

import json
import re
from typing import Optional

import anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM = (
    "You are an expert researcher specialising in Indian pharmaceutical contract manufacturers. "
    "You extract structured data from web content about CMOs and CDMOs. "
    "Return ONLY valid JSON — no markdown fences, no prose."
)

PROMPT_TEMPLATE = """\
Extract information about a pharmaceutical contract manufacturer (CMO/CDMO) from the content below.

ALREADY EXTRACTED (trust these, do not ignore them):
{pre_extracted}

Now extract the remaining fields from the content.
Return a single JSON object (use null for missing fields):

{{
  "company_name":     "string — official company name",
  "plant_address":    "string — complete manufacturing plant/facility address (not office address)",
  "city":             "string",
  "state":            "string — Indian state",
  "contact_person":   "string — name + designation of sales/BD contact if found",
  "phone":            "string — use pre-extracted if available",
  "email":            "string — use pre-extracted if available",
  "website":          "string",
  "dosage_forms":     ["list of dosage forms they manufacture e.g. tablets, capsules, injectables"],
  "certifications":   ["WHO-GMP", "USFDA", "EU GMP", "ISO 9001", etc.],
  "capacity":         "string — production capacity if mentioned",
  "min_order":        "string — minimum order quantity if mentioned",
  "specialisation":   "string — any niche or speciality (e.g. oncology, hormones, controlled release)",
  "description":      "string — 2–3 sentence summary of their CMO/CDMO capabilities",
  "is_manufacturer":  true
}}

Set "is_manufacturer" to false ONLY if this is clearly not a pharmaceutical manufacturer
(e.g. pure news article, job listing, generic SEO page with no company identity).
If there is ANY sign of a real company offering manufacturing services, set it to true.

Source URL: {url}
Dosage form context: {dosage_form}

--- CONTENT START ---
{content}
--- CONTENT END ---
"""


def _build_pre_extracted(scraped: dict) -> str:
    """Format pre-extracted structured data as a clear string for Claude."""
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
    # Remove markdown code fences if present
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


def _call_claude(
    content: str,
    pre_extracted: str,
    url: str,
    dosage_form: str,
    client: anthropic.Anthropic,
) -> Optional[dict]:
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
        data = _parse_json(msg.content[0].text)
        if not data:
            return None
        if not data.get("is_manufacturer", True):
            return None
        if not data.get("company_name"):
            return None
        return data
    except Exception:
        return None


def _merge_scraped_into_result(result: dict, scraped: dict) -> dict:
    """
    Overwrite result fields with hard-extracted data from scraper when Claude missed them.
    The scraper's tel:/mailto: links are more reliable than Claude's text-parsing.
    """
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
) -> Optional[dict]:
    """
    Full extraction from scrape_rich() output.
    Uses pre-extracted contacts as ground truth, Claude fills the rest.
    """
    pre_extracted = _build_pre_extracted(scraped)

    # Combine main text + contact page text
    full_text = scraped.get("text", "")
    if scraped.get("contact_page_text"):
        full_text += "\n\n[FROM CONTACT PAGE]\n" + scraped["contact_page_text"]

    result = _call_claude(full_text, pre_extracted, url, dosage_form, client)
    if not result:
        return None

    result = _merge_scraped_into_result(result, scraped)
    result["source_url"] = url
    return result


def extract_from_snippet(
    title: str,
    snippet: str,
    url: str,
    dosage_form: str,
    client: anthropic.Anthropic,
) -> Optional[dict]:
    """Quick extraction from a search-result snippet (no page fetch)."""
    content = f"Page title: {title}\n\nSearch snippet: {snippet}"
    pre = "None extracted yet."
    result = _call_claude(content, pre, url, dosage_form, client)
    if result:
        result["source_url"] = url
    return result


def extract_from_text(
    text: str,
    url: str,
    dosage_form: str,
    client: anthropic.Anthropic,
) -> Optional[dict]:
    """Backward-compatible plain-text extraction (no scraper dict)."""
    result = _call_claude(text, "None extracted yet.", url, dosage_form, client)
    if result:
        result["source_url"] = url
    return result
