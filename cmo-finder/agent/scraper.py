"""
Scraper with:
- tel:/mailto: href extraction (exact contacts, no NLP needed)
- Regex phone/email/GST/address extraction from text
- Contact page hunting (tries /contact-us, /about etc. automatically)
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ── HTTP headers ──────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/",
}

SKIP_DOMAINS = [
    "youtube.com", "facebook.com", "twitter.com", "instagram.com",
    "reddit.com", "quora.com", "amazon.com", "flipkart.com",
    "wikipedia.org", "slideshare.net", "scribd.com", "linkedin.com",
]

NOISE_TAGS = [
    "script", "style", "nav", "footer", "aside",
    "form", "iframe", "noscript", "svg", "head",
]

# Contact page slug patterns (tried in order)
CONTACT_SLUGS = [
    "/contact-us", "/contact", "/contactus",
    "/reach-us", "/get-in-touch", "/enquiry", "/enquire",
    "/about-us", "/about", "/location", "/our-location",
]

# ── Regex patterns ────────────────────────────────────────────────────────────
# Indian mobile / landline
PHONE_RE = re.compile(
    r"""(?<!\d)                                # not preceded by digit
    (?:
      (?:\+91|91)[\s\-.]?[6-9]\d{9}           # +91 / 91 + 10-digit mobile
    | (?:0\d{2,4})[\s\-.]?\d{6,8}             # STD code + landline
    | [6-9]\d{9}                               # plain 10-digit mobile
    )
    (?!\d)                                     # not followed by digit
    """,
    re.VERBOSE,
)

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# GST number (uniquely ties to a registered address in India)
GST_RE = re.compile(
    r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b"
)

# Sentences likely to contain an address
ADDRESS_KEYWORDS = [
    "plot no", "plot no.", "survey no", "khasra no", "gat no",
    "industrial area", "industrial estate", "ind. area",
    "phase", "sector", "nagar", "road", "street", "marg",
    "village", "dist.", "district", "taluka", "tehsil",
    "pin", "pincode", "pin code", "zip",
    "baddi", "solan", "chandigarh", "ahmedabad", "surat", "vadodara",
    "pune", "mumbai", "nashik", "hyderabad", "haridwar", "roorkee",
    "dehradun", "himachal", "uttarakhand", "gujarat", "maharashtra",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _should_skip(url: str) -> bool:
    return any(d in url for d in SKIP_DOMAINS)


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main", re.I))
        or soup.body
        or soup
    )
    text = main.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _extract_hrefs(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    """Pull phones and emails directly from tel:/mailto: href attributes."""
    phones, emails = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("tel:"):
            num = re.sub(r"[^\d+]", "", href[4:])
            if num:
                phones.append(num)
        elif href.startswith("mailto:"):
            email = href[7:].split("?")[0].strip()
            if "@" in email:
                emails.append(email)
    return phones, emails


def _extract_from_text(text: str) -> dict:
    """Regex-extract phones, emails, GST, address sentences from plain text."""
    phones = list(dict.fromkeys(re.findall(PHONE_RE, text)))
    emails = list(dict.fromkeys(re.findall(EMAIL_RE, text)))
    gst    = (re.findall(GST_RE, text) or [None])[0]

    # Address hints: sentences / fragments containing address keywords
    hints = []
    for chunk in re.split(r"[|\n]", text):
        chunk = chunk.strip()
        if any(kw in chunk.lower() for kw in ADDRESS_KEYWORDS):
            if 15 < len(chunk) < 350:
                hints.append(chunk)

    return {
        "phones": phones[:6],
        "emails": emails[:4],
        "gst": gst,
        "address_hints": list(dict.fromkeys(hints))[:5],
    }


def _find_contact_urls(base_url: str, soup: BeautifulSoup) -> list[str]:
    """Return up to 4 candidate contact/about page URLs for this domain."""
    parsed = urlparse(base_url)
    base   = f"{parsed.scheme}://{parsed.netloc}"
    urls   = []

    # 1. Internal links whose text or href look like contact/about pages
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        text = a.get_text().lower()
        if any(kw in href or kw in text for kw in ["contact", "reach", "enquir", "location", "about"]):
            full = urljoin(base_url, a["href"])
            if urlparse(full).netloc == parsed.netloc and full not in urls:
                urls.append(full)

    # 2. Try known slug patterns on the same domain
    for slug in CONTACT_SLUGS:
        candidate = base + slug
        if candidate not in urls:
            urls.append(candidate)

    return urls[:5]


# ── Public API ────────────────────────────────────────────────────────────────
def _fetch(url: str, timeout: int) -> tuple[requests.Response | None, BeautifulSoup | None]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return None, None
        if "text/html" not in r.headers.get("content-type", ""):
            return None, None
        return r, BeautifulSoup(r.text, "lxml")
    except Exception:
        return None, None


def scrape_rich(url: str, timeout: int = 12, follow_contact: bool = True) -> dict | None:
    """
    Scrape a URL and return a rich dict:
    {
      text          : str   — clean plain text (up to 8000 chars)
      phones        : list  — from tel: hrefs + regex
      emails        : list  — from mailto: hrefs + regex
      gst           : str | None
      address_hints : list  — text chunks containing address keywords
      contact_page_text : str | None  — text from /contact-us page
    }
    Returns None if the page cannot be fetched.
    """
    if _should_skip(url):
        return None

    resp, soup = _fetch(url, timeout)
    if not soup:
        return None

    text = _clean_text(soup)
    href_phones, href_emails = _extract_hrefs(soup)
    regex_data = _extract_from_text(text)

    # Merge href (exact) + regex (broader), dedup
    phones = list(dict.fromkeys(href_phones + regex_data["phones"]))
    emails = list(dict.fromkeys(href_emails + regex_data["emails"]))
    # Filter out junk emails (images, fonts, etc.)
    emails = [e for e in emails if not re.search(r"\.(png|jpg|gif|woff|css|js)$", e, re.I)]

    result = {
        "text": text[:8000],
        "phones": phones[:5],
        "emails": emails[:3],
        "gst": regex_data["gst"],
        "address_hints": regex_data["address_hints"],
        "contact_page_text": None,
    }

    # Hunt for a contact/about page if main page is missing contacts
    if follow_contact and not (phones and emails):
        contact_urls = _find_contact_urls(url, soup)
        for curl in contact_urls[:4]:
            if curl == url:
                continue
            cr, csoup = _fetch(curl, timeout=8)
            if not csoup:
                continue
            ctext = _clean_text(csoup)
            cp, ce = _extract_hrefs(csoup)
            cr_data = _extract_from_text(ctext)

            new_phones = list(dict.fromkeys(cp + cr_data["phones"]))
            new_emails = list(dict.fromkeys(ce + cr_data["emails"]))
            new_emails = [e for e in new_emails if not re.search(r"\.(png|jpg|gif|woff|css|js)$", e, re.I)]

            if new_phones or new_emails or cr_data["address_hints"]:
                phones = list(dict.fromkeys(phones + new_phones))
                emails = list(dict.fromkeys(emails + new_emails))
                result["phones"] = phones[:5]
                result["emails"] = emails[:3]
                result["address_hints"] = list(dict.fromkeys(
                    result["address_hints"] + cr_data["address_hints"]
                ))[:5]
                result["contact_page_text"] = ctext[:3000]
                if not result["gst"] and cr_data["gst"]:
                    result["gst"] = cr_data["gst"]
                break  # one good contact page is enough

    return result


# Kept for backward compatibility
def scrape_url(url: str, timeout: int = 12) -> str | None:
    data = scrape_rich(url, timeout=timeout, follow_contact=False)
    return data["text"] if data else None
