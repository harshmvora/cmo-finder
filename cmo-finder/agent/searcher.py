"""
DuckDuckGo-based search for Indian CMOs/CDMOs.
Generates targeted queries per dosage form and returns candidate URLs + snippets.
"""

import time
from ddgs import DDGS

DOSAGE_FORM_KEYWORDS: dict[str, list[str]] = {
    "Tablets":                          ["tablet", "oral solid dosage", "OSD"],
    "Capsules (Hard Gelatin)":          ["hard gelatin capsule", "HGC"],
    "Capsules (Soft Gelatin)":          ["soft gelatin capsule", "SGC", "softgel"],
    "Injectables (Liquid)":             ["injectable", "parenteral", "vial ampoule"],
    "Lyophilized Injectables":          ["lyophilized injectable", "freeze dried lyo"],
    "Liquids / Syrups":                 ["syrup oral liquid suspension"],
    "Topicals (Creams/Ointments)":      ["cream ointment topical gel"],
    "Nasal / Ophthalmic":               ["nasal spray eye drop ophthalmic"],
    "Aerosols / Inhalers":              ["aerosol inhaler MDI DPI"],
    "Transdermal Patches":              ["transdermal patch TDS"],
    "Suppositories":                    ["suppository"],
    "Nutraceuticals / Supplements":     ["nutraceutical dietary supplement vitamin"],
    "Hormones / High Potency":          ["hormone high potency oncology cytotoxic"],
    "APIs (Active Pharma Ingredients)": ["API active pharmaceutical ingredient"],
}

# Major Indian pharma manufacturing hubs
INDIA_PHARMA_HUBS = [
    "Baddi Himachal Pradesh",
    "Ahmedabad Gujarat",
    "Surat Gujarat",
    "Haridwar Uttarakhand",
    "Roorkee Uttarakhand",
    "Pune Maharashtra",
    "Nashik Maharashtra",
    "Hyderabad Telangana",
    "Sikkim",
]

SKIP_DOMAINS = [
    "youtube.com", "facebook.com", "twitter.com", "instagram.com",
    "reddit.com", "quora.com", "amazon.com", "wikipedia.org",
    "linkedin.com", "naukri.com", "indeed.com", "glassdoor.com",
]

# High-value Indian pharma B2B / directory domains to prioritise
PRIORITY_DOMAINS = [
    "site:indiamart.com",
    "site:pharmahopers.com",
    "site:pharmabiz.com",
    "site:exportersindia.com",
    "site:tradeindia.com",
    "site:pharmafranchiseehelp.com",
]


def _build_queries(dosage_form: str, product_name: str, requirements: str) -> list[str]:
    kws = DOSAGE_FORM_KEYWORDS.get(dosage_form, [dosage_form.lower()])
    primary = kws[0]

    # If product name given, use it as the primary search term
    product_term = f'"{product_name}"' if product_name else primary

    queries = [
        # Broad CMO search
        f'India CMO CDMO "contract manufacturer" {product_term} GMP plant address phone',
        # Third-party / loan licence terminology used in India
        f'India "third party manufacturer" OR "loan licence" {product_term} pharmaceutical contact',
        # Target B2B directories directly
        f'{product_term} contract pharma manufacturer India {PRIORITY_DOMAINS[0]} OR {PRIORITY_DOMAINS[1]}',
        # Hub-specific search
        f'pharmaceutical contract manufacturer {product_term} "Baddi" OR "Haridwar" OR "Ahmedabad" contact address',
        f'pharmaceutical contract manufacturer {product_term} "Pune" OR "Hyderabad" OR "Nashik" contact address',
        # WHO-GMP certified
        f'India "WHO-GMP" certified contract manufacturer {product_term} address phone email',
    ]

    if product_name:
        # Also search using dosage form + product together
        queries.append(f'India CMO {primary} "{product_name}" manufacturer contact address')

    if requirements:
        queries.append(
            f'India CMO {product_term} {requirements} pharmaceutical manufacturer address contact details'
        )

    return queries


def search_cmos(
    dosage_forms: list[str],
    product_name: str = "",
    requirements: str = "",
    max_results: int = 20,
) -> list[dict]:
    """
    Search DuckDuckGo for Indian CMOs/CDMOs.
    Returns list of dicts: url, title, snippet, dosage_form
    """
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    with DDGS() as ddgs:
        for form in dosage_forms:
            queries = _build_queries(form, product_name, requirements)

            for query in queries[:5]:          # up to 5 queries per dosage form
                try:
                    hits = list(ddgs.text(query, region="in-en", max_results=6))
                    for hit in hits:
                        url = hit.get("href", "")
                        if not url or url in seen_urls:
                            continue
                        if any(d in url for d in SKIP_DOMAINS):
                            continue
                        seen_urls.add(url)
                        all_results.append(
                            {
                                "url": url,
                                "title": hit.get("title", ""),
                                "snippet": hit.get("body", ""),
                                "dosage_form": form,
                            }
                        )
                    time.sleep(0.5)
                except Exception:
                    continue

    return all_results[:max_results]


def search_company_contacts(company_name: str, ddgs: DDGS) -> list[dict]:
    """
    Targeted second-pass: given a company name, search specifically for their
    contact details, plant address, phone number.
    """
    queries = [
        f'"{company_name}" India pharmaceutical manufacturer address phone contact',
        f'"{company_name}" plant address GST contact email',
    ]
    results = []
    seen = set()
    for q in queries:
        try:
            for hit in ddgs.text(q, region="in-en", max_results=4):
                url = hit.get("href", "")
                if url and url not in seen:
                    seen.add(url)
                    results.append({
                        "url": url,
                        "title": hit.get("title", ""),
                        "snippet": hit.get("body", ""),
                    })
            time.sleep(0.3)
        except Exception:
            continue
    return results[:6]
