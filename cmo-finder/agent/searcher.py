"""
DuckDuckGo-based search for Indian CMOs/CDMOs.
Targets ONLY third-party / loan-licence manufacturers — not big branded pharma.
Supports hub rotation for continuous searching.
"""
from __future__ import annotations  # allows | union syntax on Python 3.9

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

# Hub groups — rotated across search batches so each run finds new companies
HUB_GROUPS = [
    '"Baddi" OR "Nalagarh" OR "Kala Amb"',           # Himachal Pradesh
    '"Haridwar" OR "Roorkee" OR "Dehradun"',          # Uttarakhand
    '"Ahmedabad" OR "Surat" OR "Vadodara"',           # Gujarat
    '"Pune" OR "Nashik" OR "Aurangabad"',             # Maharashtra
    '"Hyderabad" OR "Vishakhapatnam"',                # Telangana / AP
    '"Sikkim" OR "Guwahati" OR "Kolkata"',            # North-East / East
    '"Delhi NCR" OR "Gurgaon" OR "Noida"',            # NCR
    '"Chandigarh" OR "Panchkula" OR "Mohali"',        # Punjab / Haryana
]

SKIP_DOMAINS = [
    "youtube.com", "facebook.com", "twitter.com", "instagram.com",
    "reddit.com", "quora.com", "amazon.com", "wikipedia.org",
    "linkedin.com", "naukri.com", "indeed.com", "glassdoor.com",
    "moneycontrol.com", "economictimes.com", "livemint.com",
    "thehindu.com", "ndtv.com", "businessstandard.com",
]

# Known big branded pharma to exclude from results
BIG_PHARMA_BLACKLIST = {
    "sun pharma", "sun pharmaceutical", "cipla", "dr reddy",
    "lupin", "aurobindo", "cadila", "zydus", "torrent pharma",
    "alkem", "mankind pharma", "glenmark", "abbott india",
    "pfizer", "novartis", "glaxosmithkline", "gsk", "sanofi",
    "ipca", "wockhardt", "emcure", "biocon", "jubilant",
    "divi's", "granules", "laurus labs", "natco pharma",
    "strides pharma", "piramal", "intas pharma",
}

PRIORITY_DOMAINS = [
    "site:indiamart.com",
    "site:exportersindia.com",
    "site:pharmahopers.com",
    "site:tradeindia.com",
    "site:pharmabiz.com",
    "site:pharmafranchiseehelp.com",
]


def is_big_pharma(title: str, url: str) -> bool:
    text = (title + " " + url).lower()
    return any(bp in text for bp in BIG_PHARMA_BLACKLIST)


def _build_queries(dosage_form: str, product_name: str, requirements: str, hub_group: str) -> list[str]:
    kws = DOSAGE_FORM_KEYWORDS.get(dosage_form, [dosage_form.lower()])
    primary = kws[0]
    product_term = f'"{product_name}"' if product_name else primary

    # Detect nutraceutical / supplement / herbal searches
    nutra_forms = {"Nutraceuticals / Supplements"}
    is_nutra = dosage_form in nutra_forms or any(
        w in primary.lower() for w in ["nutra", "supplement", "vitamin", "herbal", "ayurvedic"]
    )

    if is_nutra:
        # Nutraceutical industry uses "private label", "white label", "contract manufacturing"
        # NOT pharma-specific "loan licence" / "third party manufacturing"
        queries = [
            f'"private label" {product_term} manufacturer India contact address phone',
            f'"white label" {product_term} nutraceutical manufacturer India contact',
            f'"contract manufacturing" {product_term} supplement India {hub_group} contact',
            f'{product_term} "private label" nutraceutical India GMP FSSAI contact address',
            f'{product_term} nutraceutical manufacturer {hub_group} contact phone email',
            f'{product_term} "third party manufacturing" supplement India contact address',
        ]
    else:
        # Standard pharma — use Indian-specific TPM / loan licence terms
        queries = [
            f'"third party manufacturing" {product_term} India WHO-GMP contact address phone',
            f'"third party manufacturer" {product_term} India small company contact',
            f'"loan licence" OR "loan license" {product_term} India pharmaceutical contact address',
            f'"third party manufacturing" {product_term} {hub_group} contact phone address',
            f'{product_term} "third party" pharma manufacturer {PRIORITY_DOMAINS[0]} OR {PRIORITY_DOMAINS[1]}',
            f'{product_term} "contract manufacturing" "WHO-GMP" India SME contact {hub_group}',
        ]

    if requirements:
        queries.append(f'"contract manufacturing" {product_term} {requirements} India contact address')

    return queries


def search_cmos(
    dosage_forms: list[str],
    product_name: str = "",
    requirements: str = "",
    max_results: int = 20,
    batch: int = 0,
    already_seen: set[str] | None = None,   # URLs processed in previous batches
) -> list[dict]:
    """
    Search DuckDuckGo for Indian TPM/CMO/CDMO manufacturers.
    `batch`        — rotates hub group so each run targets a different pharma cluster.
    `already_seen` — URLs from prior batches; skipped so every batch finds fresh pages.
    """
    hub_group  = HUB_GROUPS[batch % len(HUB_GROUPS)]
    seen_urls  = set(already_seen or [])     # start with previously processed URLs
    all_results: list[dict] = []

    with DDGS() as ddgs:
        for form in dosage_forms:
            queries = _build_queries(form, product_name, requirements, hub_group)

            for query in queries[:6]:           # more queries per batch
                try:
                    hits = list(ddgs.text(query, region="in-en", max_results=8))  # more per query
                    for hit in hits:
                        url   = hit.get("href", "")
                        title = hit.get("title", "")
                        if not url or url in seen_urls:
                            continue
                        if any(d in url for d in SKIP_DOMAINS):
                            continue
                        if is_big_pharma(title, url):
                            continue
                        seen_urls.add(url)
                        all_results.append({
                            "url":         url,
                            "title":       title,
                            "snippet":     hit.get("body", ""),
                            "dosage_form": form,
                        })
                    time.sleep(0.5)
                except Exception:
                    continue

    return all_results[:max_results]


def search_company_contacts(company_name: str, ddgs: DDGS) -> list[dict]:
    """Targeted second-pass to find contact details for a known company."""
    queries = [
        f'"{company_name}" India pharmaceutical manufacturer address phone contact',
        f'"{company_name}" plant address GST contact email',
    ]
    results, seen = [], set()
    for q in queries:
        try:
            for hit in ddgs.text(q, region="in-en", max_results=4):
                url = hit.get("href", "")
                if url and url not in seen:
                    seen.add(url)
                    results.append({"url": url, "title": hit.get("title", ""), "snippet": hit.get("body", "")})
            time.sleep(0.3)
        except Exception:
            continue
    return results[:6]
