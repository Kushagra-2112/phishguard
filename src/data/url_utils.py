"""Pull URLs out of raw email bodies and normalize them for the character encoder."""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse, unquote

URL_RE = re.compile(
    r"""(?xi)
    \b(
        (?:https?://|www\.)
        [^\s<>"'\)\]}]+
    )
    """
)

HREF_RE = re.compile(r"""(?i)href\s*=\s*["']([^"']+)["']""")
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t\r\f\v]+")


def extract_urls(body: str, limit: int = 10) -> list[str]:
    """Return de-duplicated URLs from an email body, href attributes first."""
    if not body:
        return []
    body = html.unescape(body)
    found: list[str] = []
    for match in HREF_RE.findall(body):
        found.append(match)
    for match in URL_RE.findall(TAG_RE.sub(" ", body)):
        found.append(match)

    seen, out = set(), []
    for raw in found:
        url = normalize_url(raw)
        if not url or url in seen:
            continue
        if not urlparse(url).netloc:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= limit:
            break
    return out


def normalize_url(url: str) -> str:
    """Light normalization only. Do NOT lowercase the path -- CANINE reads
    characters, and mixed case in a path is itself a weak phishing signal."""
    if not isinstance(url, str):
        return ""
    url = html.unescape(url).strip().strip('.,;:!"\'()[]{}<>')
    if not url:
        return ""
    if url.startswith("//"):
        url = "http:" + url
    elif not re.match(r"(?i)^[a-z][a-z0-9+.\-]*://", url):
        url = "http://" + url
    try:
        parsed = urlparse(url)
    except ValueError:
        # Malformed strings (e.g. stray brackets that look like IPv6 hosts)
        # show up in noisy real-world email/spam text -- skip them rather
        # than crash the whole pipeline over one bad row.
        return ""
    if not parsed.netloc:
        return ""
    netloc = parsed.netloc.lower()
    rest = url.split(parsed.netloc, 1)[-1]
    return f"{parsed.scheme.lower()}://{netloc}{rest}"


def pick_primary_url(urls: list[str]) -> str:
    """One URL per sample for the encoder. Prefer the longest -- attacker
    infrastructure tends to have deeper paths and more query junk than a
    plain unsubscribe link."""
    return max(urls, key=len) if urls else ""


def clean_email_text(body: str, max_chars: int = 20000) -> str:
    """Strip HTML down to readable text, keeping the raw link targets out
    (the URL stream handles those)."""
    if not isinstance(body, str):
        return ""
    body = html.unescape(body)
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
    body = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", body)
    body = TAG_RE.sub(" ", body)
    body = WS_RE.sub(" ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()[:max_chars]


def url_surface_features(url: str) -> dict:
    """Interpretable extras -- not fed to the model, shown in the UI so an
    analyst can sanity-check the verdict."""
    parsed = urlparse(url) if url else None
    host = parsed.netloc if parsed else ""
    return {
        "host": host,
        "length": len(url),
        "num_dots": host.count("."),
        "num_hyphens": host.count("-"),
        "has_ip_host": bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?", host)),
        "has_at_symbol": "@" in url,
        "is_punycode": "xn--" in host,
        "is_https": url.lower().startswith("https://"),
        "path_depth": len([p for p in (parsed.path.split("/") if parsed else []) if p]),
        "percent_encoded": unquote(url) != url,
    }