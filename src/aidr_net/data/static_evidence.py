"""M1: local-only DOM/URL extraction. Never executes HTML or fetches URLs."""
import ipaddress
import re
from urllib.parse import urljoin, urlsplit

import tldextract
from bs4 import BeautifulSoup

# Bundled PSL, including private suffixes; no network/cache initialization needed.
_PSL = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True, cache_dir=None)
INTENT_PATTERN = re.compile(r"login|log.?in|sign.?in|password|credential|pay|checkout|download|登录|密码|支付|下载", re.I)
TAGS = {name: i + 1 for i, name in enumerate(("title", "h1", "h2", "h3", "p", "span", "a", "img", "input", "button", "form", "label", "select", "textarea"))}


def normalize_url(value):
    value = str(value or "").strip()
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", value) and not re.match(r"^[^/:]+:\d+(?:/|$)", value):
        return value
    if value and "://" not in value and not value.startswith(("//", "/", "#")):
        value = "https://" + value
    if value.startswith("//"):
        value = "https:" + value
    return value


def host_and_domain(value):
    try:
        parsed = urlsplit(normalize_url(value))
        if parsed.scheme not in {"http", "https"}:
            return "", ""
        host = (parsed.hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
        if not host:
            return "", ""
        try:
            ipaddress.ip_address(host)
            return host, host
        except ValueError:
            result = _PSL(host)
            return host, result.top_domain_under_public_suffix or host
    except (ValueError, UnicodeError):
        return "", ""


def extract_evidence(url, html, max_nodes=128, max_targets=8):
    url = normalize_url(url)
    host, domain = host_and_domain(url)
    soup = BeautifulSoup(html or "", "html.parser")
    base = soup.find("base", href=True)
    base_url = urljoin(url, base["href"]) if base else url
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    nodes = []
    for element in soup.find_all(list(TAGS)):
        # Direct text avoids repeating the entire page in every parent node.
        direct = " ".join(str(t).strip() for t in element.find_all(string=True, recursive=False) if str(t).strip())
        attrs = " ".join(f"{k}={str(element.get(k))[:256]}" for k in ("type", "name", "id", "placeholder", "aria-label", "alt", "action", "autocomplete", "href", "src", "method") if element.has_attr(k))
        text = (direct[:512] + " " + attrs).strip()
        if not text:
            continue
        appearance = 0.8 if element.name in {"title", "h1", "h2", "img"} else 0.0
        intent = 0.9 if element.name in {"input", "button", "form", "select", "textarea"} else 0.0
        if INTENT_PATTERN.search(text):
            intent = max(intent, 0.7)
        nodes.append({"text": text, "tag": TAGS[element.name], "depth": min(len(list(element.parents)), 31),
                      "role": [appearance, intent]})
    # Preserve original DOM order, but reserve truncation budget for role-bearing nodes.
    if len(nodes) > max_nodes:
        ranked = sorted(range(len(nodes)), key=lambda i: (-max(nodes[i]["role"]), i))[:max_nodes]
        nodes = [nodes[i] for i in sorted(ranked)]
    targets = []
    seen = set()
    forms = soup.find_all("form")
    actions = [(form.get("action") or "", "form.action") for form in forms]
    actions += [(e.get("formaction"), "formaction") for e in soup.find_all(["button", "input"], formaction=True)
                if e.find_parent("form") is not None or (e.get("form") and soup.find("form", id=e["form"]) is not None)]
    for action, source in actions:
        # Empty action submits to document URL, not <base> URL.
        target = urljoin(base_url, action) if action else url
        if target in seen:
            continue
        seen.add(target)
        target_host, target_domain = host_and_domain(target)
        targets.append({"url": target, "source": source, "valid": bool(target_host),
                        "facts": [float(bool(host) and host == target_host),
                                  float(bool(domain) and domain == target_domain), float(bool(target_host))]})
    return {"url": url, "host": host, "domain": domain, "nodes": nodes,
            "targets": targets[:max_targets], "targets_truncated": len(targets) > max_targets,
            "html_available": bool(str(html or "").strip())}
