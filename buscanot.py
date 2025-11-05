import asyncio
import aiohttp
import time
import json
import re
import random
import unicodedata
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from functools import lru_cache

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup, SoupStrainer
from aiohttp import ClientTimeout
from email.utils import parsedate_to_datetime
import urllib.robotparser as rp

# =========================
# Configuración general
# =========================
st.set_page_config(page_title="BuscaNot", layout="wide")
st.title("🌍 BUSCADOR DE NOTICIAS")

DB_PATH = Path("media_db.json")

BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9",
    "Connection": "keep-alive",
}

DEFAULT_TIMEOUT = ClientTimeout(total=20, connect=10, sock_connect=10, sock_read=15)
ALLOWED_SCHEMES = {"http", "https"}
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid", "mc_cid", "mc_eid"}

# =========================
# Patrones comunes para descubrir hemeroteca (si no se define)
# =========================
COMMON_ARCHIVE_PATTERNS = [
    "/hemeroteca/{yyyy}-{mm}-{dd}/",      
    "/hemeroteca/{yyyy}/{mm}/{dd}/",     
    "/archivo/{yyyy}-{mm}-{dd}/",
    "/archivo/{yyyy}/{mm}/{dd}/",
    "/{yyyy}/{mm}/{dd}/",                
]

# =========================
# Persistencia BD
# =========================
def save_db(db: Dict[str, List[Dict[str, Any]]]) -> None:
    try:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"❌ No se pudo guardar media_db.json: {e}")

def load_db() -> Dict[str, List[Dict[str, Any]]]:
    if DB_PATH.exists():
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                st.error("⚠️ El fichero media_db.json no contiene un objeto JSON de nivel raíz (debería ser { ... }).")
                return {}
        except Exception as e:
            st.error(f"⚠️ No se pudo leer media_db.json: {e}")
            return {}
    else:
        st.error("⚠️ No se encontró el fichero media_db.json en el directorio de la app. Créalo y vuelve a ejecutar.")
        return {}

# =========================
# Cachés
# =========================
@st.cache_resource
def get_html_cache() -> Dict[str, Tuple[float, str]]:
    return {}

@st.cache_resource
def get_neg_cache() -> Dict[str, float]:
    return {}

@st.cache_resource
def get_translation_cache() -> Dict[Tuple[str, str], str]:
    """(texto, lang_destino) -> traducción"""
    return {}

def cache_get(url: str, ttl_sec: int) -> Optional[str]:
    cache = get_html_cache()
    item = cache.get(url)
    if not item:
        return None
    ts, html = item
    if (time.time() - ts) <= ttl_sec:
        return html
    cache.pop(url, None)
    return None

def cache_put(url: str, html: str) -> None:
    cache = get_html_cache()
    cache[url] = (time.time(), html)

def neg_cache_hit(url: str, neg_ttl_sec: int) -> bool:
    neg = get_neg_cache()
    ts = neg.get(url)
    if not ts:
        return False
    if (time.time() - ts) <= neg_ttl_sec:
        return True
    neg.pop(url, None)
    return False

def neg_cache_put(url: str) -> None:
    neg = get_neg_cache()
    neg[url] = time.time()

# =========================
# Utilidades
# =========================
def build_headers(country: str) -> Dict[str, str]:
    h = dict(BASE_HEADERS)
    if country in {"España", "Andorra"}:
        h["Accept-Language"] = "es-ES,es;q=0.9,*;q=0.1"
    elif country in {"Portugal"}:
        h["Accept-Language"] = "pt-PT,pt;q=0.9,*;q=0.1"
    elif country in {"Francia"}:
        h["Accept-Language"] = "fr-FR,fr;q=0.9,*;q=0.1"
    elif country in {"Alemania"}:
        h["Accept-Language"] = "de-DE,de;q=0.9,*;q=0.1"
    else:
        h["Accept-Language"] = "*;q=0.1"
    return h

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def absolutize(link: str, base_url: Optional[str]) -> str:
    if not link:
        return ""
    full = urljoin(base_url or "", link)
    parsed = urlparse(full)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return ""
    qs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k not in TRACKING_PARAMS]
    cleaned = parsed._replace(query=urlencode(qs))
    return urlunparse(cleaned)

def extract_time_candidate(el) -> Optional[str]:
    t = el.find("time")
    if t and (t.get("datetime") or t.get_text(strip=True)):
        return t.get("datetime") or t.get_text(strip=True)
    for attr in ("data-published", "data-date", "data-time", "aria-label"):
        v = el.get(attr)
        if v:
            return v
    return None

def is_latin_lang(lang: str) -> bool:
    return (lang or "").lower() in {"es","fr","pt","de","en","it","ca","gl"}

def split_terms(terms: str) -> List[Tuple[str, bool]]:
    """
    Devuelve lista de (texto, exacto?). Exacto=True si venía entre comillas.
    Separa por comas/; /\n pero respeta fragmentos entre "..." o '...'.
    """
    if not terms:
        return []
    # Captura frases entre comillas
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', terms)
    exacts = {q[0] or q[1] for q in quoted if (q[0] or q[1])}
    # Elimina las frases exactas del resto
    tmp = terms
    for q in exacts:
        tmp = tmp.replace(f'"{q}"', " ").replace(f"'{q}'", " ")
    # Resto por separadores
    others = [t.strip() for t in re.split(r"[,\n;]+", tmp) if t.strip()]
    out: List[Tuple[str,bool]] = []
    out += [(t, True) for t in exacts]
    out += [(t, False) for t in others]
    return out

def build_regex_from_terms(terms: List[Tuple[str,bool]], whole_words: bool, ignore_case: bool) -> Optional[re.Pattern]:
    """
    Construye un patrón con OR. Para exactos => subcadena literal.
    Para no exactos => palabra completa si whole_words, si no subcadena.
    """
    if not terms:
        return None
    parts: List[str] = []
    for text, is_exact in terms:
        esc = re.escape(text)
        if is_exact:
            parts.append(f"(?:{esc})")
        else:
            if whole_words:
                parts.append(rf"(?<!\w)(?:{esc})(?!\w)")
            else:
                parts.append(f"(?:{esc})")
    body = "|".join(parts)
    flags = re.IGNORECASE if ignore_case else 0
    try:
        return re.compile(body, flags)
    except re.error as e:
        st.error(f"Error en la expresión regular: {e}")
        return None

# =========================
# Traducción (Google Translate endpoint público)
# =========================
async def translate_text(session: aiohttp.ClientSession, text: str, target_lang: str) -> str:
    """
    Usa endpoint público de Google Translate (sin API key).
    Retorna texto traducido (top-1).
    """
    if not text or not target_lang:
        return text
    target_lang = target_lang.lower()
    cache = get_translation_cache()
    key = (text, target_lang)
    if key in cache:
        return cache[key]
    try:
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        url = "https://translate.googleapis.com/translate_a/single"
        async with session.get(url, params=params, timeout=DEFAULT_TIMEOUT, ssl=True, headers={"User-Agent": BASE_HEADERS["User-Agent"]}) as r:
            r.raise_for_status()
            data = await r.json(content_type=None)
        # data[0] es lista de segmentos; cada segmento [trad, orig, ...]
        translated_segments = [seg[0] for seg in data[0] if seg and seg[0]]
        out = "".join(translated_segments).strip()
        if out:
            cache[key] = out
            return out
    except Exception:
        # fallo silencioso: devolvemos original
        return text
    return text

async def translate_terms_list(session: aiohttp.ClientSession, terms: List[Tuple[str,bool]], target_lang: str) -> List[Tuple[str,bool]]:
    out: List[Tuple[str,bool]] = []
    for t, ex in terms:
        tt = await translate_text(session, t, target_lang)
        if tt and tt.lower() != t.lower():
            out.append((tt, ex))
    return out

async def prepare_terms_per_language(
    session: aiohttp.ClientSession,
    include_terms_raw: str,
    exclude_terms_raw: str,
    languages: List[str],
    user_whole_words: bool,
    ignore_case: bool,
) -> Dict[str, Dict[str, Any]]:

    base_inc = split_terms(include_terms_raw)
    base_exc = split_terms(exclude_terms_raw)

    result: Dict[str, Dict[str, Any]] = {}

    for lang in languages:
        lang = (lang or "").lower() or "es"

        # empezamos con los originales
        inc_terms = list(base_inc)
        exc_terms = list(base_exc)

        # si el idioma no es "sin traducir", intentamos traducir
        if lang not in {"", "es"}:
            inc_trans = await translate_terms_list(session, base_inc, lang)
            exc_trans = await translate_terms_list(session, base_exc, lang)

            def merge(a, b):
                seen = set()
                out = []
                for t in a + b:
                    key = (t[0].lower(), t[1])
                    if key not in seen:
                        seen.add(key)
                        out.append(t)
                return out

            inc_terms = merge(inc_terms, inc_trans)
            exc_terms = merge(exc_terms, exc_trans)

        effective_whole_words = user_whole_words and is_latin_lang(lang)

        inc_re = build_regex_from_terms(inc_terms, whole_words=effective_whole_words, ignore_case=ignore_case) if inc_terms else None
        exc_re = build_regex_from_terms(exc_terms, whole_words=effective_whole_words, ignore_case=ignore_case) if exc_terms else None

        result[lang] = {
            "include_terms": inc_terms,
            "exclude_terms": exc_terms,
            "include_re": inc_re,
            "exclude_re": exc_re,
        }

    return result
    
# =========================
# Respetar robots (caché parsers)
# =========================
@st.cache_resource
def get_robot_cache():
    return {}

async def is_allowed(session: aiohttp.ClientSession, headers: Dict[str, str], site_url: str, path: str = "/") -> bool:
    try:
        parsed = urlparse(site_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = urljoin(base, "/robots.txt")
        cache = get_robot_cache()
        parser = cache.get(robots_url)
        if not parser:
            async with session.get(robots_url, headers=headers, timeout=DEFAULT_TIMEOUT, ssl=True) as r:
                if r.status != 200:
                    return True
                txt = await r.text()
            parser = rp.RobotFileParser()
            parser.parse(txt.splitlines())
            cache[robots_url] = parser
        return parser.can_fetch(headers.get("User-Agent", BASE_HEADERS["User-Agent"]), path or "/")
    except Exception:
        return True

# =========================
# Ayuda hemeroteca
# =========================
def iter_archive_urls(source: Dict[str, Any], start_date: date, end_date: date, day_cap: int = 31) -> List[str]:
    patt = source.get("archive_pattern")
    if not patt:
        return []
    urls: List[str] = []
    d = start_date
    count = 0
    while d <= end_date and count < day_cap:
        urls.append(patt.format(yyyy=d.strftime("%Y"), mm=d.strftime("%m"), dd=d.strftime("%d")))
        d += timedelta(days=1)
        count += 1
    return urls

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def iter_archive_urls_for_dates(source: Dict[str, Any], dates: List[date]) -> List[str]:
    patt = source.get("archive_pattern")
    if not patt:
        return []
    urls: List[str] = []
    for d in dates:
        urls.append(patt.format(yyyy=d.strftime("%Y"), mm=d.strftime("%m"), dd=d.strftime("%d")))
    return urls

def is_date_based_pattern(pattern: Optional[str]) -> bool:
    if not pattern:
        return False
    return any(tag in pattern for tag in ["{yyyy}", "{mm}", "{dd}"])

def is_page_based_pattern(pattern: Optional[str]) -> bool:
    if not pattern:
        return False
    return "{page}" in pattern

def iter_archive_urls_for_pages(source: Dict[str, Any], max_pages: int = 10) -> List[str]:
    pattern = source.get("archive_pattern")
    if not pattern or not is_page_based_pattern(pattern):
        return []
    return [pattern.format(page=i) for i in range(1, max_pages + 1)]

# =========================
# Networking asíncrono + RSS/HTML
# =========================
from functools import lru_cache

@lru_cache(maxsize=128)
def fetch_html_cached(url: str, headers_str: str = "", respect_robots: bool = True) -> str:
    return None

async def fetch_html(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    timeout: ClientTimeout,
    ttl_sec: int,
    neg_ttl_sec: int,
    respect_robots: bool,
) -> str:
    # ✅ 1) Cache LRU en memoria (ultrarrápida)
    cache_key = (url, str(sorted(headers.items())), respect_robots)
    cached_html = fetch_html_cached(*cache_key)
    if cached_html:
        return cached_html

    # ✅ 2) Cache tradicional (disco o memoria persistente)
    if neg_cache_hit(url, neg_ttl_sec):
        return ""

    cached = cache_get(url, ttl_sec)
    if cached is not None:
        # También almacenamos en LRU para futuras peticiones rápidas
        fetch_html_cached(url, str(sorted(headers.items())), respect_robots)
        return cached

    # ✅ 3) Respeto a robots.txt (si aplica)
    if respect_robots:
        parsed = urlparse(url)
        if not await is_allowed(session, headers, f"{parsed.scheme}://{parsed.netloc}", parsed.path):
            st.session_state.setdefault("logs", []).append(f"🤖 Robots bloquea {url}")
            neg_cache_put(url)
            return ""

    # ✅ 4) Intentos de descarga con reintentos exponenciales
    tries = 3
    for attempt in range(1, tries + 1):
        try:
            async with session.get(url, headers=headers, timeout=timeout, ssl=True) as resp:
                resp.raise_for_status()
                html = await resp.text()

                # Guarda en ambas cachés
                cache_put(url, html)
                fetch_html_cached(url, str(sorted(headers.items())), respect_robots)

                return html

        except Exception as e:
            if attempt == tries:
                neg_cache_put(url)
                st.session_state.setdefault("logs", []).append(f"❌ GET {url}: {e}")
                return ""

            await asyncio.sleep((2 ** (attempt - 1)) + random.random())

async def try_fetch_rss(
    session: aiohttp.ClientSession,
    page_url: str,
    headers: Dict[str, str],
    timeout: ClientTimeout,
    rss_url: Optional[str] = None,
) -> Optional[List[Tuple[str, str, Optional[datetime]]]]:
    async def fetch_and_parse(feed_url: str) -> Optional[List[Tuple[str, str, Optional[datetime]]]]:
        try:
            async with session.get(feed_url, headers=headers, timeout=timeout, ssl=True) as r2:
                if r2.status != 200:
                    return None
                xml = await r2.text()
            feed = BeautifulSoup(xml, "xml")
            items = feed.find_all(["item", "entry"])
            out: List[Tuple[str, str, Optional[datetime]]] = []
            for it in items:
                title = (it.title.get_text(strip=True) if it.title else "").strip()
                link = (
                    it.link.get("href")
                    if it.link and it.link.has_attr("href")
                    else (it.link.get_text(strip=True) if it.link else "")
                ).strip()
                pub = None
                if it.find("pubDate"):
                    try:
                        pub = parsedate_to_datetime(it.find("pubDate").get_text(strip=True))
                    except Exception:
                        pub = None
                elif it.find("updated"):
                    try:
                        pub = pd.to_datetime(it.find("updated").get_text(strip=True), utc=True, errors="coerce").to_pydatetime()
                    except Exception:
                        pub = None
                out.append((title, link, pub))
            return out or None
        except Exception:
            return None
    if rss_url:
        rss_data = await fetch_and_parse(rss_url)
        if rss_data:
            return rss_data
    try:
        async with session.get(page_url, headers=headers, timeout=timeout, ssl=True) as resp:
            resp.raise_for_status()
            html = await resp.text()
        only_head = SoupStrainer("link")
        soup = BeautifulSoup(html, "html.parser", parse_only=only_head)
        rss_links = [
            l.get("href")
            for l in soup.find_all(
                "link",
                rel=lambda v: v and "alternate" in v,
                type=lambda t: t and ("rss" in t or "atom" in t or "xml" in t),
            )
            if l.get("href")
        ]
        candidates = rss_links + [urljoin(page_url, path) for path in ("/rss", "/feed", "/feeds", "/index.xml")]
        for feed_url in candidates:
            if not feed_url:
                continue
            rss_data = await fetch_and_parse(feed_url)
            if rss_data:
                return rss_data
    except Exception:
        pass
    return None

# =========================
# Scraper (con traducción por medio)
# =========================
async def scrape_source_async(
    session: aiohttp.ClientSession,
    source: Dict[str, Any],
    terms_by_lang: Dict[str, Dict[str, Any]],
    include_terms_raw: str,
    exclude_terms_raw: str,
    timeout: ClientTimeout,
    ttl_sec: int,
    neg_ttl_sec: int,
    headers: Dict[str, str],
    respect_robots: bool,
    use_date_filter: bool,
    date_field: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:

    name = source.get("name")
    url = source.get("url")
    selector_home = source.get("selector")
    selector_archive = source.get("archive_selector") or selector_home
    base_url = source.get("base_url") or None
    lang = (source.get("lang") or "").lower() or "es"

    html_disabled = bool(source.get("html_disabled", False))
    disable_robots = bool(source.get("disable_robots", False))
    rss_fallback = source.get("rss_fallback")

    if not (name and url):
        return []

    # =========================
    # 1) términos ya preparados por idioma
    # =========================
    lang_terms = terms_by_lang.get(lang)
    if not lang_terms:
        # si por lo que sea no tenemos ese idioma, usamos los originales en bruto (sin regex)
        lang_terms = {
            "include_re": None,
            "exclude_re": None,
        }

    include_re = lang_terms.get("include_re")
    exclude_re = lang_terms.get("exclude_re")

    def is_relevant(title: str) -> bool:
        if include_re is None and exclude_re is None:
            return True
        if include_re is not None and not include_re.search(title or ""):
            return False
        if exclude_re is not None and exclude_re.search(title or ""):
            return False
        return True

    rows: List[Dict[str, Any]] = []

    # Particiona rango en pasados y hoy cuando se filtra por publicación
    today = date.today()
    past_days: List[date] = []
    include_today = False
    if use_date_filter and date_field.startswith("Fecha de publicación"):
        for d in daterange(start_date, end_date):
            past_days.append(d)
    else:
        include_today = True

    # =========================
    # 2) ARCHIVOS (pasado)
    # =========================
    if past_days and not html_disabled:
        archive_pattern = source.get("archive_pattern")
        archive_urls = []

        if is_date_based_pattern(archive_pattern):
            archive_urls = iter_archive_urls_for_dates(source, past_days)
        elif is_page_based_pattern(archive_pattern):
            archive_urls = iter_archive_urls_for_pages(source, max_pages=10)

        if archive_urls and selector_archive:
            for page in archive_urls:
                html = await fetch_html(
                    session,
                    page,
                    headers=headers,
                    timeout=timeout,
                    ttl_sec=ttl_sec,
                    neg_ttl_sec=neg_ttl_sec,
                    respect_robots=(respect_robots and not disable_robots),
                )
                if not html:
                    continue
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    elements = soup.select(selector_archive) if selector_archive else []
                    for el in elements:
                        title = el.get_text(strip=True)
                        href = el.get("href")
                        if not href or not title:
                            continue
                        full_url = absolutize(href, base_url or page)
                        if not full_url:
                            continue
                        if is_relevant(title):
                            raw_dt = extract_time_candidate(el)
                            pub_iso = None
                            if raw_dt:
                                try:
                                    dt = pd.to_datetime(raw_dt, utc=True, errors="coerce")
                                    if pd.notnull(dt):
                                        pub_iso = dt.isoformat()
                                except Exception:
                                    pub_iso = None
                            rows.append({
                                "medio": name,
                                "título": title,
                                "url": full_url,
                                "fecha_extraccion": datetime.now().strftime("%Y-%m-%d"),
                                "publicado": pub_iso,
                                "fuente": "html-archivo",
                            })
                except Exception as e:
                    st.session_state.setdefault("logs", []).append(f"❌ {name} ({page}): {e}")

    # =========================
    # 3) HOY (RSS + portada)
    # =========================
    if include_today:
        # a) RSS prioritario: si el medio define rss_fallback lo usamos
        rss_urls_to_try = []
        if rss_fallback:
            rss_urls_to_try.append(rss_fallback)
        rss_urls_to_try.append(url)

        got_rss = False
        for candidate in rss_urls_to_try:
            rss = await try_fetch_rss(session, candidate, headers, timeout)
            if rss:
                got_rss = True
                for title, href, dt in rss:
                    full_url = absolutize(href, base_url or candidate)
                    if not full_url or not title:
                        continue
                    if is_relevant(title):
                        rows.append({
                            "medio": name,
                            "título": title,
                            "url": full_url,
                            "fecha_extraccion": datetime.now().strftime("%Y-%m-%d"),
                            "publicado": (dt.isoformat() if dt else None),
                            "fuente": "rss",
                        })
                break  # no intentamos más RSS

        # b) HTML portada (si no está deshabilitado)
        if selector_home and not html_disabled:
            html = await fetch_html(
                session,
                url,
                headers=headers,
                timeout=timeout,
                ttl_sec=ttl_sec,
                neg_ttl_sec=neg_ttl_sec,
                respect_robots=(respect_robots and not disable_robots),
            )
            if html:
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    elements = soup.select(selector_home) if selector_home else []
                    for el in elements:
                        title = el.get_text(strip=True)
                        href = el.get("href")
                        if not href or not title:
                            continue
                        full_url = absolutize(href, base_url or url)
                        if not full_url:
                            continue
                        if is_relevant(title):
                            raw_dt = extract_time_candidate(el)
                            pub_iso = None
                            if raw_dt:
                                try:
                                    dt = pd.to_datetime(raw_dt, utc=True, errors="coerce")
                                    if pd.notnull(dt):
                                        pub_iso = dt.isoformat()
                                except Exception:
                                    pub_iso = None
                            rows.append({
                                "medio": name,
                                "título": title,
                                "url": full_url,
                                "fecha_extraccion": datetime.now().strftime("%Y-%m-%d"),
                                "publicado": pub_iso,
                                "fuente": "html",
                            })
                except Exception as e:
                    st.session_state.setdefault("logs", []).append(f"❌ {name} (portada): {e}")

    return rows

# =========================
# Paralelización
# =========================
async def run_parallel(
    sources: List[Dict[str, Any]],
    terms_by_lang: Dict[str, Dict[str, Any]],
    include_terms_raw: str,
    exclude_terms_raw: str,
    timeout: ClientTimeout,
    concurrency: int,
    ttl_sec: int,
    neg_ttl_sec: int,
    headers: Dict[str, str],
    respect_robots: bool,
    use_date_filter: bool,
    date_field: str,
    start_date: date,
    end_date: date,
    progress_cb=None,
) -> List[Dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit_per_host=concurrency, ssl=None)
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession(connector=connector) as session:

        async def wrapped(src):
            async with sem:
                out = await scrape_source_async(
                    session=session,
                    source=src,
                    terms_by_lang=terms_by_lang,
                    include_terms_raw=include_terms_raw,
                    exclude_terms_raw=exclude_terms_raw,
                    timeout=timeout,
                    ttl_sec=ttl_sec,
                    neg_ttl_sec=neg_ttl_sec,
                    headers=headers,
                    respect_robots=respect_robots,
                    use_date_filter=use_date_filter,
                    date_field=date_field,
                    start_date=start_date,
                    end_date=end_date,
                )
                if progress_cb:
                    progress_cb(src.get("name", "¿medio?"), len(out))
                return out

        tasks = [asyncio.create_task(wrapped(s)) for s in sources]
        for coro in asyncio.as_completed(tasks):
            out = await coro
            results.extend(out)

    return results

def dedupe_news(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in rows:
        key = (norm_text(r.get("título", "")), r.get("url"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    by_title: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for r in uniq:
        t = norm_text(r.get("título", ""))
        score = (1 if (r.get("fuente","") or "").startswith("rss") else 0) + (1 if r.get("publicado") else 0)
        cur = by_title.get(t)
        if (not cur) or (score > cur[0]):
            by_title[t] = (score, r)
    return [v[1] for v in by_title.values()]

# =========================
# Estado inicial
# =========================
if "db" not in st.session_state:
    st.session_state.db = load_db()
if not st.session_state.db:
    st.stop()
if "logs" not in st.session_state:
    st.session_state.logs = []
# Mapeo país -> continente
COUNTRY_TO_CONTINENT = {
    # Europa
    "España": "Europa",
    "Portugal": "Europa",
    "Francia": "Europa",
    "Albania": "Europa",
    "Andorra": "Europa",
    "Alemania": "Europa",
    "Austria": "Europa",
    "Bélgica": "Europa",
    "Bosnia y Herzegovina": "Europa",
    "Bulgaria": "Europa",
    "Chipre": "Europa",
    "Croacia": "Europa",
    "Dinamarca": "Europa",
    "República Checa": "Europa",
    "Eslovaquia": "Europa",
    "Eslovenia": "Europa",
    "Estonia": "Europa",
    "Finlandia": "Europa",
    "Grecia": "Europa",
    "Hungría": "Europa",
    "Irlanda": "Europa",
    "Islandia": "Europa",
    "Italia": "Europa",
    "Isla de Man": "Europa",
    "Kosovo": "Europa",
    "Luxemburgo": "Europa",
    "Macedonia": "Europa",
    "Malta": "Europa",
    "Mónaco": "Europa",
    "Montenegro": "Europa",
    "Países Bajos": "Europa",
    "Noruega": "Europa",
    "Polonia": "Europa",
    "Reino Unido": "Europa",
    "Rumanía": "Europa",
    "Rusia": "Europa",
    "San Marino": "Europa",
    "Serbia": "Europa",
    "Suecia": "Europa",
    "Suiza": "Europa",
    "Ucrania": "Europa",
    # África
    "Algeria": "África",
    "Marruecos": "África",
    "Argelia": "África",
    "Burundi": "África",
    "Camerún": "África",
    "Cabo Verde": "África",
    "República Democrática del Congo": "África",
    "Yibuti": "África",
    "Egipto": "África",
    "Eritrea": "África",
    "Etiopía": "África",
    "Gambia": "África",
    "Guinea": "África",
    "Kenia": "África",
    "Liberia": "África",
    "Madagascar": "África",
    "Malaui": "África",
    "Mauricio": "África",
    "Namibia": "África",
    "Níger": "África",
    "Nigeria": "África",
    "Senegal": "África",
    "Somalia": "África",
    "Sudáfrica": "África",
    "Uganda": "África",
    "Zambia": "África",
    "Zimbabue": "África",
    # América
    "Barbados": "América",
    "Belice": "América",
    "Bermudas": "América",
    "Bolivia": "América",
    "Brasil": "América",
    "Canadá": "América",
    "Chile": "América",
    "Colombia": "América",
    "Costa Rica": "América",
    "Cuba": "América",
    "Dominica": "América",
    "República Dominicana": "América",
    "Ecuador": "América",
    "Argentina": "América",
    "Bahamas": "América",
    "Estados Unidos": "América",
    "Aruba": "América",
    "Guatemala": "América",
    "Guyana": "América",
    "Haití": "América",
    "Jamaica": "América",
    "México": "América",
    "Panamá": "América",
    "Perú": "América",
    "Uruguay": "América",
    "Venezuela": "América",
    "Saint Kitts y Nevis": "América",
    "Surinam": "América",
    "Martinica": "América",
    # Asia
    "Afghanistan": "Asia",
    "Arabía Saudí": "Asia",
    "Armenia": "Asia",
    "Azerbaiyán": "Asia",
    "Baréin": "Asia",
    "Bangladés": "Asia",
    "Birmania": "Asia",
    "Japón": "Asia",
    "Palestina": "Asia",
    "Israel": "Asia",
    "India": "Asia",
    "Indonesia": "Asia",
    "Irán": "Asia",
    "Hong Kong": "Asia",
    "Jordania": "Asia",
    "Kazajistán": "Asia",
    "Kuwait": "Asia",
    "Kirguistán": "Asia",
    "Malasia": "Asia",
    "Maldivas": "Asia",
    "Macao": "Asia",
    "Mongolia": "Asia",
    "Nepal": "Asia",
    "Pakistán": "Asia",
    "Palestina": "Asia",
    "Catar": "Asia",
    "Arabia Saudita": "Asia",
    "Sri Lanka": "Asia",
    "Siria": "Asia",
    "Tailandia": "Asia",
    "Timor Oriental": "Asia",
    "Turquía": "Asia",
    "Vietnam": "Asia",
    "Yemen": "Asia",
    # Oceanía
    "Australia": "Oceanía",
    "Polinesia Francesa": "Oceanía",
    "Islas Caimán": "Oceanía",
    "Guam": "Oceanía",
    "Samoa": "Oceanía",
    "Nueva Zelanda": "Oceanía",
}

def get_continent_for_country(country: str) -> str:
    return COUNTRY_TO_CONTINENT.get(country, "Otros")

# Defaults filtro temporal y traducción
if "use_date_filter" not in st.session_state:
    st.session_state.use_date_filter = False
if "date_range" not in st.session_state:
    st.session_state.date_range = (date.today() - timedelta(days=7), date.today())
if "date_field" not in st.session_state:
    st.session_state.date_field = "Fecha de publicación (recomendado)"
if "include_na_pub" not in st.session_state:
    st.session_state.include_na_pub = True
if "translate_per_source" not in st.session_state:
    st.session_state.translate_per_source = True

# =========================
# Sidebar: gestión BD
# =========================
all_countries = sorted(st.session_state.db.keys())

continent_to_countries = {}
for c in all_countries:
    cont = get_continent_for_country(c)
    continent_to_countries.setdefault(cont, []).append(c)

all_continents = sorted(continent_to_countries.keys())

st.sidebar.header("🗂️ LISTADO DE MEDIOS")

sidebar_cont = st.sidebar.selectbox("Continente", all_continents, key="sidebar_continent")
countries_in_sidebar_cont = sorted(continent_to_countries.get(sidebar_cont, []))

default_country = "España" if "España" in countries_in_sidebar_cont else (countries_in_sidebar_cont[0] if countries_in_sidebar_cont else "")
country = st.sidebar.selectbox(
    "País",
    countries_in_sidebar_cont,
    index=countries_in_sidebar_cont.index(default_country) if default_country in countries_in_sidebar_cont else 0,
    key="sidebar_country",
)

with st.sidebar.expander(
    f"📜 Medios en {country} ({len(st.session_state.db.get(country, []))})",
    expanded=True
):
    medios = st.session_state.db.get(country, [])
    if not medios:
        st.caption("Sin medios registrados.")
    else:
        medios = sorted(medios, key=lambda x: x.get("name", ""))
        cols = st.columns(2)
        for i, s in enumerate(medios):
            with cols[i % 2]:
                st.link_button(s["name"], s["url"], use_container_width=True)

st.sidebar.markdown("---")

# =========================
# Controles de búsqueda
# =========================
all_countries = sorted(st.session_state.db.keys())

# Construimos continentes a partir de lo que haya en el JSON
continent_to_countries = {}
for c in all_countries:
    cont = get_continent_for_country(c)
    continent_to_countries.setdefault(cont, []).append(c)

all_continents = sorted(continent_to_countries.keys())

# 1) selector de continente
selected_continent = st.selectbox(
    "Continente",
    all_continents,
    key="search_continent",
)

# 2) selector de país (filtrado por continente)
countries_in_cont = sorted(continent_to_countries.get(selected_continent, []))

# si el país guardado en sesión no está en este continente, ponemos el primero
default_country_for_cont = (
    st.session_state.get("search_country")
    if st.session_state.get("search_country") in countries_in_cont
    else (countries_in_cont[0] if countries_in_cont else "")
)

search_country = st.selectbox(
    "País a buscar",
    countries_in_cont,
    index=countries_in_cont.index(default_country_for_cont) if default_country_for_cont in countries_in_cont else 0,
    key="search_country",
)

# (2) Texto y opciones básicas
colA, colB, colC = st.columns([2, 2, 1])
with colA:
    include_terms = st.text_input(
        "Términos a incluir:",
        placeholder='Ej.: "Cambio climático", flotilla, Gobierno',
    )
with colB:
    exclude_terms = st.text_input("Términos a excluir (opcional):", placeholder='Ej.: "Guerra civil", subvención, militar')
with colC:
    whole_words = st.checkbox("Coincidencia por palabra", value=False, help="Se aplicará solo en idiomas latinos automáticamente.")
    ignore_case = st.checkbox("Ignorar mayúsc./minúsc.", value=True)

# (3) Filtro temporal
with st.expander("🗓️ Filtro temporal"):
    st.checkbox("Filtrar por periodo de fechas", key="use_date_filter", value=st.session_state.use_date_filter)

    if st.session_state.use_date_filter:
        st.date_input(
            "Periodo (inicio y fin)",
            key="date_range",
            value=st.session_state.date_range,
            format="DD-MM-YYYY",
        )

        st.selectbox(
            "Campo de fecha a usar",
            ["Fecha de publicación (recomendado)", "Fecha de extracción"],
            key="date_field",
            index=0 if st.session_state.date_field.startswith("Fecha de publicación") else 1,
        )

        st.checkbox(
            "Incluir noticias sin fecha de publicación",
            key="include_na_pub",
            value=st.session_state.include_na_pub,
            help="Sólo aplica cuando filtras por 'Fecha de publicación'.",
        )

# (3b) Traducción por medio
with st.expander("🌐 Traducción de términos"):
    st.checkbox(
        "Traducir términos por idioma de cada medio (Google Translate)",
        key="translate_per_source",
        help="Mantiene el término original y añade su traducción. Si usas comillas, también se hace coincidencia exacta."
    )
    st.caption("La traducción se aplica al pulsar **Buscar** para evitar llamadas innecesarias. Se cachea por texto+idioma.")

# Leer/normalizar valores del filtro
use_date_filter = bool(st.session_state.get("use_date_filter", False))
dr = st.session_state.get("date_range", None)
if isinstance(dr, (list, tuple)) and len(dr) == 2:
    start_date, end_date = dr
else:
    d = dr or (date.today(),)
    d = d[0] if isinstance(d, (list, tuple)) else d
    start_date, end_date = d, d
date_field = st.session_state.get("date_field", "Fecha de publicación (recomendado)")
include_na_pub = bool(st.session_state.get("include_na_pub", True))
translate_per_source = bool(st.session_state.get("translate_per_source", True))

# (4) Opciones avanzadas
with st.expander("⚙️ Opciones avanzadas"):
    current_sources = st.session_state.db.get(search_country, [])
    preselect = [s["name"] for s in current_sources]
    selected_names = st.multiselect(
        f"Medios de {search_country} a incluir",
        options=preselect,
        default=preselect,
        key=f"sources_{search_country}",
    )
    timeout_sec = st.slider("Timeout por petición (seg.)", min_value=5, max_value=40, value=20, step=1)
    concurrency = st.slider("Concurrencia (simultáneos)", min_value=2, max_value=20, value=8, step=1)
    ttl_sec = st.slider("TTL caché positiva (seg.)", min_value=60, max_value=3600, value=900, step=30)
    neg_ttl_sec = st.slider("TTL caché negativa (seg.)", min_value=5, max_value=120, value=30, step=5)
    respect_robots = st.checkbox("Respetar robots.txt (beta)", value=True)
    show_log = st.checkbox("Mostrar LOG al finalizar", value=True)

# =========================
# Acción: Buscar
# =========================
def add_log_line(name: str, n: int):
    st.session_state.logs.append(f"✅ {name}: {n} resultados")

if st.button("🔍 Buscar en país seleccionado", type="primary"):
    st.session_state.logs = []

    # Fuentes a consultar
    sources = [s for s in st.session_state.db.get(search_country, []) if s["name"] in selected_names]

    if not sources:
        st.warning(f"No hay medios seleccionados para {search_country}.")
    else:
        start = time.time()
        headers = build_headers(search_country)
        timeout = ClientTimeout(
            total=timeout_sec,
            connect=min(10, timeout_sec),
            sock_connect=min(10, timeout_sec),
            sock_read=min(max(5, timeout_sec - 5), timeout_sec),
        )
        with st.spinner("Buscando en medios…"):
            # 2) ejecuto todo en un solo loop
            async def main():
                async with aiohttp.ClientSession() as session:
                    # 2a) preparar términos por idioma
                    needed_langs = sorted(set((s.get("lang") or "es").lower() for s in sources))
                    terms_by_lang = await prepare_terms_per_language(
                        session=session,
                        include_terms_raw=include_terms,
                        exclude_terms_raw=exclude_terms,
                        languages=needed_langs,
                        user_whole_words=whole_words,
                        ignore_case=ignore_case,
                    )
                    # 2b) lanzar scrapers en paralelo
                    rows_all = await run_parallel(
                        sources=sources,
                        terms_by_lang=terms_by_lang,
                        include_terms_raw=include_terms,
                        exclude_terms_raw=exclude_terms,
                        timeout=timeout,
                        concurrency=concurrency,
                        ttl_sec=ttl_sec,
                        neg_ttl_sec=neg_ttl_sec,
                        headers=headers,
                        respect_robots=respect_robots,
                        use_date_filter=use_date_filter,
                        date_field=date_field,
                        start_date=start_date,
                        end_date=end_date,
                        progress_cb=add_log_line,
                    )
                    return rows_all
        
            rows_all: List[Dict[str, Any]] = asyncio.run(main())

        elapsed = time.time() - start
        rows_all = dedupe_news(rows_all)

        if not rows_all:
            st.warning("🚫 No se encontraron noticias con los criterios dados.")
        else:
            df = pd.DataFrame(rows_all)

            # Filtrado temporal post (seguridad)
            if use_date_filter:
                df["_pub_dt"] = pd.to_datetime(df.get("publicado"), utc=True, errors="coerce")
                df["_ext_dt"] = pd.to_datetime(df.get("fecha_extraccion"), format="%Y-%m-%d", errors="coerce")

                start_ts_utc = pd.Timestamp(start_date).tz_localize("UTC")
                end_ts_utc = (pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
                start_ts_naive = pd.Timestamp(start_date)
                end_ts_naive = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))

                if date_field.startswith("Fecha de publicación"):
                    mask_pub = df["_pub_dt"].between(start_ts_utc, end_ts_utc, inclusive="both")
                    if include_na_pub:
                        df = df[mask_pub | df["_pub_dt"].isna()].copy()
                    else:
                        df = df[mask_pub].copy()
                else:
                    mask_ext = df["_ext_dt"].between(start_ts_naive, end_ts_naive, inclusive="both")
                    df = df[mask_ext].copy()

            if "publicado" in df.columns:
                df["_dt"] = pd.to_datetime(df["publicado"], utc=True, errors="coerce")
            else:
                df["_dt"] = pd.NaT
            df = df.sort_values(["_dt", "fecha_extraccion"], ascending=[False, False]).drop(columns=["_dt"], errors="ignore")

            st.success(f"✅ {len(df)} noticias encontradas en {search_country} (en {elapsed:.1f}s).")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.write("### Resultados")
            for _, row in df.iterrows():
                fuente = row.get("fuente") or "-"
                publicado = row.get("publicado") or "—"
                st.markdown(
                    f"""
                    <div style="border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;margin-bottom:10px;">
                      <p style="margin:0;color:#6b7280;font-size:13px;">📰 <strong>{row['medio']}</strong> · <span style="background:#eef2ff;padding:2px 6px;border-radius:6px;">{(fuente or '').upper()}</span></p>
                      <p style="margin:6px 0;">
                        <a href="{row['url']}" target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer" style="font-size:16px;text-decoration:none;">
                          {row['título']}
                        </a>
                      </p>
                      <p style="margin:0;color:#6b7280;font-size:12px;">📅 Publicado: {publicado} · Extraído: {row['fecha_extraccion']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar CSV",
                csv,
                file_name=f"noticias_{search_country}_{datetime.now().date()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if show_log and st.session_state.logs:
            st.markdown("### LOG")
            st.code("\n".join(st.session_state.logs), language="text")
