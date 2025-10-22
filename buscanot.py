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
st.title("🌍 Buscador de noticias por país")

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
# Base por defecto con patrones de hemeroteca donde se conocen
# (puedes ir ampliando según cada medio)
DEFAULT_DB: Dict[str, List[Dict[str, Any]]] = {
    "España": [
        {
            "name": "El País",
            "url": "https://elpais.com/",
            "selector": "article h2 a, h3 a, .headline a",
            "base_url": "https://elpais.com",
            "archive_pattern": "https://elpais.com/archivo/{yyyy}-{mm}-{dd}/",
        },
        {
            "name": "El Mundo",
            "url": "https://www.elmundo.es/",
            "selector": "article h2 a, h3 a, .ue-c-cover-content__link",
            "base_url": "https://www.elmundo.es",
            "archive_pattern": "https://www.elmundo.es/elmundo/hemeroteca/{yyyy}/{mm}/{dd}/",
        },
        {
            "name": "ABC",
            "url": "https://www.abc.es/",
            "selector": "article h2 a, h3 a, .titular a",
            "base_url": "https://www.abc.es",
            "archive_pattern": "https://www.abc.es/archivo/{yyyy}-{mm}-{dd}/",
        },
        {
            "name": "La Vanguardia",
            "url": "https://www.lavanguardia.com/",
            "selector": "article h2 a, h3 a, .headline a",
            "base_url": "https://www.lavanguardia.com",
            "archive_pattern": "https://www.lavanguardia.com/hemeroteca/{yyyy}/{mm}/{dd}",
        },
        {
            "name": "El Confidencial",
            "url": "https://www.elconfidencial.com/",
            "selector": "article h2 a, h3 a, .news__title a",
            "base_url": "https://www.elconfidencial.com",
            # sin hemeroteca pública por fecha conocida
        },
        {
            "name": "20minutos",
            "url": "https://www.20minutos.es/",
            "selector": "article h2 a, h3 a, .headline a",
            "base_url": "https://www.20minutos.es",
            # sin patrón estable público
        },
        {
            "name": "RTVE",
            "url": "https://www.rtve.es/noticias/",
            "selector": "article h2 a, h3 a, .headline a",
            "base_url": "https://www.rtve.es/noticias/",
            # sin patrón estable público
        },
        {
            "name": "La Razón",
            "url": "https://www.larazon.es/",
            "selector": "article h2 a, h3 a, .headline a",
            "base_url": "https://www.larazon.es/",
            # sin patrón estable público
        },
        {
            "name": "El Periodico de Catalunya",
            "url": "https://www.elperiodico.com/es/",
            "selector": "article h2 a, h3 a, .headline a",
            "base_url": "https://www.elperiodico.com/",
            # sin patrón estable público
        },
    ],
    "Marruecos": [
        {"name": "Hespress (AR)", "url": "https://www.hespress.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.hespress.com"},
        {"name": "Hespress (EN)", "url": "https://en.hespress.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://en.hespress.com"},
        {"name": "Le Matin", "url": "https://lematin.ma/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://lematin.ma"},
        {"name": "L'Economiste", "url": "https://leconomiste.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://leconomiste.com"},
        {"name": "TelQuel", "url": "https://telquel.ma/", "selector": "article h2 a, h3 a, .post-title a", "base_url": "https://telquel.ma"},
        {"name": "Aujourd'hui Le Maroc", "url": "https://aujourdhui.ma/", "selector": "article h2 a, h3 a, .entry-title a", "base_url": "https://aujourdhui.ma"},
        {"name": "Médias24", "url": "https://medias24.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://medias24.com"},
        {"name": "Morocco World News", "url": "https://www.moroccoworldnews.com/", "selector": "article h2 a, h3 a, .card-title a", "base_url": "https://www.moroccoworldnews.com"},
    ],
    "Francia": [
        {"name": "Le Monde", "url": "https://www.lemonde.fr/", "selector": "article h2 a, h3 a, .article__title a", "base_url": "https://www.lemonde.fr"},
        {"name": "Le Figaro", "url": "https://www.lefigaro.fr/", "selector": "article h2 a, h3 a, .fig-headline a", "base_url": "https://www.lefigaro.fr"},
        {"name": "Libération", "url": "https://www.liberation.fr/", "selector": "article h2 a, h3 a, .article-card a", "base_url": "https://www.liberation.fr"},
        {"name": "Le Parisien", "url": "https://www.leparisien.fr/", "selector": "article h2 a, h3 a, .article-title a", "base_url": "https://www.leparisien.fr"},
        {"name": "Les Échos", "url": "https://www.lesechos.fr/", "selector": "article h2 a, h3 a, .teaser__title a", "base_url": "https://www.lesechos.fr"},
        {"name": "La Croix", "url": "https://www.la-croix.com/", "selector": "article h2 a, h3 a, .article__title a", "base_url": "https://www.la-croix.com"},
        {"name": "Ouest-France", "url": "https://www.ouest-france.fr/", "selector": "article h2 a, h3 a, .teaser-title a", "base_url": "https://www.ouest-france.fr"},
    ],
    "Portugal": [
        {"name": "Público", "url": "https://www.publico.pt/", "selector": "article h2 a, h3 a, .headline a", "base_url": "https://www.publico.pt"},
        {"name": "Diário de Notícias", "url": "https://www.dn.pt/", "selector": "article h2 a, h3 a, .card-title a", "base_url": "https://www.dn.pt"},
        {"name": "Expresso", "url": "https://expresso.pt/", "selector": "article h2 a, h3 a, .article-title a", "base_url": "https://expresso.pt"},
        {"name": "Observador", "url": "https://observador.pt/", "selector": "article h2 a, h3 a, .headline a", "base_url": "https://observador.pt"},
        {"name": "Correio da Manhã", "url": "https://www.cmjornal.pt/", "selector": "article h2 a, h3 a, .tit a", "base_url": "https://www.cmjornal.pt"},
        {"name": "Jornal de Notícias", "url": "https://www.jn.pt/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.jn.pt"},
    ],
    "Andorra": [
        {"name": "Diari d'Andorra", "url": "https://www.diariandorra.ad/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.diariandorra.ad"},
        {"name": "Bondia Andorra", "url": "https://www.bondia.ad/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.bondia.ad"},
        {"name": "Altaveu", "url": "https://www.altaveu.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.altaveu.com"},
    ],
    "Alemania": [
        {"name": "Frankfurter Allgemeine (FAZ)", "url": "https://www.faz.net/", "selector": "article h2 a, h3 a, .headline a", "base_url": "https://www.faz.net"},
        {"name": "Süddeutsche Zeitung", "url": "https://www.sueddeutsche.de/", "selector": "article h2 a, h3 a, .sz-article__title a", "base_url": "https://www.sueddeutsche.de"},
        {"name": "WELT", "url": "https://www.welt.de/", "selector": "article h2 a, h3 a, .c-teaser__headline a", "base_url": "https://www.welt.de"},
        {"name": "Der Spiegel", "url": "https://www.spiegel.de/", "selector": "article h2 a, h3 a, .leading-article a", "base_url": "https://www.spiegel.de"},
        {"name": "Die Zeit", "url": "https://www.zeit.de/index", "selector": "article h2 a, h3 a, .zon-teaser-standard__title a", "base_url": "https://www.zeit.de"},
        {"name": "BILD", "url": "https://www.bild.de/", "selector": "article h2 a, h3 a, .headline a", "base_url": "https://www.bild.de"},
        {"name": "tagesschau (ARD)", "url": "https://www.tagesschau.de/", "selector": "article h2 a, h3 a, .teaser__title a", "base_url": "https://www.tagesschau.de"},
    ],
}

# =========================
# Persistencia BD
# =========================
def save_db(db: Dict[str, List[Dict[str, Any]]]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def load_db() -> Dict[str, List[Dict[str, Any]]]:
    if DB_PATH.exists():
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            st.warning("No se pudo leer media_db.json. Se carga la base por defecto.")
    save_db(DEFAULT_DB)
    return json.loads(json.dumps(DEFAULT_DB))

# =========================
# Caché en memoria con TTL (positiva y negativa)
# =========================
@st.cache_resource
def get_html_cache() -> Dict[str, Tuple[float, str]]:
    """Dict: url -> (timestamp_epoch, html_text)"""
    return {}

@st.cache_resource
def get_neg_cache() -> Dict[str, float]:
    """Dict: url -> timestamp_epoch (para caché negativa corta)"""
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

def build_regex(terms: str, whole_words: bool = False, ignore_case: bool = True) -> Optional[re.Pattern]:
    cleaned = [t.strip() for t in re.split(r"[,\n;]+", terms) if t.strip()]
    if not cleaned:
        return None
    escaped = [re.escape(t) for t in cleaned]
    body = "|".join(escaped)
    if whole_words:
        body = rf"(?<!\w)(?:{body})(?!\w)"
    else:
        body = rf"(?:{body})"
    flags = re.IGNORECASE if ignore_case else 0
    try:
        return re.compile(body, flags)
    except re.error as e:
        st.error(f"Error en la expresión regular: {e}")
        return None

def is_relevant(title: str, include_re: Optional[re.Pattern], exclude_re: Optional[re.Pattern]) -> bool:
    if include_re is None and exclude_re is None:
        return True
    if include_re is not None and not include_re.search(title or ""):
        return False
    if exclude_re is not None and exclude_re.search(title or ""):
        return False
    return True

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

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_time_candidate(el) -> Optional[str]:
    t = el.find("time")
    if t and (t.get("datetime") or t.get_text(strip=True)):
        return t.get("datetime") or t.get_text(strip=True)
    for attr in ("data-published", "data-date", "data-time", "aria-label"):
        v = el.get(attr)
        if v:
            return v
    return None

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
        return True  # en caso de duda, permitir

# =========================
# Ayuda hemeroteca
# =========================
def iter_archive_urls(source: Dict[str, Any], start_date: date, end_date: date, day_cap: int = 31) -> List[str]:
    """
    Genera URLs diarias usando source['archive_pattern'] si existe.
    Limita a 'day_cap' días para evitar abusos accidentales.
    """
    patt = source.get("archive_pattern")
    if not patt:
        return []
    urls: List[str] = []
    d = start_date
    count = 0
    while d <= end_date and count < day_cap:
        urls.append(
            patt.format(
                yyyy=d.strftime("%Y"), mm=d.strftime("%m"), dd=d.strftime("%d")
            )
        )
        d += timedelta(days=1)
        count += 1
    return urls

def date_range_includes_today(start_date: date, end_date: date) -> bool:
    today = date.today()
    return start_date <= today <= end_date

# =========================
# Networking asíncrono + RSS/HTML
# =========================
async def fetch_html(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    timeout: ClientTimeout,
    ttl_sec: int,
    neg_ttl_sec: int,
    respect_robots: bool,
) -> str:
    """Devuelve HTML usando caché por URL con TTL y caché negativa corta."""
    if neg_cache_hit(url, neg_ttl_sec):
        return ""

    cached = cache_get(url, ttl_sec)
    if cached is not None:
        return cached

    if respect_robots:
        parsed = urlparse(url)
        if not await is_allowed(session, headers, f"{parsed.scheme}://{parsed.netloc}", parsed.path):
            st.session_state.setdefault("logs", []).append(f"🤖 Robots bloquea {url}")
            neg_cache_put(url)
            return ""

    tries = 3
    for attempt in range(1, tries + 1):
        try:
            async with session.get(url, headers=headers, timeout=timeout, ssl=True) as resp:
                resp.raise_for_status()
                html = await resp.text()
                cache_put(url, html)
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
) -> Optional[List[Tuple[str, str, Optional[datetime]]]]:
    """Intenta localizar y descargar un feed RSS/Atom desde la portada."""
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
            try:
                async with session.get(feed_url, headers=headers, timeout=timeout, ssl=True) as r2:
                    if r2.status != 200:
                        continue
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
                if out:
                    return out
            except Exception:
                continue
    except Exception:
        pass
    return None

async def scrape_source_async(
    session: aiohttp.ClientSession,
    source: Dict[str, Any],
    include_re: Optional[re.Pattern],
    exclude_re: Optional[re.Pattern],
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
    """
    Scrapea una fuente. Si el filtro temporal está activo y se filtra por
    Fecha de publicación y hay 'archive_pattern', recorre las URLs diarias.
    Si no hay patrón, usa portada/RSS como siempre.
    """
    name = source.get("name")
    url = source.get("url")
    selector = source.get("selector")
    base_url = source.get("base_url") or None

    if not (name and url):
        return []

    rows: List[Dict[str, Any]] = []

    # ¿Tenemos rango por publicación y patrón de archivo?
    use_archives = False
    archive_urls: List[str] = []
    if use_date_filter and date_field.startswith("Fecha de publicación"):
        archive_urls = iter_archive_urls(source, start_date, end_date)
        use_archives = len(archive_urls) > 0

    # 1) RSS/Atom (si NO estamos usando páginas de archivo por día)
    if not use_archives:
        rss = await try_fetch_rss(session, url, headers, timeout)
        if rss:
            for title, href, dt in rss:
                full_url = absolutize(href, base_url or url)
                if not full_url or not title:
                    continue
                if is_relevant(title, include_re, exclude_re):
                    rows.append(
                        {
                            "medio": name,
                            "título": title,
                            "url": full_url,
                            "fecha_extraccion": datetime.now().strftime("%Y-%m-%d"),
                            "publicado": (dt.isoformat() if dt else None),
                            "fuente": "rss",
                        }
                    )
            if rows:
                return rows

    # 2) HTML (portada o páginas de archivo diarias)
    targets = archive_urls if use_archives else [url]
    if not selector:
        return rows  # si veníamos de RSS puede haber filas; si no, vacío

    for page in targets:
        html = await fetch_html(
            session, page, headers=headers, timeout=timeout,
            ttl_sec=ttl_sec, neg_ttl_sec=neg_ttl_sec, respect_robots=respect_robots
        )
        if not html:
            continue

        try:
            soup = BeautifulSoup(html, "html.parser")
            elements = soup.select(selector) if selector else []
            for el in elements:
                title = el.get_text(strip=True)
                href = el.get("href")
                if not href or not title:
                    continue
                full_url = absolutize(href, base_url or page)
                if not full_url:
                    continue
                if is_relevant(title, include_re, exclude_re):
                    raw_dt = extract_time_candidate(el)
                    pub_iso = None
                    if raw_dt:
                        try:
                            dt = pd.to_datetime(raw_dt, utc=True, errors="coerce")
                            if pd.notnull(dt):
                                pub_iso = dt.isoformat()
                        except Exception:
                            pub_iso = None
                    rows.append(
                        {
                            "medio": name,
                            "título": title,
                            "url": full_url,
                            "fecha_extraccion": datetime.now().strftime("%Y-%m-%d"),
                            "publicado": pub_iso,
                            "fuente": "html-archivo" if use_archives else "html",
                        }
                    )
        except Exception as e:
            st.session_state.setdefault("logs", []).append(f"❌ {name} ({page}): {e}")
            continue

    return rows

async def run_parallel(
    sources: List[Dict[str, Any]],
    include_re: Optional[re.Pattern],
    exclude_re: Optional[re.Pattern],
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
    """
    Ejecuta scraping en paralelo con límite de concurrencia.
    """
    connector = aiohttp.TCPConnector(limit_per_host=concurrency, ssl=None)  # verificación TLS por defecto
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession(connector=connector) as session:

        async def wrapped(src):
            async with sem:
                out = await scrape_source_async(
                    session, src, include_re, exclude_re, timeout, ttl_sec, neg_ttl_sec, headers, respect_robots,
                    use_date_filter, date_field, start_date, end_date
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
    # 1) de-dup exactos por (titulo normalizado, url)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in rows:
        key = (norm_text(r.get("título", "")), r.get("url"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    # 2) Agrupa por título y elige el mejor (RSS y/o con fecha gana)
    by_title: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for r in uniq:
        t = norm_text(r.get("título", ""))
        score = (1 if r.get("fuente", "").startswith("rss") else 0) + (1 if r.get("publicado") else 0)
        cur = by_title.get(t)
        if (not cur) or (score > cur[0]):
            by_title[t] = (score, r)
    return [v[1] for v in by_title.values()]

# =========================
# Estado inicial
# =========================
if "db" not in st.session_state:
    st.session_state.db = load_db()
if "logs" not in st.session_state:
    st.session_state.logs = []
if "search_country" not in st.session_state:
    all_keys = sorted(st.session_state.db.keys())
    st.session_state.search_country = "España" if "España" in all_keys else (all_keys[0] if all_keys else "")

# Defaults seguros para filtro temporal (en session_state)
if "use_date_filter" not in st.session_state:
    st.session_state.use_date_filter = False
if "date_range" not in st.session_state:
    st.session_state.date_range = (date.today() - timedelta(days=7), date.today())
if "date_field" not in st.session_state:
    st.session_state.date_field = "Fecha de publicación (recomendado)"
if "include_na_pub" not in st.session_state:
    st.session_state.include_na_pub = True

# =========================
# Sidebar: gestión BD
# =========================
st.sidebar.header("🗂️ Listado de medios por país")

all_countries = sorted(st.session_state.db.keys())
default_country = "España" if "España" in all_countries else (all_countries[0] if all_countries else "España")
country = st.sidebar.selectbox("País (para gestionar)", all_countries, index=all_countries.index(default_country))

with st.sidebar.expander(f"📜 Medios en {country} ({len(st.session_state.db.get(country, []))})", expanded=True):
    if not st.session_state.db.get(country):
        st.caption("Sin medios registrados.")
    else:
        for s in st.session_state.db[country]:
            st.markdown(f"- **{s['name']}** — [{s['url']}]({s['url']})  \n  `selector`: `{s['selector']}`  · `base_url`: `{s.get('base_url') or ''}`" + (f"  · `archive_pattern`: `{s.get('archive_pattern')}`" if s.get("archive_pattern") else ""))

st.sidebar.markdown("---")

# =========================
# Controles de búsqueda
# =========================
st.header("🔍 Búsqueda")

# (1) País a buscar — FUERA del expandible
search_country = st.selectbox(
    "País a buscar",
    sorted(st.session_state.db.keys()),
    key="search_country",  # mantiene valor entre reruns
)

# (2) Texto y opciones básicas
colA, colB, colC = st.columns([2, 2, 1])
with colA:
    include_terms = st.text_input(
        "Términos a incluir (coma, punto y coma o saltos de línea):",
        placeholder="Ej.: altercado, flotilla, Gobierno",
    )
with colB:
    exclude_terms = st.text_input("Términos a excluir (opcional):", placeholder="ej.: subvención, guerra, militar")
with colC:
    whole_words = st.checkbox("Coincidencia por palabra", value=False)
    ignore_case = st.checkbox("Ignorar mayúsc./minúsc.", value=True)

# (3) Filtro temporal (widgets con key)
with st.expander("🗓️ Filtro temporal"):
    st.checkbox(
        "Filtrar por periodo de fechas",
        key="use_date_filter",
        value=st.session_state.use_date_filter,
    )
    if st.session_state.use_date_filter:
        st.date_input(
            "Periodo (inicio y fin)",
            key="date_range",
            value=st.session_state.date_range,
            format="YYYY-MM-DD",
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

# UX: aviso sobre hemerotecas faltantes si aplica
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

if use_date_filter and date_field.startswith("Fecha de publicación"):
    # lista medios del país actual sin archive_pattern
    missing = [s["name"] for s in st.session_state.db.get(search_country, []) if s.get("name") and not s.get("archive_pattern")]
    if missing and not date_range_includes_today(start_date, end_date):
        st.info(
            "ℹ️ Para fechas **anteriores** a hoy, algunos medios no tienen `archive_pattern` y pueden no devolver resultados: "
            + ", ".join(missing)
        )

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

if st.button("🚀 Buscar en medios del país seleccionado", type="primary"):
    st.session_state.logs = []

    include_re = build_regex(include_terms, whole_words=whole_words, ignore_case=ignore_case)
    exclude_re = build_regex(exclude_terms, whole_words=whole_words, ignore_case=ignore_case)

    # Fuentes a consultar
    sources = [s for s in st.session_state.db.get(search_country, []) if s["name"] in selected_names]

    if not sources:
        st.warning(f"No hay medios seleccionados para {search_country}.")
    else:
        start = time.time()
        headers = build_headers(search_country)
        # timeout personalizado
        timeout = ClientTimeout(
            total=timeout_sec,
            connect=min(10, timeout_sec),
            sock_connect=min(10, timeout_sec),
            sock_read=min(max(5, timeout_sec - 5), timeout_sec),
        )
        with st.spinner("Buscando en paralelo…"):
            rows_all: List[Dict[str, Any]] = asyncio.run(
                run_parallel(
                    sources=sources,
                    include_re=include_re,
                    exclude_re=exclude_re,
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
            )

        elapsed = time.time() - start
        rows_all = dedupe_news(rows_all)

        if not rows_all:
            st.warning("🚫 No se encontraron noticias con los criterios dados.")
        else:
            df = pd.DataFrame(rows_all)

            # --- Filtrado temporal (aplicado tras construir df) ---
            if use_date_filter:
                # Normaliza a datetimes
                df["_pub_dt"] = pd.to_datetime(df.get("publicado"), utc=True, errors="coerce")   # tz-aware UTC
                df["_ext_dt"] = pd.to_datetime(df.get("fecha_extraccion"), format="%Y-%m-%d", errors="coerce")  # naive

                # Límites inclusivos por día completo
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

            # Ordenar por fecha publicada y luego por extracción
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
                      <p style="margin:0;color:#6b7280;font-size:13px;">📰 <strong>{row['medio']}</strong> · <span style="background:#eef2ff;padding:2px 6px;border-radius:6px;">{fuente.upper()}</span></p>
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
