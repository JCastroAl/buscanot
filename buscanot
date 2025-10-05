# app.py
import asyncio
import aiohttp
import time
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# =========================
# Configuración general
# =========================
st.set_page_config(page_title="Buscador de noticias por país", layout="wide")
st.title("🌍⚡ Buscador de noticias por país (con caché y paralelo)")

DB_PATH = Path("media_db.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9",
    "Connection": "keep-alive",
}

# =========================
# BD por defecto (España + Marruecos)
# Ajusta los selectores si cambian las webs.
# =========================
DEFAULT_DB: Dict[str, List[Dict[str, Any]]] = {
    "España": [
        {"name": "EnergyNews", "url": "https://www.energynews.es/", "selector": "article h2.entry-title a", "base_url": None},
        {"name": "EFEVerde (Energía)", "url": "https://efeverde.com/energia/", "selector": "article h2 a", "base_url": None},
        {"name": "El Periódico de la Energía", "url": "https://elperiodicodelaenergia.com/renovables", "selector": "h3.entry-title a", "base_url": None},
        {"name": "Energías Renovables", "url": "https://www.energias-renovables.com/", "selector": "div.enrTitularNoticia a", "base_url": "https://www.energias-renovables.com"},
        {"name": "Review Energy", "url": "https://www.review-energy.com/", "selector": "div.card-title a", "base_url": None},
        {"name": "Diario de la Energía", "url": "https://www.diariodelaenergia.com/", "selector": "h3.entry-title a", "base_url": None},
        {"name": "Europa Press (Economía)", "url": "https://www.europapress.es/economia/", "selector": "div.noticiacuerpo h2 a, article h2 a", "base_url": "https://www.europapress.es"},
    ],
    "Marruecos": [
        # Nota: varios medios marroquíes están en francés o inglés.
        # Ajusta las secciones de energía si conviene.
        {"name": "Morocco World News (Energy tag)", "url": "https://www.moroccoworldnews.com/tag/energy", "selector": "article h2 a, h3 a", "base_url": "https://www.moroccoworldnews.com"},
        {"name": "Hespress (Economy)", "url": "https://www.hespress.com/economie", "selector": "article h2 a, h3 a", "base_url": "https://www.hespress.com"},
        {"name": "Le Matin (Économie)", "url": "https://lematin.ma/economies", "selector": "article h2 a, h3 a, .title a", "base_url": "https://lematin.ma"},
        {"name": "L'Economiste (Economie)", "url": "https://leconomiste.com/section/economie", "selector": "h2 a, .title a", "base_url": "https://leconomiste.com"},
        {"name": "Aujourd'hui Le Maroc (Économie)", "url": "https://aujourdhui.ma/economie", "selector": "h2 a, .entry-title a", "base_url": "https://aujourdhui.ma"},
        {"name": "TelQuel (Économie)", "url": "https://telquel.ma/rubrique/economie", "selector": "h2 a, .post-title a", "base_url": "https://telquel.ma"},
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
    # Devolver copia profunda segura
    return json.loads(json.dumps(DEFAULT_DB))

# =========================
# Caché en memoria con TTL
# Guardada como recurso para durar la sesión.
# =========================
@st.cache_resource
def get_html_cache() -> Dict[str, Tuple[float, str]]:
    """
    Devuelve un dict { url: (timestamp_epoch, html_text) }
    """
    return {}

def cache_get(url: str, ttl_sec: int) -> Optional[str]:
    cache = get_html_cache()
    item = cache.get(url)
    if not item:
        return None
    ts, html = item
    if (time.time() - ts) <= ttl_sec:
        return html
    # Expirado
    cache.pop(url, None)
    return None

def cache_put(url: str, html: str) -> None:
    cache = get_html_cache()
    cache[url] = (time.time(), html)

# =========================
# Utilidades
# =========================
def build_regex(terms: str, whole_words: bool = False, ignore_case: bool = True) -> Optional[re.Pattern]:
    cleaned = [t.strip() for t in re.split(r"[,\n;]+", terms) if t.strip()]
    if not cleaned:
        return None
    escaped = [re.escape(t) for t in cleaned]
    body = "|".join(escaped)
    if whole_words:
        body = rf"\b(?:{body})\b"
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
        return link
    if link.startswith(("http://", "https://")):
        return link
    if base_url:
        if base_url.endswith("/") and link.startswith("/"):
            return base_url.rstrip("/") + link
        if (not base_url.endswith("/")) and (not link.startswith("/")):
            return base_url + "/" + link
        return base_url + link
    return link

# =========================
# Networking asíncrono + caché
# =========================
async def fetch_html(session: aiohttp.ClientSession, url: str, timeout: int, ttl_sec: int) -> str:
    """Devuelve HTML usando caché por URL con TTL."""
    cached = cache_get(url, ttl_sec)
    if cached is not None:
        return cached

    # No está en caché -> descargar
    try:
        async with session.get(url, headers=HEADERS, timeout=timeout) as resp:
            resp.raise_for_status()
            html = await resp.text()
            cache_put(url, html)
            return html
    except Exception as e:
        # Guarda en caché negativa muy corta para evitar bucles (opcional)
        cache_put(url, "")
        raise e

async def scrape_source_async(
    session: aiohttp.ClientSession,
    source: Dict[str, Any],
    include_re: Optional[re.Pattern],
    exclude_re: Optional[re.Pattern],
    timeout: int,
    ttl_sec: int,
) -> List[Dict[str, Any]]:
    name = source.get("name")
    url = source.get("url")
    selector = source.get("selector")
    base_url = source.get("base_url") or None

    if not (name and url and selector):
        return []

    try:
        html = await fetch_html(session, url, timeout=timeout, ttl_sec=ttl_sec)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector) if selector else []
        rows: List[Dict[str, Any]] = []

        for el in elements:
            title = el.get_text(strip=True)
            href = el.get("href")
            if not href or not title:
                continue
            full_url = absolutize(href, base_url)
            if is_relevant(title, include_re, exclude_re):
                rows.append(
                    {
                        "medio": name,
                        "título": title,
                        "url": full_url,
                        "fecha_extraccion": datetime.now().strftime("%Y-%m-%d"),
                    }
                )
        return rows
    except Exception as e:
        st.session_state.setdefault("logs", []).append(f"❌ {name}: {e}")
        return []

async def run_parallel(
    sources: List[Dict[str, Any]],
    include_re: Optional[re.Pattern],
    exclude_re: Optional[re.Pattern],
    timeout: int,
    concurrency: int,
    ttl_sec: int,
    progress_cb=None,
) -> List[Dict[str, Any]]:
    """
    Ejecuta scraping en paralelo con límite de concurrencia.
    """
    connector = aiohttp.TCPConnector(limit_per_host=concurrency, ssl=False)
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession(connector=connector) as session:

        async def wrapped(src):
            async with sem:
                out = await scrape_source_async(session, src, include_re, exclude_re, timeout, ttl_sec)
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
    out = []
    for r in rows:
        key = (r.get("medio"), r.get("título"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

# =========================
# Estado inicial
# =========================
if "db" not in st.session_state:
    st.session_state.db = load_db()
if "logs" not in st.session_state:
    st.session_state.logs = []

# =========================
# Sidebar: gestión BD
# =========================
st.sidebar.header("🗂️ Base de medios por país")

all_countries = sorted(st.session_state.db.keys())
default_country = "España" if "España" in all_countries else (all_countries[0] if all_countries else "España")
country = st.sidebar.selectbox("País (para gestionar)", all_countries, index=all_countries.index(default_country))

with st.sidebar.expander(f"📜 Medios en {country} ({len(st.session_state.db.get(country, []))})", expanded=True):
    if not st.session_state.db.get(country):
        st.caption("Sin medios registrados.")
    else:
        for s in st.session_state.db[country]:
            st.markdown(f"- **{s['name']}** — [{s['url']}]({s['url']})  \n  `selector`: `{s['selector']}`  · `base_url`: `{s.get('base_url') or ''}`")

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Añadir medio")
with st.sidebar.form("form_add_source", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        new_country = st.text_input("País (opcional para crear uno nuevo)")
    with col2:
        use_current = st.checkbox("Usar país seleccionado", value=True)
    target_country = country if use_current or not new_country.strip() else new_country.strip()

    name = st.text_input("Nombre del medio", placeholder="Ej. El Periódico de la Energía")
    url = st.text_input("URL de portada/listado", placeholder="https://ejemplo.com/seccion")
    selector = st.text_input("Selector CSS de enlaces a noticias", placeholder="article h2 a")
    base_url = st.text_input("Base URL (opcional)", placeholder="https://ejemplo.com")
    submitted_add = st.form_submit_button("Añadir")

    if submitted_add:
        if not (name and url and selector and target_country):
            st.sidebar.error("Rellena país/nombre/URL/selector.")
        else:
            db = st.session_state.db
            if target_country not in db:
                db[target_country] = []
            exists = any(s["name"].lower() == name.lower() for s in db[target_country])
            if exists:
                st.sidebar.warning("Ese medio ya existe en el país seleccionado.")
            else:
                db[target_country].append(
                    {"name": name.strip(), "url": url.strip(), "selector": selector.strip(), "base_url": base_url.strip() or None}
                )
                save_db(db)
                st.sidebar.success(f"Añadido {name} a {target_country} ✅")

st.sidebar.subheader("🗑️ Eliminar medios")
if st.session_state.db.get(country):
    names = [s["name"] for s in st.session_state.db[country]]
    to_delete = st.sidebar.multiselect("Selecciona medios a eliminar", names)
    if st.sidebar.button("Eliminar seleccionados"):
        before = len(st.session_state.db[country])
        st.session_state.db[country] = [s for s in st.session_state.db[country] if s["name"] not in to_delete]
        save_db(st.session_state.db)
        after = len(st.session_state.db[country])
        st.sidebar.success(f"Eliminados {before - after} medios de {country}.")
else:
    st.sidebar.caption("No hay medios que eliminar.")

st.sidebar.markdown("---")
st.sidebar.subheader("⬆️⬇️ Importar / Exportar")
col_exp, col_imp = st.sidebar.columns(2)
with col_exp:
    st.download_button(
        "Exportar JSON",
        data=json.dumps(st.session_state.db, ensure_ascii=False, indent=2),
        file_name="media_db.json",
        mime="application/json",
        use_container_width=True,
    )
with col_imp:
    uploaded = st.file_uploader("Importar JSON", type=["json"])
    if uploaded is not None:
        try:
            imported = json.load(uploaded)
            if isinstance(imported, dict):
                st.session_state.db = imported
                save_db(imported)
                st.success("Base importada correctamente ✅")
            else:
                st.error("El archivo no tiene el formato esperado (objeto JSON).")
        except Exception as e:
            st.error(f"No se pudo importar: {e}")

# =========================
# Controles de búsqueda
# =========================
st.header("🔍 Búsqueda")

colA, colB, colC = st.columns([2, 2, 1])
with colA:
    include_terms = st.text_input(
        "Términos a incluir (coma, punto y coma o saltos de línea):",
        placeholder="ej.: energía, renovables, solar, eólica",
    )
with colB:
    exclude_terms = st.text_input("Términos a excluir (opcional):", placeholder="ej.: subvención, guerra, militar")
with colC:
    whole_words = st.checkbox("Coincidencia por palabra", value=False)
    ignore_case = st.checkbox("Ignorar mayúsc./minúsc.", value=True)

with st.expander("⚙️ Opciones avanzadas"):
    search_country = st.selectbox(
        "País a buscar",
        sorted(st.session_state.db.keys()),
        index=sorted(st.session_state.db.keys()).index(default_country),
    )
    current_sources = st.session_state.db.get(search_country, [])
    preselect = [s["name"] for s in current_sources]
    selected_names = st.multiselect(
        f"Medios de {search_country} a incluir",
        options=preselect,
        default=preselect,
    )
    timeout = st.slider("Timeout por petición (seg.)", min_value=5, max_value=40, value=15, step=1)
    concurrency = st.slider("Concurrencia (simultáneos)", min_value=2, max_value=20, value=8, step=1)
    ttl_sec = st.slider("TTL caché (seg.)", min_value=60, max_value=3600, value=900, step=30)
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
        with st.spinner("Buscando en paralelo…"):
            # Ejecutamos el scraping en paralelo
            rows_all: List[Dict[str, Any]] = asyncio.run(
                run_parallel(
                    sources=sources,
                    include_re=include_re,
                    exclude_re=exclude_re,
                    timeout=timeout,
                    concurrency=concurrency,
                    ttl_sec=ttl_sec,
                    progress_cb=add_log_line,  # acumula en logs
                )
            )

        elapsed = time.time() - start
        rows_all = dedupe_news(rows_all)

        if not rows_all:
            st.warning("🚫 No se encontraron noticias con los criterios dados.")
        else:
            df = pd.DataFrame(rows_all)
            st.success(f"✅ {len(df)} noticias encontradas en {search_country} (en {elapsed:.1f}s).")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.write("### Resultados")
            for _, row in df.iterrows():
                st.markdown(
                    f"""
                    <div style="border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;margin-bottom:10px;">
                      <p style="margin:0;color:#6b7280;font-size:13px;">📰 <strong>{row['medio']}</strong></p>
                      <p style="margin:6px 0;">
                        <a href="{row['url']}" target="_blank" style="font-size:16px;text-decoration:none;">
                          {row['título']}
                        </a>
                      </p>
                      <p style="margin:0;color:#6b7280;font-size:12px;">📅 {row['fecha_extraccion']}</p>
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
