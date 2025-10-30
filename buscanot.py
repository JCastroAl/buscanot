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
# Base por defecto con patrones de hemeroteca + idioma por medio
DEFAULT_DB: Dict[str, List[Dict[str, Any]]] = {
    "España": [
        {"name": "El País", "url": "https://elpais.com/", "selector": "article h2 a, h3 a, .headline a", "archive_selector": "h2 a, h3 a, .c_t a", "base_url": "https://elpais.com", "archive_pattern": "https://elpais.com/hemeroteca/{yyyy}-{mm}-{dd}/", "lang": "es"},
        {"name": "El Mundo", "url": "https://www.elmundo.es/", "selector": "article h2 a, h3 a, .ue-c-cover-content__link", "archive_selector": "h2 a, h3 a, .ue-c-cover-content__link, .mod-title a, .headline a", "base_url": "https://www.elmundo.es", "archive_pattern": "https://www.elmundo.es/elmundo/hemeroteca/{yyyy}/{mm}/{dd}/noticias.html", "lang": "es"},
        {"name": "ABC", "url": "https://www.abc.es/", "selector": "article h2 a, h3 a, .titular a", "archive_selector": None, "base_url": "https://www.abc.es", "archive_pattern": None, "lang": "es"},
        {"name": "La Vanguardia", "url": "https://www.lavanguardia.com/", "selector": "article h2 a, h3 a, .headline a", "archive_selector": None, "base_url": "https://www.lavanguardia.com", "archive_pattern": None, "lang": "es"},
        {"name": "El Confidencial", "url": "https://www.elconfidencial.com/", "selector": "article h2 a, h3 a, .news__title a", "archive_selector": None, "base_url": "https://www.elconfidencial.com", "lang": "es"},
        {"name": "20minutos", "url": "https://www.20minutos.es/", "selector": "article h2 a, h3 a, .headline a", "archive_selector": None, "base_url": "https://www.20minutos.es", "lang": "es"},
        {"name": "RTVE", "url": "https://www.rtve.es/noticias/", "selector": "article h2 a, h3 a, .headline a", "archive_selector": None, "base_url": "https://www.rtve.es/noticias/", "lang": "es"},
        {"name": "La Razón", "url": "https://www.larazon.es/", "selector": "article h2 a, h3 a, .headline a", "archive_selector": None, "base_url": "https://www.larazon.es/", "lang": "es"},
        {"name": "El Periodico de Catalunya", "url": "https://www.elperiodico.com/es/", "selector": "article h2 a, h3 a, .headline a", "archive_selector": None, "base_url": "https://www.elperiodico.com/", "lang": "es"},
    ],
    "Marruecos": [
        {"name": "Hespress (AR)", "url": "https://www.hespress.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.hespress.com", "lang": "ar"},
        {"name": "Hespress (EN)", "url": "https://en.hespress.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://en.hespress.com", "lang": "en"},
        {"name": "Le Matin", "url": "https://lematin.ma/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://lematin.ma", "lang": "fr"},
        {"name": "L'Economiste", "url": "https://leconomiste.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://leconomiste.com", "lang": "fr"},
        {"name": "TelQuel", "url": "https://telquel.ma/", "selector": "article h2 a, h3 a, .post-title a", "base_url": "https://telquel.ma", "lang": "fr"},
        {"name": "Aujourd'hui Le Maroc", "url": "https://aujourdhui.ma/", "selector": "article h2 a, h3 a, .entry-title a", "base_url": "https://aujourdhui.ma", "lang": "fr"},
        {"name": "Médias24", "url": "https://medias24.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://medias24.com", "lang": "fr"},
        {"name": "Morocco World News", "url": "https://www.moroccoworldnews.com/", "selector": "article h2 a, h3 a, .card-title a", "base_url": "https://www.moroccoworldnews.com", "lang": "en"},
    ],
    "Francia": [
        {"name": "Le Monde", "url": "https://www.lemonde.fr/", "selector": "article h2 a, h3 a, .article__title a", "base_url": "https://www.lemonde.fr", "lang": "fr"},
        {"name": "Le Figaro", "url": "https://www.lefigaro.fr/", "selector": "article h2 a, h3 a, .fig-headline a", "base_url": "https://www.lefigaro.fr", "lang": "fr"},
        {"name": "Libération", "url": "https://www.liberation.fr/", "selector": "article h2 a, h3 a, .article-card a", "base_url": "https://www.liberation.fr", "lang": "fr"},
        {"name": "Le Parisien", "url": "https://www.leparisien.fr/", "selector": "article h2 a, h3 a, .article-title a", "base_url": "https://www.leparisien.fr", "lang": "fr"},
        {"name": "Les Échos", "url": "https://www.lesechos.fr/", "selector": "article h2 a, h3 a, .teaser__title a", "base_url": "https://www.lesechos.fr", "lang": "fr"},
        {"name": "La Croix", "url": "https://www.la-croix.com/", "selector": "article h2 a, h3 a, .article__title a", "base_url": "https://www.la-croix.com", "lang": "fr"},
        {"name": "Ouest-France", "url": "https://www.ouest-france.fr/", "selector": "article h2 a, h3 a, .teaser-title a", "base_url": "https://www.ouest-france.fr", "lang": "fr"},
    ],
    "Portugal": [
        {"name": "Público", "url": "https://www.publico.pt/", "selector": "article h2 a, h3 a, .headline a", "base_url": "https://www.publico.pt", "lang": "pt"},
        {"name": "Diário de Notícias", "url": "https://www.dn.pt/", "selector": "article h2 a, h3 a, .card-title a", "base_url": "https://www.dn.pt", "lang": "pt"},
        {"name": "Expresso", "url": "https://expresso.pt/", "selector": "article h2 a, h3 a, .article-title a", "base_url": "https://expresso.pt", "lang": "pt"},
        {"name": "Observador", "url": "https://observador.pt/", "selector": "article h2 a, h3 a, .headline a", "base_url": "https://observador.pt", "lang": "pt"},
        {"name": "Correio da Manhã", "url": "https://www.cmjornal.pt/", "selector": "article h2 a, h3 a, .tit a", "base_url": "https://www.cmjornal.pt", "lang": "pt"},
        {"name": "Jornal de Notícias", "url": "https://www.jn.pt/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.jn.pt", "lang": "pt"},
    ],
    "Andorra": [
        {"name": "Diari d'Andorra", "url": "https://www.diariandorra.ad/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.diariandorra.ad", "lang": "ca"},
        {"name": "Bondia Andorra", "url": "https://www.bondia.ad/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.bondia.ad", "lang": "ca"},
        {"name": "Altaveu", "url": "https://www.altaveu.com/", "selector": "article h2 a, h3 a, .title a", "base_url": "https://www.altaveu.com", "lang": "ca"},
    ],
    "Alemania": [
        {"name": "Frankfurter Allgemeine (FAZ)", "url": "https://www.faz.net/", "selector": "article h2 a, h3 a, .headline a", "base_url": "https://www.faz.net", "lang": "de"},
        {"name": "Süddeutsche Zeitung", "url": "https://www.sueddeutsche.de/", "selector": "article h2 a, h3 a, .sz-article__title a", "base_url": "https://www.sueddeutsche.de", "lang": "de"},
        {"name": "WELT", "url": "https://www.welt.de/", "selector": "article h2 a, h3 a, .c-teaser__headline a", "base_url": "https://www.welt.de", "lang": "de"},
        {"name": "Der Spiegel", "url": "https://www.spiegel.de/", "selector": "article h2 a, h3 a, .leading-article a", "base_url": "https://www.spiegel.de", "lang": "de"},
        {"name": "Die Zeit", "url": "https://www.zeit.de/index", "selector": "article h2 a, h3 a, .zon-teaser-standard__title a", "base_url": "https://www.zeit.de", "lang": "de"},
        {"name": "BILD", "url": "https://www.bild.de/", "selector": "article h2 a, h3 a, .headline a", "base_url": "https://www.bild.de", "lang": "de"},
        {"name": "tagesschau (ARD)", "url": "https://www.tagesschau.de/", "selector": "article h2 a, h3 a, .teaser__title a", "base_url": "https://www.tagesschau.de", "lang": "de"},
        {"name": "Handelsblatt", "url": "https://www.handelsblatt.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.handelsblatt.com", "lang": "de"},
    ],
    "Austria": [
        {"name": "Der Standard", "url": "https://www.derstandard.at/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.derstandard.at", "lang": "de"},
        {"name": "Die Presse", "url": "https://www.diepresse.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.diepresse.com", "lang": "de"},
        {"name": "Kurier", "url": "https://kurier.at/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://kurier.at", "lang": "de"},
        {"name": "Kronen Zeitung", "url": "https://www.krone.at/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.krone.at", "lang": "de"},
    ],
    "Bélgica": [
        {"name": "Le Soir", "url": "https://www.lesoir.be/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.lesoir.be", "lang": "fr"},
        {"name": "La Libre Belgique", "url": "https://www.lalibre.be/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.lalibre.be", "lang": "fr"},
        {"name": "De Standaard", "url": "https://www.standaard.be/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.standaard.be", "lang": "nl"},
        {"name": "Het Laatste Nieuws", "url": "https://www.hln.be/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.hln.be", "lang": "nl"},
        {"name": "De Morgen", "url": "https://www.demorgen.be/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.demorgen.be", "lang": "nl"},
    ],
    "Bosnia y Herzegovina": [
        {"name": "Dnevni Avaz", "url": "https://www.avaz.ba/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.avaz.ba", "lang": "bs"},
        {"name": "Oslobođenje", "url": "https://www.oslobodjenje.ba/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.oslobodjenje.ba", "lang": "bs"},
        {"name": "Nezavisne novine", "url": "https://www.nezavisne.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.nezavisne.com", "lang": "sr"},
    ],
    "Bulgaria": [
        {"name": "Dnevnik", "url": "https://www.dnevnik.bg/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.dnevnik.bg", "lang": "bg"},
        {"name": "24 Chasa", "url": "https://www.24chasa.bg/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.24chasa.bg", "lang": "bg"},
        {"name": "Trud", "url": "https://trud.bg/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://trud.bg", "lang": "bg"},
        {"name": "Sega", "url": "https://www.segabg.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.segabg.com", "lang": "bg"},
    ],
    "Chipre": [
        {"name": "Cyprus Mail", "url": "https://cyprus-mail.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://cyprus-mail.com", "lang": "en"},
        {"name": "Politis", "url": "https://politis.com.cy/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://politis.com.cy", "lang": "el"},
        {"name": "Phileleftheros", "url": "https://www.philenews.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.philenews.com", "lang": "el"},
    ],
    "Croacia": [
        {"name": "Jutarnji list", "url": "https://www.jutarnji.hr/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.jutarnji.hr", "lang": "hr"},
        {"name": "Večernji list", "url": "https://www.vecernji.hr/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.vecernji.hr", "lang": "hr"},
        {"name": "Slobodna Dalmacija", "url": "https://slobodnadalmacija.hr/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://slobodnadalmacija.hr", "lang": "hr"},
    ],
    "Dinamarca": [
        {"name": "Politiken", "url": "https://politiken.dk/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://politiken.dk", "lang": "da"},
        {"name": "Berlingske", "url": "https://www.berlingske.dk/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.berlingske.dk", "lang": "da"},
        {"name": "Jyllands-Posten", "url": "https://jyllands-posten.dk/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://jyllands-posten.dk", "lang": "da"},
        {"name": "Ekstra Bladet", "url": "https://ekstrabladet.dk/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://ekstrabladet.dk", "lang": "da"},
        {"name": "Børsen", "url": "https://borsen.dk/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://borsen.dk", "lang": "da"},
    ],
    "Eslovaquia": [
        {"name": "SME", "url": "https://www.sme.sk/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.sme.sk", "lang": "sk"},
        {"name": "Pravda", "url": "https://www.pravda.sk/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.pravda.sk", "lang": "sk"},
        {"name": "Hospodárske noviny", "url": "https://www.hnonline.sk/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.hnonline.sk", "lang": "sk"},
    ],
    "Eslovenia": [
        {"name": "Delo", "url": "https://www.delo.si/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.delo.si", "lang": "sl"},
        {"name": "Dnevnik", "url": "https://www.dnevnik.si/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.dnevnik.si", "lang": "sl"},
        {"name": "Večer", "url": "https://www.vecer.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.vecer.com", "lang": "sl"},
    ],
    "Estonia": [
        {"name": "Postimees", "url": "https://www.postimees.ee/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.postimees.ee", "lang": "et"},
        {"name": "Eesti Päevaleht", "url": "https://www.delfi.ee/ekspress/epl", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.delfi.ee", "lang": "et"},
        {"name": "Õhtuleht", "url": "https://www.ohtuleht.ee/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.ohtuleht.ee", "lang": "et"},
    ],
    "Finlandia": [
        {"name": "Helsingin Sanomat", "url": "https://www.hs.fi/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.hs.fi", "lang": "fi"},
        {"name": "Ilta-Sanomat", "url": "https://www.is.fi/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.is.fi", "lang": "fi"},
        {"name": "Iltalehti", "url": "https://www.iltalehti.fi/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.iltalehti.fi", "lang": "fi"},
        {"name": "Kauppalehti", "url": "https://www.kauppalehti.fi/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.kauppalehti.fi", "lang": "fi"},
    ],
    "Grecia": [
        {"name": "Kathimerini", "url": "https://www.kathimerini.gr/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.kathimerini.gr", "lang": "el"},
        {"name": "Ta Nea", "url": "https://www.tanea.gr/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.tanea.gr", "lang": "el"},
        {"name": "To Vima", "url": "https://www.tovima.gr/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.tovima.gr", "lang": "el"},
        {"name": "Proto Thema", "url": "https://www.protothema.gr/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.protothema.gr", "lang": "el"},
        {"name": "Athens-Macedonian News Agency", "url": "https://www.amna.gr/en", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.amna.gr", "lang": "en"},
    ],
    "Hungría": [
        {"name": "Népszava", "url": "https://nepszava.hu/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://nepszava.hu", "lang": "hu"},
        {"name": "Magyar Nemzet", "url": "https://magyarnemzet.hu/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://magyarnemzet.hu", "lang": "hu"},
        {"name": "Világgazdaság", "url": "https://www.vg.hu/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.vg.hu", "lang": "hu"},
        {"name": "Blikk", "url": "https://www.blikk.hu/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.blikk.hu", "lang": "hu"},
    ],
    "Irlanda": [
        {"name": "The Irish Times", "url": "https://www.irishtimes.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.irishtimes.com", "lang": "en"},
        {"name": "Irish Independent", "url": "https://www.independent.ie/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.independent.ie", "lang": "en"},
        {"name": "Irish Examiner", "url": "https://www.irishexaminer.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.irishexaminer.com", "lang": "en"},
        {"name": "TheJournal.ie", "url": "https://www.thejournal.ie/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.thejournal.ie", "lang": "en"},
    ],
    "Islandia": [
        {"name": "Morgunblaðið", "url": "https://www.mbl.is/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.mbl.is", "lang": "is"},
        {"name": "Fréttablaðið", "url": "https://www.frettabladid.is/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.frettabladid.is", "lang": "is"},
        {"name": "Vísir", "url": "https://www.visir.is/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.visir.is", "lang": "is"},
    ],
    "Italia": [
        {"name": "Corriere della Sera", "url": "https://www.corriere.it/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.corriere.it", "lang": "it"},
        {"name": "la Repubblica", "url": "https://www.repubblica.it/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.repubblica.it", "lang": "it"},
        {"name": "La Stampa", "url": "https://www.lastampa.it/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.lastampa.it", "lang": "it"},
        {"name": "Il Messaggero", "url": "https://www.ilmessaggero.it/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.ilmessaggero.it", "lang": "it"},
        {"name": "Il Sole 24 Ore", "url": "https://www.ilsole24ore.com/", "selector": "article h2 a, h3 a, .headline a, .title a", "base_url": "https://www.ilsole24ore.com", "lang": "it"},
    ],
"Estados Unidos": [
        {"name": "The Boston Globe", "url": "https://www.bostonglobe.com/", "selector": "article h2 a, h3 a, .headline a, .story a", "archive_selector": "article h2 a, h3 a, .headline a, .story a", "base_url": "https://www.bostonglobe.com", "archive_pattern": None, "lang": "en"},
        {"name": "Los Angeles Times", "url": "https://www.latimes.com/", "selector": "article h2 a, h3 a, .promo-title, .story-title a", "archive_selector": "article h2 a, h3 a, .promo-title, .story-title a", "base_url": "https://www.latimes.com", "archive_pattern": "https://www.latimes.com/archives", "lang": "en"},
        {"name": "USA TODAY", "url": "https://www.usatoday.com/", "selector": "article h2 a, h3 a, a.gnt_m_flm_a, a.gnt_m_flm_bt", "archive_selector": None, "base_url": "https://www.usatoday.com", "archive_pattern": None, "lang": "en"},
        {"name": "The New York Times", "url": "https://www.nytimes.com/", "selector": "article h2 a, h3 a, .css-1g7m0tk a, .story-wrapper a", "archive_selector": "a", "base_url": "https://www.nytimes.com", "archive_pattern": "https://www.nytimes.com/sitemap/{yyyy}/{mm}/{dd}/", "lang": "en"},
        {"name": "NPR", "url": "https://www.npr.org/", "selector": "article h2 a, h3 a, .title a, .story-text a", "archive_selector": "article h2 a, h3 a, .title a, .story-text a", "base_url": "https://www.npr.org", "archive_pattern": "https://www.npr.org/sections/news/archive?date={yyyy}-{mm}-{dd}", "lang": "en"},
        {"name": "Bloomberg", "url": "https://www.bloomberg.com/feed/podcast?type=story", "selector": "article h2 a, h3 a, a[data-testid='Headline'], .story-package-module__story__headline-link, .headline__link", "archive_selector": "article h2 a, h3 a, a[data-testid='Headline'], .story-package-module__story__headline-link, .headline__link", "base_url": "https://www.bloomberg.com", "archive_pattern": "https://www.bloomberg.com/archive/{yyyy}-{mm}-{dd}", "lang": "en", "needs_headers": True, "rss_fallback": "https://www.bloomberg.com/feed/podcast?type=story"},
        {"name": "The Washington Post", "url": "https://www.washingtonpost.com/?outputType=amp", "selector": "article h2 a, h3 a, .wpds-c-card__title a, a[data-qa='headline-link']", "archive_selector": "a", "base_url": "https://www.washingtonpost.com", "archive_pattern": "https://www.washingtonpost.com/sitemap/{yyyy}/", "lang": "en", "use_amp": True, "disable_robots": True, "rss_fallback": "https://feeds.washingtonpost.com/rss/national"},
        {"name": "AP News", "url": "https://apnews.com/", "selector": "article h2 a, h3 a, .headline a", "archive_selector": "article h2 a, h3 a, .headline a", "base_url": "https://apnews.com", "archive_pattern": "https://apnews.com/hub/archive", "lang": "en"},
        {"name": "ABC News", "url": "https://abcnews.go.com/", "selector": "article h2 a, h3 a, .headline a, .story-title a", "archive_selector": None, "base_url": "https://abcnews.go.com", "archive_pattern": None, "lang": "en"},
        {"name": "CNN", "url": "https://edition.cnn.com/", "selector": "article h2 a, h3 a, .cd__headline a", "archive_selector": None, "base_url": "https://edition.cnn.com", "archive_pattern": None, "lang": "en"},
        {"name": "CBS News", "url": "https://www.cbsnews.com/", "selector": "article h2 a, h3 a, .headline a, .feature-list__link", "archive_selector": None, "base_url": "https://www.cbsnews.com", "archive_pattern": None, "lang": "en"},
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
    """
    Traduce cada término, manteniendo la marca (exacto?).
    """
    out: List[Tuple[str,bool]] = []
    for t, ex in terms:
        tt = await translate_text(session, t, target_lang)
        if tt and tt.lower() != t.lower():
            out.append((tt, ex))
    return out

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

# =========================
# Scraper (con traducción por medio)
# =========================
async def scrape_source_async(
    session: aiohttp.ClientSession,
    source: Dict[str, Any],
    include_terms_raw: str,
    exclude_terms_raw: str,
    user_whole_words: bool,
    ignore_case: bool,
    translate_per_source: bool,
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

    if not (name and url):
        return []

    # 1) Construcción de regex por medio
    inc_terms = split_terms(include_terms_raw)
    exc_terms = split_terms(exclude_terms_raw)

    if translate_per_source and lang:
        # Traduce manteniendo exactitud si el término venía entre comillas
        inc_trans = await translate_terms_list(session, inc_terms, lang)
        exc_trans = await translate_terms_list(session, exc_terms, lang)
        # Unión (evita duplicados case-insensitive)
        def merge(a: List[Tuple[str,bool]], b: List[Tuple[str,bool]]) -> List[Tuple[str,bool]]:
            seen = set()
            out: List[Tuple[str,bool]] = []
            for t in a + b:
                key = (t[0].lower(), t[1])
                if key not in seen:
                    seen.add(key)
                    out.append(t)
            return out
        inc_terms = merge(inc_terms, inc_trans)
        exc_terms = merge(exc_terms, exc_trans)

    # Coincidencia por palabra sólo si idioma latino
    effective_whole_words = user_whole_words and is_latin_lang(lang)

    include_re = build_regex_from_terms(inc_terms, whole_words=effective_whole_words, ignore_case=ignore_case) if inc_terms else None
    exclude_re = build_regex_from_terms(exc_terms, whole_words=effective_whole_words, ignore_case=ignore_case) if exc_terms else None

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
            past_days.append(d)  # incluye también hoy
    else:
        include_today = True

    # 2) ARCHIVOS (pasado)
    if past_days:
        archive_urls = iter_archive_urls_for_dates(source, past_days)
        if archive_urls and selector_archive:
            for page in archive_urls:
                html = await fetch_html(session, page, headers=headers, timeout=timeout, ttl_sec=ttl_sec, neg_ttl_sec=neg_ttl_sec, respect_robots=respect_robots)
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

    # 3) HOY (RSS + portada)
    if include_today:
        rss = await try_fetch_rss(session, url, headers, timeout)
        if rss:
            for title, href, dt in rss:
                full_url = absolutize(href, base_url or url)
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
        if selector_home:
            html = await fetch_html(session, url, headers=headers, timeout=timeout, ttl_sec=ttl_sec, neg_ttl_sec=neg_ttl_sec, respect_robots=respect_robots)
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
    include_terms_raw: str,
    exclude_terms_raw: str,
    user_whole_words: bool,
    ignore_case: bool,
    translate_per_source: bool,
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
                    session, src,
                    include_terms_raw, exclude_terms_raw,
                    user_whole_words, ignore_case, translate_per_source,
                    timeout, ttl_sec, neg_ttl_sec, headers, respect_robots,
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
if "logs" not in st.session_state:
    st.session_state.logs = []
if "search_country" not in st.session_state:
    all_keys = sorted(st.session_state.db.keys())
    st.session_state.search_country = "España" if "España" in all_keys else (all_keys[0] if all_keys else "")

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
st.sidebar.header("🗂️ LISTADO DE MEDIOS")

all_countries = sorted(st.session_state.db.keys())
default_country = "España" if "España" in all_countries else (all_countries[0] if all_countries else "España")
country = st.sidebar.selectbox("Escoge una nación", all_countries, index=all_countries.index(default_country))

with st.sidebar.expander(
    f"📜 Medios en {country} ({len(st.session_state.db.get(country, []))})",
    expanded=True
):
    medios = st.session_state.db.get(country, [])
    if not medios:
        st.caption("Sin medios registrados.")
    else:
        cols = st.columns(2)
        for i, s in enumerate(medios):
            with cols[i % 2]:
                st.link_button(s["name"], s["url"], use_container_width=True)
            
st.sidebar.markdown("---")

# =========================
# Controles de búsqueda
# =========================
# (1) País a buscar
search_country = st.selectbox(
    "País a buscar",
    sorted(st.session_state.db.keys()),
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
        st.date_input("Periodo (inicio y fin)", key="date_range", value=st.session_state.date_range, format="YYYY-MM-DD")
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
            rows_all: List[Dict[str, Any]] = asyncio.run(
                run_parallel(
                    sources=sources,
                    include_terms_raw=include_terms,
                    exclude_terms_raw=exclude_terms,
                    user_whole_words=whole_words,
                    ignore_case=ignore_case,
                    translate_per_source=translate_per_source,
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
