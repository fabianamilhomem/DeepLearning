# -*- coding: utf-8 -*-
"""
Raspador de dispositivos de "Sentença" no PJe/TJDFT (produção, enxuto)

Entrada:
  data/tjdft_saude_ids_sentenca.csv  (do coletor de produção)

Saída:
  data/tjdft_saude_sentencas_texto.csv
  data/metrics_scraper.json
  data/log_scraper.jsonl
  data/docs_pje/  (PDFs baixados, quando necessário)

Parâmetros (env):
  TARGET_COUNT            -> alvo mínimo de sentenças válidas (padrão: 500)
  GRAUS_FILTER            -> filtrar graus (ex.: "G1,JE,TR,G2") [opcional]
  MIN_CHARS_DISPOSITIVO   -> tamanho mínimo do texto do dispositivo (padrão: 300)
  RATE_MIN / RATE_MAX     -> atraso aleatório entre requests (padrão: 0.8 / 1.8)
  REQUEST_TIMEOUT         -> timeout por request (padrão: 45)

Observações:
- Tenta padrões de URL do PJe/TJDFT para abrir o detalhe por número CNJ.
- Extrai o dispositivo por heurística de marcadores; se houver PDF, faz fallback.
"""

import os
import re
import time
import json
import random
import traceback
import unicodedata
from typing import Optional, Dict, Any, List, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader

# ----------------------------
# Configurações (produção)
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data"))
DOC_DIR = os.path.join(DATA_DIR, "docs_pje")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)

IN_CSV = os.path.join(DATA_DIR, "tjdft_saude_ids_sentenca.csv")
OUT_CSV = os.path.join(DATA_DIR, "tjdft_saude_sentencas_texto.csv")
LOG_PATH = os.path.join(DATA_DIR, "log_scraper.jsonl")
METRICS_PATH = os.path.join(DATA_DIR, "metrics_scraper.json")

TARGET_COUNT = int(os.getenv("TARGET_COUNT", "500"))
GRAUS_FILTER = [g.strip() for g in os.getenv("GRAUS_FILTER", "").split(",") if g.strip()]
MIN_CHARS = int(os.getenv("MIN_CHARS_DISPOSITIVO", "300"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))
RATE_MIN = float(os.getenv("RATE_MIN", "0.8"))
RATE_MAX = float(os.getenv("RATE_MAX", "1.8"))

# Cabeçalhos básicos
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
}

# Padrões de URL (PJe/TJDFT) — variações possíveis no portal
PJE_URL_PATTERNS = [
    # Detalhe por número (padrões comuns em instâncias do PJe)
    "https://pje.tjdft.jus.br/consultapublica/ConsultaPublica/ProcessoConsultaPublica.jsf?numeroProcesso={num}",
    "https://pje.tjdft.jus.br/consultapublica/ConsultaPublica/ProcessoConsultaPublica.jsf?numero={num}",
    "https://pje.tjdft.jus.br/consultapublica/ConsultaPublica/listView.seam?numero={num}",
    # Variante sem querystring (lista → clique) – na prática exigiria fluxo JSF;
    # incluída para documentação e eventual uso futuro.
]

# Heurísticas de marcadores para extrair o "dispositivo"
DISP_PATTERNS = [
    r"\b(i[s|sto]\s+posto|isso\s+posto)\b",
    r"\bantec?e\s+o\s+exposto\b",
    r"\bante\s+o\s+exposto\b",
    r"\bdispositivo[s]?\b",
    r"\bdecido\b",
    r"\bjulgo\b",
    r"\bcondeno\b",
    r"\bdefiro\b",
    r"\bindefiro\b",
    r"\bhomologo\b",
    r"\bd[eé]claro\b",
    r"\bresolvo\b"
]

# ----------------------------
# Utilitários
# ----------------------------
def log_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def polite_sleep():
    time.sleep(random.uniform(RATE_MIN, RATE_MAX))

def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_dispositivo(text: str, window_chars: int = 1800) -> Optional[str]:
    t = text
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    for pat in DISP_PATTERNS:
        m = re.search(pat, t, flags=re.I)
        if m:
            start = max(0, m.start())
            end = min(len(t), start + window_chars)
            return normalize_ws(t[start:end])
    # fallback: pega o "final" do texto
    tail = t[-2000:].strip()
    return normalize_ws(tail) if tail else None

def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.max_redirects = 5
    return s

# ----------------------------
# Core de raspagem
# ----------------------------
def try_direct_detail(session: requests.Session, numero: str) -> Tuple[Optional[str], Optional[BeautifulSoup]]:
    """
    Tenta abrir URLs diretas de detalhe no PJe por número (CNJ).
    Retorna (url, soup) em caso de sucesso.
    """
    for pat in PJE_URL_PATTERNS:
        url = pat.format(num=numero)
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            polite_sleep()
            if r.status_code == 200 and r.text and len(r.text) > 1000:
                soup = BeautifulSoup(r.text, "lxml")
                # Heurística simples: página tem “Processo”, “Movimentações”, etc.
                if soup.find(text=re.compile(r"Processo|Movimenta", re.I)):
                    return url, soup
        except Exception as e:
            log_jsonl(LOG_PATH, {"step": "try_direct_detail", "numero": numero, "url": url,
                                 "error": str(e)})
            polite_sleep()
            continue
    return None, None

def find_pdf_link(soup: BeautifulSoup) -> Optional[str]:
    """
    Procura link para documento/PDF do teor (ex.: “inteiro teor”, “sentença”).
    """
    for a in soup.find_all("a", href=True):
        label = " ".join(a.get_text(strip=True).split())
        href = a["href"]
        if re.search(r"\.pdf(\?|$)", href, flags=re.I) or \
           re.search(r"inteiro\s+teor|senten[cç]a|decis[aã]o", label, flags=re.I):
            return href
    return None

def absolutize(base_url: str, href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        # raiz do host
        from urllib.parse import urlparse, urlunparse
        p = urlparse(base_url)
        return urlunparse((p.scheme, p.netloc, href, "", "", ""))
    # relativo ao caminho
    from urllib.parse import urljoin
    return urljoin(base_url, href)

def extract_text_from_html(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrai (magistrado, texto_dispositivo) de uma página HTML de detalhe.
    """
    full_text = []
    # áreas plausíveis
    containers = soup.find_all(["div", "section", "article"])
    for c in containers:
        txt = c.get_text("\n", strip=True)
        if txt and len(txt) > 200:
            full_text.append(txt)
    joined = "\n\n".join(full_text)

    # magistrado (heurístico)
    mag = None
    # padrões: Relator(a), Juiz(a), Desembargador(a)
    m = re.search(r"(Relator(?:a)?|Ju[ií]z(?:a)?|Desembargador(?:a)?)\s*:\s*(.+)", joined, flags=re.I)
    if m:
        mag = m.group(2).split("\n")[0].strip()

    disp = extract_dispositivo(joined) if joined else None
    return mag, disp

def download_pdf(session: requests.Session, url: str, numero: str) -> Optional[str]:
    """
    Baixa PDF para pasta local e retorna caminho do arquivo.
    """
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        polite_sleep()
        if r.status_code == 200 and r.content and len(r.content) > 1500:
            safe_num = re.sub(r"[^\d]", "", numero)[-20:]
            fname = f"{safe_num}_{int(time.time())}.pdf"
            fpath = os.path.join(DOC_DIR, fname)
            with open(fpath, "wb") as f:
                f.write(r.content)
            return fpath
    except Exception as e:
        log_jsonl(LOG_PATH, {"step": "download_pdf", "numero": numero, "url": url, "error": str(e)})
    return None

def pdf_to_text(pdf_path: str) -> Optional[str]:
    """
    Extrai texto de PDF com PyPDF2.
    """
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        log_jsonl(LOG_PATH, {"step": "pdf_to_text", "pdf": pdf_path, "error": str(e)})
        return None

def process_one(session: requests.Session, numero: str) -> Dict[str, Any]:
    """
    Fluxo compacto:
      1) Tenta abrir detalhe por padrões de URL.
      2) Extrai magistrado + dispositivo do HTML.
      3) Se não houver, tenta PDF (inteiro teor).
    """
    result = {
        "numero": numero,
        "fonte_url": None,
        "magistrado": None,
        "texto_dispositivo": None,
        "status": "detail_not_found",
    }
    try:
        url, soup = try_direct_detail(session, numero)
        if not url or not soup:
            return result

        result["fonte_url"] = url
        mag, disp = extract_text_from_html(soup)

        # Se não veio, tenta PDF
        if not disp or len(disp) < MIN_CHARS:
            pdf_href = find_pdf_link(soup)
            if pdf_href:
                pdf_url = absolutize(url, pdf_href)
                pdf_path = download_pdf(session, pdf_url, numero)
                if pdf_path:
                    txt = pdf_to_text(pdf_path) or ""
                    disp2 = extract_dispositivo(txt) if txt else None
                    if disp2 and len(disp2) >= MIN_CHARS:
                        result["texto_dispositivo"] = disp2
                        result["magistrado"] = mag
                        result["status"] = "ok_pdf"
                        return result

        # Se HTML serviu
        if disp and len(disp) >= MIN_CHARS:
            result["texto_dispositivo"] = disp
            result["magistrado"] = mag
            result["status"] = "ok_html"
            return result

        # Caso geral
        result["magistrado"] = mag
        result["status"] = "short_or_notfound"
        return result

    except Exception as e:
        log_jsonl(LOG_PATH, {"step": "process_one_exception", "numero": numero,
                             "error": str(e), "trace": traceback.format_exc()})
        result["status"] = "exception"
        return result

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {IN_CSV}")
    df_ids = pd.read_csv(IN_CSV)

    # Filtro opcional por graus, se desejado
    if GRAUS_FILTER and "grau" in df_ids.columns:
        df_ids = df_ids[df_ids["grau"].astype(str).isin(GRAUS_FILTER)].copy()

    # Resume: se já existir saída, evitamos reprocessar
    done = set()
    if os.path.exists(OUT_CSV):
        df_done = pd.read_csv(OUT_CSV)
        if "numero" in df_done.columns:
            done = set(df_done["numero"].astype(str).tolist())

    session = get_session()

    results: List[Dict[str, Any]] = []
    valid_count = 0
    counters: Dict[str, int] = {}
    t0 = time.time()

    for i, row in df_ids.iterrows():
        numero = str(row.get("numero") or row.get("numeroProcesso") or "").strip()
        if not numero:
            continue
        if numero in done:
            continue

        print(f"[{i+1}/{len(df_ids)}] {numero} ...")
        res = process_one(session, numero)

        # Aplica mínimos de qualidade
        txt = res.get("texto_dispositivo") or ""
        if txt and len(txt) >= MIN_CHARS:
            valid = True
            valid_count += 1
        else:
            valid = False

        # agrega metadados do CSV de entrada (se existirem)
        for col in ("grau", "orgao_julgador", "classe", "data_ajuizamento", "data_sentenca"):
            if col in df_ids.columns:
                res[col] = row.get(col)

        results.append(res)
        counters[res["status"]] = counters.get(res["status"], 0) + 1

        # Checkpoint a cada 50
        if len(results) % 50 == 0:
            df_out = pd.DataFrame(results)
            mode = "a" if os.path.exists(OUT_CSV) else "w"
            header = not os.path.exists(OUT_CSV)
            df_out.to_csv(OUT_CSV, index=False, encoding="utf-8", mode=mode, header=header)
            results = []
            print(f"   -> checkpoint salvo. válidas até agora: {valid_count}")

        # Para quando atingir o alvo
        if TARGET_COUNT > 0 and valid_count >= TARGET_COUNT:
            print(f"[alvo] Atingido TARGET_COUNT={TARGET_COUNT}. Encerrando.")
            break

        polite_sleep()

    # grava o restante
    if results:
        df_out = pd.DataFrame(results)
        mode = "a" if os.path.exists(OUT_CSV) else "w"
        header = not os.path.exists(OUT_CSV)
        df_out.to_csv(OUT_CSV, index=False, encoding="utf-8", mode=mode, header=header)

    # Métricas
    dur = round(time.time() - t0, 2)
    metrics = {
        "target_count": TARGET_COUNT,
        "min_chars_dispositivo": MIN_CHARS,
        "tot_iterados": int(len(df_ids)),
        "validos": int(valid_count),
        "status_counts": counters,
        "duracao_s": dur,
        "rate_min": RATE_MIN,
        "rate_max": RATE_MAX,
        "timeout_s": REQUEST_TIMEOUT
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[ok] Métricas salvas em: {METRICS_PATH}")
    print(f"[resumo] válidos={valid_count} | duracao_s={dur} | status={counters}")


if __name__ == "__main__":
    main()