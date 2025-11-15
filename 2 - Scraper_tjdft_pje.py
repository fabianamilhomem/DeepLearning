# Lê data/tjdft_saude_ids_sentenca.csv (saída do 1º script).
# Para cada número de processo, tenta localizar a página de detalhe no PJe e extrair:
    ## Nome do(a) magistrado(a) (heurísticas robustas em HTML/texto),
    ## Texto da decisão (priorizando “dispositivo”; faz fallback para inteiro teor/ementa),
    ## Metadados úteis (grau, órgão julgador, datas, links).
    ## Faz download de PDFs quando disponíveis e usa PyPDF2 para extrair texto.
    ## Respeita robots.txt, aplica rate limiting e registra logs de erro.
    ## Gera data/tjdft_saude_sentencas.csv (pronto para a etapa de PLN).

# -*- coding: utf-8 -*-
"""
Scraper do PJe 1º Grau do TJDFT a partir de números de processo (CSV).
Objetivo: obter nome do(a) magistrado(a) e texto da decisão (priorizando o "dispositivo")
para posterior análise de sentimento por gênero do julgador.

Fontes:
- PJe 1º Grau TJDFT (Consulta Pública): https://pje.tjdft.jus.br/consultapublica/ConsultaPublica/listView.seam
- Jurisprudência TJDFT (2º grau, fallback opcional): https://www.tjdft.jus.br/consultas/jurisprudencia
- Contexto DataJud/TJDFT: https://www.tjdft.jus.br/transparencia/tecnologia-da-informacao-e-comunicacao/dados-abertos/datajud-tjdft

Observações:
- O PJe costuma usar JSF/SEAM, gerando token "ca" na URL de detalhe. O script tenta múltiplas
  estratégias para localizar a página de detalhe e os links de documentos (HTML/PDF).
- Respeita robots.txt, faz backoff, e salva logs de erro em data/log_scraper.jsonl.
"""

import os, re, time, json, random
import unicodedata
import traceback
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter

import requests
from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader
import numpy as np

# =========================
# CONFIGS
# =========================

# Base PJe Consulta Pública – 1º Grau (TJDFT)
PJE_BASE = "https://pje.tjdft.jus.br/consultapublica/ConsultaPublica"
PJE_LIST_URL = f"{PJE_BASE}/listView.seam"
PJE_DETALHE_BASE = f"{PJE_BASE}/DetalheProcessoConsultaPublica"

# Arquivos de entrada/saída
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
IN_CSV = os.path.join(DATA_DIR, "tjdft_saude_ids_sentenca.csv")
OUT_CSV = os.path.join(DATA_DIR, "tjdft_saude_sentencas.csv")
DOC_DIR = os.path.join(DATA_DIR, "docs")
LOG_PATH = os.path.join(DATA_DIR, "log_scraper.jsonl")

os.makedirs(DOC_DIR, exist_ok=True)

# Modo de execução
USE_SELENIUM = bool(os.getenv("USE_SELENIUM", ""))
HEADLESS = bool(os.getenv("HEADLESS", "True"))
REQUEST_TIMEOUT = 60
RATE_MIN, RATE_MAX = float(os.getenv("RATE_MIN", "0.8")), float(os.getenv("RATE_MAX", "1.8"))

# Reprodutibilidade do atraso aleatório
random.seed(42); np.random.seed(42)

# Heurísticas de extração de dispositivo
DISP_PATTERNS = [
    r"\bDISPOSITIVO\b",
    r"\bDISPOSIÇÃO\b",
    r"\bISTO POSTO\b",
    r"\bPOSTO ISSO\b",
    r"\bANTE O EXPOSTO\b",
    r"\bDIANTE DO EXPOSTO\b",
    r"\bASSIM, JULGO\b",
    r"\bJULGO PROCEDENTE\b",
    r"\bJULGO IMPROCEDENTE\b",
    r"\bJULGO PARCIALMENTE PROCEDENTE\b",
]

# =========================
# UTILITÁRIOS
# =========================

def log_jsonl(path: str, record: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")
    return text[:200]

def polite_sleep():
    delay = random.uniform(RATE_MIN, RATE_MAX)
    time.sleep(delay)

def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    return s

def check_robots_allow(url_base: str, path_check: str = "/consultapublica/"):
    robots = os.path.join(DOC_DIR, "robots.txt")
    try:
        r = requests.get("https://pje.tjdft.jus.br/robots.txt", timeout=30)
        if r.status_code == 200:
            with open(robots, "w", encoding="utf-8") as f:
                f.write(r.text)
            # verificação simples
            disallows = re.findall(r"Disallow:\s*(\S+)", r.text, re.I)
            for d in disallows:
                if path_check.startswith(d.rstrip("*")) and d != "/":
                    print(f"[AVISO] robots.txt contém Disallow para {d}. Verifique política antes de continuar.")
                    # não aborta automaticamente, mas registra:
                    log_jsonl(LOG_PATH, {"warn": "robots_disallow", "path": d})
                    break
    except Exception as e:
        log_jsonl(LOG_PATH, {"warn": "robots_fetch_error", "error": str(e)})

# =========================
# TENTATIVAS DE LOCALIZAR A PÁGINA DE DETALHE DO PROCESSO
# =========================

def possible_detail_urls(numero: str) -> List[str]:
    """
    Retorna padrões de URLs de detalhe para tentativa direta.
    Em alguns PJe, há query por número; em outros, exige token 'ca'.
    Aqui deixamos algumas alternativas comuns.
    """
    urls = [
        f"{PJE_DETALHE_BASE}/listView.seam?numeroProcesso={numero}",
        f"{PJE_DETALHE_BASE}/listView.seam?numProcesso={numero}",
        # quando se conhece 'ca', formato típico:
        # f"{PJE_DETALHE_BASE}/listView.seam?ca=<TOKEN>",
    ]
    return urls

def fetch_html(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    polite_sleep()
    r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    if r.status_code == 200 and r.text:
        return BeautifulSoup(r.text, "lxml")
    return None

def try_direct_detail_by_patterns(session: requests.Session, numero: str) -> Tuple[Optional[str], Optional[BeautifulSoup]]:
    """
    Tenta acessar a página de detalhe usando padrões de URL por número.
    Se houver redirecionamento que gere 'ca', seguimos e retornamos a URL final.
    """
    for u in possible_detail_urls(numero):
        try:
            polite_sleep()
            r = session.get(u, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if r.status_code == 200 and "DetalheProcesso" in r.url:
                soup = BeautifulSoup(r.text, "lxml")
                return r.url, soup
        except Exception as e:
            log_jsonl(LOG_PATH, {"step": "try_direct_detail", "numero": numero, "error": str(e)})
    return None, None

def try_search_flow_jsf(session: requests.Session, numero: str) -> Tuple[Optional[str], Optional[BeautifulSoup]]:
    """
    Fluxo genérico JSF:
    1) GET listView.seam para cookies e ViewState.
    2) POST com campos de busca (heurístico) para tentar obter a lista e clicar no processo.
    Observação: como os nomes dos campos variam, este método tenta chaves comuns e procura anchors com "Detalhe".
    """
    try:
        soup = fetch_html(session, PJE_LIST_URL)
        if not soup:
            return None, None

        # Captura ViewState (JSF)
        viewstate = None
        vs = soup.find("input", {"name": "javax.faces.ViewState"})
        if vs and vs.has_attr("value"):
            viewstate = vs["value"]

        # Heurística de nomes de campo (podem variar por instância)
        form = soup.find("form")
        if not form:
            return None, None

        action = form.get("action") or PJE_LIST_URL
        if not action.startswith("http"):
            action = requests.compat.urljoin(PJE_LIST_URL, action)

        # Palavras-chave utilizadas em PJe – tentativa múltipla
        candidate_fields = [
            "numeroProcesso:numeroDigitoAnoUnificado",       # padrão comum
            "fPP:numeroProcesso:numeroDigitoAnoUnificado",   # com prefixo de form
            "numeroProcesso", "numProcesso", "processoRef",  # genéricos
        ]

        data = {}
        # Preenche alguns campos ocultos padrão de JSF/SEAM
        if viewstate:
            data["javax.faces.ViewState"] = viewstate

        # Insere o número nos campos que existirem
        inputs = form.find_all("input")
        used_key = None
        for inp in inputs:
            name = inp.get("name")
            if not name:
                continue
            if any(k in name for k in candidate_fields):
                data[name] = numero
                used_key = name
                break

        # Botão de busca (heurística)
        submit_name = None
        for btn in form.find_all(["input", "button"]):
            nm = btn.get("name") or ""
            val = (btn.get("value") or "").lower()
            if "pesquisar" in nm.lower() or "pesquisar" in val or "buscar" in val:
                submit_name = nm
                break
        if submit_name:
            data[submit_name] = btn.get("value") or "Pesquisar"

        if not used_key:
            # Se não conseguimos achar um campo claro de número, aborta este fluxo
            return None, None

        polite_sleep()
        r = session.post(action, data=data, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            return None, None

        soup2 = BeautifulSoup(r.text, "lxml")

        # Procura link para detalhe com token 'ca' ou texto "Detalhe do Processo"
        link = None
        for a in soup2.find_all("a", href=True):
            href = a["href"]
            if "DetalheProcessoConsultaPublica" in href or "listView.seam?ca=" in href:
                link = requests.compat.urljoin(PJE_LIST_URL, href)
                break

        if link:
            polite_sleep()
            r2 = session.get(link, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if r2.status_code == 200:
                return r2.url, BeautifulSoup(r2.text, "lxml")

    except Exception as e:
        log_jsonl(LOG_PATH, {"step": "try_search_flow_jsf", "numero": numero, "error": str(e)})

    return None, None

# =========================
# EXTRAÇÃO DE MAGISTRADO E TEXTO
# =========================

MAG_PATTERNS = [
    r"\bJu[ií]z(?:a)?(?: de Direito)?\s*[:\-]\s*(?P<nome>[A-ZÁÉÍÓÚÃÕÂÊÎÔÛÇ][\wÁÉÍÓÚÃÕÂÊÎÔÛÇ\'\.\- ]{4,})",
    r"\bMagistrad[oa]\s*[:\-]\s*(?P<nome>[A-ZÁÉÍÓÚÃÕÂÊÎÔÛÇ][\wÁÉÍÓÚÃÕÂÊÎÔÛÇ\'\.\- ]{4,})",
    r"\bJulgador[oa]\s*[:\-]\s*(?P<nome>[A-ZÁÉÍÓÚÃÕÂÊÎÔÛÇ][\wÁÉÍÓÚÃÕÂÊÎÔÛÇ\'\.\- ]{4,})",
    r"\bRelator(?:a)?\s*[:\-]\s*(?P<nome>[A-ZÁÉÍÓÚÃÕÂÊÎÔÛÇ][\wÁÉÍÓÚÃÕÂÊÎÔÛÇ\'\.\- ]{4,})",  # fallback (mais comum em 2º grau)
    # assinatura:
    r"\bAssinado(?: eletronicamente)? por:\s*(?P<nome>[A-ZÁÉÍÓÚÃÕÂÊÎÔÛÇ][\wÁÉÍÓÚÃÕÂÊÎÔÛÇ\'\.\- ]{4,})",
]

def extract_magistrate(text: str) -> Optional[str]:
    clean = re.sub(r"\s+", " ", text.strip())
    for pat in MAG_PATTERNS:
        m = re.search(pat, clean, flags=re.I)
        if m and m.group("nome"):
            nome = m.group("nome").strip(" .,-;:")
            # limpa sobras
            nome = re.sub(r"\s{2,}", " ", nome)
            # corta sufixos óbvios
            nome = re.sub(r"(,?\s*Ju[ií]z(?:a)?(?: de Direito)?)$", "", nome, flags=re.I).strip()
            return nome
    return None

def extract_dispositivo(text: str, window_chars: int = 1800) -> Optional[str]:
    """
    Procura marcadores de dispositivo/frase de decisão e extrai uma janela de texto
    para a análise de sentimento. Ajuste window_chars conforme necessidade.
    """
    t = text
    # normaliza quebras
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # tenta por marcadores
    for pat in DISP_PATTERNS:
        m = re.search(pat, t, flags=re.I)
        if m:
            start = max(0, m.start())
            end = min(len(t), start + window_chars)
            return t[start:end].strip()
    # fallback: pega últimos 1500-2000 chars (muitas sentenças concentram dispositivo no fim)
    tail = t[-2000:].strip()
    return tail if tail else None

def soup_text(soup: BeautifulSoup) -> str:
    # remove scripts e estilos
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def download_pdf(session: requests.Session, url: str, numero: str) -> Optional[str]:
    try:
        polite_sleep()
        r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and r.headers.get("Content-Type", "").lower().startswith("application/pdf"):
            fname = os.path.join(DOC_DIR, f"{slugify(numero)}_{slugify(os.path.basename(url))}.pdf")
            with open(fname, "wb") as f:
                f.write(r.content)
            return fname
    except Exception as e:
        log_jsonl(LOG_PATH, {"step": "download_pdf", "url": url, "error": str(e)})
    return None

def extract_text_from_pdf(path: str) -> str:
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return "\n".join(pages)
    except Exception as e:
        log_jsonl(LOG_PATH, {"step": "pdf_extract_error", "file": path, "error": str(e)})
        return ""

def extract_from_detail_page(session: requests.Session, detail_url: str, numero: str) -> Dict[str, Any]:
    result = {
        "numero": numero,
        "fonte_url": detail_url,
        "magistrado": None,
        "texto_dispositivo": None,
        "status": "ok",
    }

    soup = fetch_html(session, detail_url)
    if not soup:
        result["status"] = "detail_fetch_failed"
        return result

    # Tenta capturar links para documentos (HTML/PDF) que contenham "Sentença"/"Inteiro Teor"
    doc_links = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").strip().lower()
        href = a["href"]
        if any(k in text for k in ["sentença", "inteiro teor", "decisão", "ementa"]):
            full = requests.compat.urljoin(detail_url, href)
            doc_links.append(full)

    # 1) Tenta extrair diretamente do HTML da página de detalhe (se o texto já estiver lá)
    page_text = soup_text(soup)
    mag = extract_magistrate(page_text)
    disp = extract_dispositivo(page_text)
    if mag:
        result["magistrado"] = mag
    if disp and len(disp) > 100:
        result["texto_dispositivo"] = disp

    # 2) Se ainda faltar texto/magistrado, tenta abrir documentos linkados
    if (not result["texto_dispositivo"] or not result["magistrado"]) and doc_links:
        for link in doc_links:
            # Heurística: prioriza PDFs
            if link.lower().endswith(".pdf"):
                pdf_path = download_pdf(session, link, numero)
                if pdf_path:
                    pdf_text = extract_text_from_pdf(pdf_path)
                    if pdf_text:
                        if not result["magistrado"]:
                            m = extract_magistrate(pdf_text)
                            if m:
                                result["magistrado"] = m
                        if not result["texto_dispositivo"]:
                            d = extract_dispositivo(pdf_text)
                            if d and len(d) > 100:
                                result["texto_dispositivo"] = d
                        if result["magistrado"] and result["texto_dispositivo"]:
                            break
            else:
                # documento HTML
                s2 = fetch_html(session, link)
                if s2:
                    t2 = soup_text(s2)
                    if not result["magistrado"]:
                        m = extract_magistrate(t2)
                        if m:
                            result["magistrado"] = m
                    if not result["texto_dispositivo"]:
                        d = extract_dispositivo(t2)
                        if d and len(d) > 100:
                            result["texto_dispositivo"] = d
                    if result["magistrado"] and result["texto_dispositivo"]:
                        break

    # Se não achou nada, marca status
    if not result["texto_dispositivo"]:
        result["status"] = "no_text_found"
    if not result["magistrado"]:
        # mantém status ok/no_text_found conforme caso; só acrescenta nota
        result["status"] = (result["status"] + "|no_magistrate").strip("|")

    return result

# =========================
# (OPCIONAL) SELENIUM
# =========================

def selenium_fetch_detail(numero: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fallback com Selenium para páginas dinâmicas.
    Retorna (page_text, current_url) ou (None, None).
    Requer: selenium + webdriver-manager.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        opts = Options()
        if HEADLESS:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
        driver.set_page_load_timeout(60)

        driver.get(PJE_LIST_URL)
        time.sleep(2)

        # Tenta localizar um campo com "Número do processo"
        candidates = [
            (By.NAME, "numeroProcesso:numeroDigitoAnoUnificado"),
            (By.NAME, "fPP:numeroProcesso:numeroDigitoAnoUnificado"),
            (By.NAME, "numeroProcesso"),
        ]
        field = None
        for by, nm in candidates:
            try:
                field = WebDriverWait(driver, 5).until(EC.presence_of_element_located((by, nm)))
                break
            except:
                continue
        if not field:
            driver.quit()
            return None, None

        field.clear()
        field.send_keys(numero)

        # Clica em "Pesquisar"
        btn = None
        try:
            btn = driver.find_element(By.XPATH, "//input[contains(@value,'esquisar') or contains(@value,'Buscar')]")
        except:
            pass
        if not btn:
            try:
                btn = driver.find_element(By.XPATH, "//button[contains(.,'Pesquisar') or contains(.,'Buscar')]")
            except:
                driver.quit()
                return None, None
        btn.click()

        # Aguarda lista e clica no primeiro detalhe
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
        links = driver.find_elements(By.XPATH, "//a[contains(@href,'DetalheProcessoConsultaPublica') or contains(@href,'listView.seam?ca=')]")
        if not links:
            driver.quit()
            return None, None
        links[0].click()

        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html = driver.page_source
        url = driver.current_url
        driver.quit()
        return html, url
    except Exception as e:
        log_jsonl(LOG_PATH, {"step": "selenium_error", "numero": numero, "error": str(e)})
        return None, None

# =========================
# EXECUÇÃO PRINCIPAL
# =========================

def process_one(session: requests.Session, numero: str) -> Dict[str, Any]:
    """
    Fluxo consolidado para um processo:
    1) Tenta acessar detalhe por padrões de URL diretos.
    2) Tenta fluxo JSF (form de busca).
    3) (Opcional) Selenium.
    4) Extrai magistrado + texto (HTML/PDF).
    """
    try:
        # 1) padrões diretos
        url, soup = try_direct_detail_by_patterns(session, numero)
        if not url or not soup:
            # 2) fluxo de busca JSF
            url, soup = try_search_flow_jsf(session, numero)

        # 3) Selenium (opcional)
        if (not url or not soup) and USE_SELENIUM:
            html, curr = selenium_fetch_detail(numero)
            if html and curr:
                soup = BeautifulSoup(html, "lxml")
                url = curr

        if not url or not soup:
            return {
                "numero": numero,
                "fonte_url": None,
                "magistrado": None,
                "texto_dispositivo": None,
                "status": "detail_not_found",
            }

        # 4) extrai da página
        result = extract_from_detail_page(session, url, numero)
        return result

    except Exception as e:
        log_jsonl(LOG_PATH, {
            "step": "process_one_exception",
            "numero": numero,
            "error": str(e),
            "trace": traceback.format_exc()
        })
        return {
            "numero": numero,
            "fonte_url": None,
            "magistrado": None,
            "texto_dispositivo": None,
            "status": "exception"
        }

def main():
    # Checa robots.txt (respeitar políticas)
    check_robots_allow("https://pje.tjdft.jus.br", "/consultapublica/")

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {IN_CSV}")

    df_ids = pd.read_csv(IN_CSV)
    if "numero" not in df_ids.columns:
        raise ValueError("CSV de entrada precisa conter a coluna 'numero'.")

    # Resume: se já existir saída, evitamos reprocessar
    done = set()
    if os.path.exists(OUT_CSV):
        df_done = pd.read_csv(OUT_CSV)
        if "numero" in df_done.columns:
            done = set(df_done["numero"].astype(str).tolist())

    session = get_session()

    # parâmetros de filtragem de qualidade do texto
    MIN_CHARS = int(os.getenv("MIN_CHARS_DISPOSITIVO", "300"))

    # contadores de status
    counters = Counter()
    t0 = time.time()

    results: List[Dict[str, Any]] = []
    batch = 0

    for i, row in df_ids.iterrows():
        batch += 1
        numero = str(row["numero"]).strip()
        if not numero or numero in done:
            continue

        print(f"[{i+1}/{len(df_ids)}] Processando {numero} ...")
        res = process_one(session, numero)

        # filtro de qualidade do dispositivo: impõe mínimo de caracteres
        txt = res.get("texto_dispositivo") or ""
        if txt and len(txt) < MIN_CHARS:
            res["status"] = (res["status"] + "|short_text").replace("ok|", "")
            res["texto_dispositivo"] = None

        # agrega metadados do CSV de entrada
        res["grau"] = row.get("grau") if "grau" in df_ids.columns else None
        res["orgao_julgador"] = row.get("orgao_julgador") if "orgao_julgador" in df_ids.columns else None
        res["classe"] = row.get("classe") if "classe" in df_ids.columns else None
        res["data_ajuizamento"] = row.get("data_ajuizamento") if "data_ajuizamento" in df_ids.columns else None
        res["data_sentenca"] = row.get("data_sentenca") if "data_sentenca" in df_ids.columns else None

        results.append(res)

        # atualiza contadores
        counters[res["status"]] += 1
        if res.get("texto_dispositivo"): counters["valid_text"] += 1
        if res.get("magistrado"):        counters["has_magistrate"] += 1

        # grava incrementalmente a cada 20 itens
        if batch % 20 == 0:
            df_out = pd.DataFrame(results)
            mode = "a" if os.path.exists(OUT_CSV) else "w"
            header = not os.path.exists(OUT_CSV)
            df_out.to_csv(OUT_CSV, index=False, encoding="utf-8", mode=mode, header=header)
            results = []
            print(f"  -> checkpoint salvo ({batch} itens).")

    # grava o restante
    if results:
        df_out = pd.DataFrame(results)
        mode = "a" if os.path.exists(OUT_CSV) else "w"
        header = not os.path.exists(OUT_CSV)
        df_out.to_csv(OUT_CSV, index=False, encoding="utf-8", mode=mode, header=header)

    print(f"✔ Finalizado. Saída em: {OUT_CSV}")

    # --- métricas/diagnóstico da raspagem ---
    total_processados = sum(counters.values()) - counters.get("valid_text", 0) - counters.get("has_magistrate", 0)
    metrics = {
        "min_chars_dispositivo": MIN_CHARS,
        "status_counts": dict(counters),
        "total_itens_iterados": int(len(df_ids) - len(done)),
        "duracao_s": round(time.time() - t0, 2)
    }
    with open(os.path.join(DATA_DIR, "metrics_scraper.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] Métricas da raspagem salvas em data/metrics_scraper.json: {metrics}")


if __name__ == "__main__":
    main()