# Como usar:
    ## Exporte sua chave: export DATAJUD_API_KEY="sua_chave_publica"
    ## Rode: python Coleta_datajud_tjdft_saude.py
    ## Saída: data/tjdft_saude_ids_sentenca.csv

# -*- coding: utf-8 -*-
"""
Coleta de metadados via API Pública do DataJud (CNJ) - TJDFT
Recorte: Saúde (código 12480) + processos com "Sentença" (G1/JE), 2023-01-01 até hoje.
Saída: CSV com números de processo e metadados para posterior scraping no PJe/Jurisprudência do TJDFT.

Requisitos: requests, pandas
Defina a variável de ambiente DATAJUD_API_KEY com a chave pública do CNJ ou informá-la no código.
Docs: 
- Tutorial da API Pública (aliases, headers, search_after): CNJ (PDF)
- Wiki da API: datajud-wiki.cnj.jus.br/api-publica/
"""

import os
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import random
import numpy as np

# =========================
# 1) CONFIGURAÇÃO GERAL
# =========================

DATAJUD_URL = "https://api-publica.datajud.cnj.jus.br/api_publica_tjdft/_search"
# Fallback para reprodutibilidade: usa env se houver; senão, usa a chave pública informada
# Fallback: usa env se existir; senão, chave pública informada pela autora (pode ser rotacionada pelo CNJ)
API_KEY = os.getenv("DATAJUD_API_KEY") or "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="
SAIDA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "tjdft_saude_ids_sentenca.csv")

# Janela temporal do estudo (ajuste se necessário)
DATA_INICIO = "2023-01-01"
DATA_FIM = datetime.today().strftime("%Y-%m-%d")

# Assunto (TPU) alvo do nicho "Saúde"
ASSUNTO_SAUDE = 12480  # Direito da Saúde (macro-código na TPU / SGT)

# Parâmetros de paginação robustos
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "200"))
MAX_DOCS  = int(os.getenv("MAX_DOCS",  "6000"))  # ligeiro aumento para garantir estoque à raspagem
SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0.5"))

# Reprodutibilidade
random.seed(42); np.random.seed(42)

# =========================
# 2) UTILITÁRIOS HTTP
# =========================

def _headers() -> dict:
    return {
        "Authorization": f"APIKey {API_KEY}",
        "Content-Type": "application/json",
    }

def post_datajud(payload: Dict[str, Any], max_retries: int = 5, backoff: float = 2.0) -> Dict[str, Any]:
    """POST robusto ao DataJud com backoff exponencial simples para 429/5xx."""
    for attempt in range(1, max_retries + 1):
        r = requests.post(DATAJUD_URL, headers=_headers(), data=json.dumps(payload), timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff * attempt)
            continue
        raise RuntimeError(f"Erro {r.status_code}: {r.text}")
    raise RuntimeError(f"Falha após {max_retries} tentativas.")

# =========================
# 3) AGREGACOES (VOLUME)
# =========================

def aggs_volume(inicio: str = DATA_INICIO, fim: str = DATA_FIM) -> Dict[str, Any]:
    """Executa agregações por assunto/classe/grau para mapear volume no TJDFT."""
    payload_aggs = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"term": {"sigilo": False}},
                    {"terms": {"grau": ["G1", "JE", "G2"]}},
                    {"range": {"dadosBasicos.dataAjuizamento": {"gte": inicio, "lte": fim}}}
                ]
            }
        },
        "aggs": {
            "por_assunto": { "terms": { "field": "assuntos.codigo", "size": 50 } },
            "por_classe":  { "terms": { "field": "classeProcessual.codigo", "size": 50 } },
            "por_grau":    { "terms": { "field": "grau", "size": 5 } }
        }
    }
    return post_datajud(payload_aggs)

# =========================
# 4) BUSCA PAGINADA (SAÚDE + "SENTENÇA")
# =========================

def base_query_saude_sentenca(inicio: str = DATA_INICIO, fim: str = DATA_FIM) -> Dict[str, Any]:
    """Monta a query base para Saúde (12480) + G1/JE + 'Sentença' nas movimentações."""
    return {
        "size": PAGE_SIZE,
        "_source": ["numero", "grau", "classeProcessual.*", "assuntos.*",
                    "orgaoJulgador.*", "movimentos.*", "dadosBasicos.*"],
        "sort": [
            {"dadosBasicos.dataAjuizamento": "desc"},
            {"_id": "asc"}
        ],
        "query": {
            "bool": {
                "must": [
                    {"term": {"sigilo": False}},
                    {"terms": {"grau": ["G1", "JE"]}},
                    {"terms": {"assuntos.codigo": [ASSUNTO_SAUDE]}},
                    {"range": {"dadosBasicos.dataAjuizamento": {"gte": inicio, "lte": fim}}},
                    {"match_phrase": {"movimentos.descricao": "Sentença"}}
                ]
            }
        }
    }

def scroll_search_after(query_base: Dict[str, Any], max_docs: int = MAX_DOCS) -> List[Dict[str, Any]]:
    """Percorre resultados com search_after, acumulando documentos até max_docs."""
    docs: List[Dict[str, Any]] = []
    q = json.loads(json.dumps(query_base))  # deep copy
    last_sort = None

    while len(docs) < max_docs:
        if last_sort:
            q["search_after"] = last_sort
        res = post_datajud(q)
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            src = h.get("_source", {}) or {}
            docs.append(src)
        last_sort = hits[-1]["sort"]
        time.sleep(SLEEP_BETWEEN)
    return docs

# =========================
# 5) LIMPEZA: DATAS DE SENTENÇA & DEDUP
# =========================

def extrair_data_sentenca(movs: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not movs:
        return None
    # captura a última ocorrência de “Sentença”
    candidatos = [m for m in movs if isinstance(m, dict) and
                  re.search(r"\bsentença\b", str(m.get("descricao","")), flags=re.I) and
                  "dataHora" in m]
    if not candidatos:
        return None
    candidatos.sort(key=lambda m: m["dataHora"], reverse=True)
    return candidatos[0]["dataHora"]

def montar_dataframe_ids(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for d in docs:
        numero = d.get("numero")
        grau = d.get("grau")
        classe = (d.get("classeProcessual") or {}).get("nome")
        orgao = (d.get("orgaoJulgador") or {}).get("nome")
        assuntos = d.get("assuntos")
        data_ajuiz = (d.get("dadosBasicos") or {}).get("dataAjuizamento")
        data_sent = extrair_data_sentenca(d.get("movimentos"))

        if numero and data_sent:
            rows.append({
                "numero": numero,
                "grau": grau,
                "orgao_julgador": orgao,
                "classe": classe,
                "data_ajuizamento": data_ajuiz,
                "data_sentenca": data_sent,
                "assuntos": json.dumps(assuntos, ensure_ascii=False)
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["numero"]).sort_values("data_sentenca", ascending=False)
    return df

# =========================
# 6) MAIN
# =========================

def main():
    t0 = time.time()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"[1/4] Agregações de volume no TJDFT ({DATA_INICIO} a {DATA_FIM}) ...")
    agg = aggs_volume()
    # (Opcional) salvar para auditoria/diagnóstico
    with open(os.path.join(data_dir, "agg_tjdft.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print("... aggs salvas em data/agg_tjdft.json")

    print(f"[2/4] Montando query Saúde (TPU {ASSUNTO_SAUDE}) + 'Sentença' ...")
    q = base_query_saude_sentenca()

    print(f"[3/4] Coletando documentos (até {MAX_DOCS}) com search_after ...")
    docs = scroll_search_after(q, max_docs=MAX_DOCS)
    print(f"... coletados {len(docs)} documentos (pré-filtro).")

    print(f"[4/4] Filtrando com data de sentença e exportando CSV ...")
    df = montar_dataframe_ids(docs)
    df.to_csv(SAIDA_CSV, index=False, encoding="utf-8")
    print(f"✔ CSV salvo em: {SAIDA_CSV}\n {len(df)} processos com 'Sentença'.")

    # --- Métricas/diagnóstico da coleta ---
    metrics = {
        "periodo": {"inicio": DATA_INICIO, "fim": DATA_FIM},
        "total_docs_raw": len(docs),
        "total_docs_validados": int(len(df)),
        "page_size": PAGE_SIZE,
        "sleep_between_s": SLEEP_BETWEEN,
        "max_docs_param": MAX_DOCS,
        "duracao_s": round(time.time() - t0, 2)
    }
    with open(os.path.join(data_dir, "metrics_coleta.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] Métricas da coleta salvas em data/metrics_coleta.json: {metrics}")

if __name__ == "__main__":
    main()