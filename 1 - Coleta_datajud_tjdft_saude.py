# -*- coding: utf-8 -*-
"""
TJDFT • Estoque de SENTENÇAS (G1/JE; opcional TR) • Execução única com early-stop

- Tribunal: TJDFT (API Pública DataJud).
- Graus: por padrão **G1 e JE** (NÃO inclui 2º grau). Opcional incluir **TR** via flag.
- SEM filtro/ordenador por data no ES.
- Particionamento automático por **ano no número CNJ** (NNNNNNN-DD.AAAA.J.TR.OOOO), com `regexp` em numeroProcesso.keyword.
- Pré-filtro no ES por nomes típicos de **sentença** (should + minimum_should_match=1) para acelerar; fallback sem pré-filtro se necessário.
- Ordenação **numeroProcesso.keyword ASC** + **id.keyword ASC** e paginação via **search_after**.
- Filtra **Sentença** (equivalentes 1º grau) a cada página, **dedup por número**, e **para** ao atingir TARGET_MIN.
- Saída: **1 CSV** final (sem juntar manualmente).

Recomendações do Elastic/DataJud para paginação:
- Use `search_after` com ordenação em campos com `doc_values` (ex.: id.keyword / keyword), **evite `_id`**.  # [1](https://dataloop.ai/library/model/spacy_pt_core_news_sm/)
Atenção a `date` em ES: `range` exige mapeamento/format compatível (não presumir ISO).  # [2](https://github.com/turicas/genero-nomes)
"""

import os
import time
import json
import re
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "DATAJUD_URL": "https://api-publica.datajud.cnj.jus.br/api_publica_tjdft/_search",
    "API_KEY": os.getenv("DATAJUD_API_KEY") or "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw==",

    # >>> Graus (TR opcional) <<<
    "INCLUIR_TR": True,     # mude para False se quiser G1/JE puro

    # Pré-filtro por "nomes de sentença" no ES (acelera). Fallback sem isso se necessário.
    "ES_PREFILTER_SENTENCA": True,

    # Laço anual automático (ano mais recente → ano mais antigo)
    "ANO_INICIO": datetime.today().year,  # 2025
    "ANO_FIM": 1995,                      # ajuste se quiser ir mais longe

    # Paginação + early-stop
    "PAGE_SIZE": 10000,       # até 10.000 por página (ES)
    "MAX_DOCS_HARD": 3000000, # failsafe global
    "TARGET_MIN": 650,        # meta >=500 (sobra para Saúde)
    "TARGET_HEADROOM": 100,   # folga acima da meta
    "SLEEP_BETWEEN": 0.08,    # prudência com rate limit

    # Ordenação para search_after (NÃO usar _id)
    "SORT_FIELDS": ["numeroProcesso.keyword", "id.keyword"],

    # Exclusões (preparatórios / "sobre" a sentença)
    "EXCLUDE_MOVES": [
        "para julgamento","para julgamento de mérito","incluído em pauta","pauta de julgamento",
        "concluso","conclusão","conclusos para julgamento","vista","remessa","distribuição",
        "recebimento","intimação","publicação","despacho","decisão interlocutória",
        "anulação de sentença","anulação de sentença/acórdão",
        "impugnação ao cumprimento de sentença","cumprimento de sentença",
        "execução","extinção da execução","extinção do cumprimento da sentença",
        "acordo em execução","acordo em cumprimento de sentença",
        "convenção das partes para satisfação voluntária","satisfação voluntária da obrigação",
        "baixa dos autos","trânsito em julgado",
        "embargos de declaração"
    ],

    # Saída
    "DIR_DATA": "data",
    "ARQ_CSV": "tjdft_g1je_sentencas_estoque.csv",
}

# =========================================================
# HTTP utilitários
# =========================================================
def _headers() -> Dict[str, str]:
    return {"Authorization": f"APIKey {CONFIG['API_KEY']}", "Content-Type": "application/json"}

def post_datajud(payload: Dict[str, Any], max_retries: int = 5, backoff: float = 2.0) -> Dict[str, Any]:
    for attempt in range(1, max_retries + 1):
        r = requests.post(CONFIG["DATAJUD_URL"], headers=_headers(), data=json.dumps(payload), timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff * attempt)
            continue
        raise RuntimeError(f"Erro {r.status_code}: {r.text}")
    raise RuntimeError(f"Falha após {max_retries} tentativas.")

# =========================================================
# Query base (ano CNJ + graus + pré-filtro de movimento)
# =========================================================
def graus_ok() -> list:
    g = ["G1", "JE"]
    if CONFIG.get("INCLUIR_TR", False):
        g.append("TR")
    return g

def _mov_should_list() -> list:
    # frases de sentença p/ 'should' (pré-filtro no ES)
    frases = [
        "Sentença",
        "Julgado procedente","Julgado improcedente","Julgado parcialmente procedente",
        "Procedência","Improcedência","Procedência em Parte","Procedência parcial",
        "Extinção do processo com resolução do mérito",
        "Julgamento antecipado do mérito",
        "Homologação de transação","Homologação de acordo",
        "Indeferimento da petição inicial"
    ]
    should = [{"match_phrase": {"movimentos.nome": f}} for f in frases]
    # também tentar no campo 'descricao' se existir
    should += [{"match_phrase": {"movimentos.descricao": f}} for f in frases]
    return should

def build_query_ano(ano: int, usar_prefiltro: bool) -> Dict[str, Any]:
    must = [
        {"term": {"nivelSigilo": 0}},
        {"terms": {"grau.keyword": graus_ok()}},
        # Filtra por ANO do CNJ no numeroProcesso.keyword: ... .AAAA .
        {"regexp": {"numeroProcesso.keyword": f".*\\.{ano}\\..*"}}
    ]
    bool_q = {"must": must}
    if usar_prefiltro:
        bool_q["should"] = _mov_should_list()
        bool_q["minimum_should_match"] = 1

    sort_clause = [{CONFIG["SORT_FIELDS"][0]: {"order": "asc", "unmapped_type": "keyword"}},
                   {CONFIG["SORT_FIELDS"][1]: {"order": "asc", "unmapped_type": "keyword"}}]

    return {
        "size": CONFIG["PAGE_SIZE"],
        "_source": [
            "numeroProcesso","grau","classe.*","assuntos.*",
            "orgaoJulgador.*","movimentos.*",
            "dataAjuizamento","dataHoraUltimaAtualizacao","id"
        ],
        "sort": sort_clause,
        "query": {"bool": bool_q}
    }

# =========================================================
# Classificador (POS/NEG)
# =========================================================
_POS_RE_SENTENCA       = re.compile(r"\bsenten[çc]a\b", re.I)
_POS_RE_PROLACAO       = re.compile(r"\bprola[cç][aã]o\s+de\s+senten[çc]a\b", re.I)
_POS_RE_TIPOS_SENT     = re.compile(r"\bsenten[çc]a\s+(terminativa|definitiva|de\s+m[eé]rito)\b", re.I)

_POS_RE_JULGADO        = re.compile(r"\bjulgado(?:s)?\s+(procedente|improcedente|parcialmente\s+procedente)\b", re.I)
_POS_RE_PROCEDENCIA    = re.compile(r"\bproced[eê]ncia(?:\s+em\s+parte|(?:\s+parcial)?)\b", re.I)
_POS_RE_IMPROCED       = re.compile(r"\bimproced[eê]ncia\b", re.I)

_POS_RE_EXT_MERITO     = re.compile(r"\bextin[cç][aã]o\s+do\s+processo\s+com\s+resolu[cç][aã]o\s+do\s+m[eé]rito\b", re.I)
_POS_RE_EXT_SEM_MERITO = re.compile(r"\bextin[cç][aã]o\s+do\s+processo\s+sem\s+resolu[cç][aã]o\s+do\s+m[eé]rito\b", re.I)
_POS_RE_JULG_ANTEC     = re.compile(r"\bjulgamento\s+antecipado\s+do\s+m[eé]rito\b", re.I)
_POS_RE_HOMOL_TRANS    = re.compile(r"\bhomologa[cç][aã]o\s+de\s+transa[cç][aã]o\b", re.I)
_POS_RE_HOMOL_ACORDO   = re.compile(r"\bhomologa[cç][aã]o\s+de\s+acord[oa]\b", re.I)
_POS_RE_HOMOL_DESIST   = re.compile(r"\bhomologa[cç][aã]o\s+de\s+desist[eê]ncia\b", re.I)
_POS_RE_HOMOL_RENUNC   = re.compile(r"\bhomologa[cç][aã]o\s+de\s+ren[úu]ncia\b", re.I)
_POS_RE_INDEP_INI      = re.compile(r"\bindefirimento\s+da\s+peti[cç][aã]o\s+inicial\b", re.I)

_NEG_FIXED = [e.lower() for e in CONFIG["EXCLUDE_MOVES"]]
_NEG_RE_LIST = [
    re.compile(r"\banulaç[aã]o\s+de\s+senten[çc]a", re.I),
    re.compile(r"\banulaç[aã]o\s+de\s+ac[óo]rd[ãa]o", re.I),
    re.compile(r"\bimpugnaç[aã]o\s+ao\s+cumprimento\s+de\s+senten[çc]a\b", re.I),
    re.compile(r"\bcumprimento\s+de\s+senten[çc]a\b", re.I),
    re.compile(r"\bexecu[cç][aã]o\b", re.I),
    re.compile(r"\bextinç[aã]o\s+da?\s+(execu[cç][aã]o|do\s+cumprimento\s+da?\s+senten[çc]a)\b", re.I),
    re.compile(r"\bacordo\s+em\s+(execu[cç][aã]o|cumprimento\s+de\s+senten[çc]a)\b", re.I),
    re.compile(r"\bconvenç[aã]o\s+das\s+partes\b", re.I),
    re.compile(r"\bsatisfa[cç][aã]o\s+volunt[aá]ria\s+da?\s+obriga[cç][aã]o\b", re.I),
    re.compile(r"\btr[aâ]nsito\s+em\s+julgado\b", re.I),
    re.compile(r"\bbaixa\s+dos?\s+autos\b", re.I),
    re.compile(r"\bembargos?\s+de\s+declara[cç][aã]o\b", re.I),
    re.compile(r"\bpublica[cç][aã]o\s+de\s+senten[çc]a\b", re.I),
    re.compile(r"\bretifica[cç][aã]o\s+de\s+senten[çc]a\b", re.I),
    re.compile(r"\bcomplementa[cç][aã]o\s+de\s+senten[çc]a\b", re.I),
]

def _is_negative(nome_l: str) -> bool:
    if any(term in nome_l for term in _NEG_FIXED): return True
    return any(rx.search(nome_l) for rx in _NEG_RE_LIST)

def _is_positive_sentenca(nome_l: str) -> bool:
    return (
        _POS_RE_SENTENCA.search(nome_l) is not None or
        _POS_RE_PROLACAO.search(nome_l) is not None or
        _POS_RE_TIPOS_SENT.search(nome_l) is not None or
        _POS_RE_JULGADO.search(nome_l) is not None or
        _POS_RE_PROCEDENCIA.search(nome_l) is not None or
        _POS_RE_IMPROCED.search(nome_l) is not None or
        _POS_RE_EXT_MERITO.search(nome_l) is not None or
        _POS_RE_EXT_SEM_MERITO.search(nome_l) is not None or
        _POS_RE_JULG_ANTEC.search(nome_l) is not None or
        _POS_RE_HOMOL_TRANS.search(nome_l) is not None or
        _POS_RE_HOMOL_ACORDO.search(nome_l) is not None or
        _POS_RE_HOMOL_DESIST.search(nome_l) is not None or
        _POS_RE_HOMOL_RENUNC.search(nome_l) is not None or
        _POS_RE_INDEP_INI.search(nome_l) is not None
    )

def extrair_decisao(movs: Optional[List[Dict[str, Any]]], src: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
    if not movs: return None
    candidatos = []
    for m in movs:
        if not isinstance(m, dict): continue
        nome = (m.get("nome") or m.get("descricao") or "").strip()
        if not nome: continue
        nome_l = nome.lower()
        if _is_negative(nome_l): continue
        if not _is_positive_sentenca(nome_l): continue
        data = m.get("dataHora") or (src or {}).get("dataHoraUltimaAtualizacao")
        if data:
            candidatos.append({"data": data, "tipo": nome})
    if not candidatos:
        return None
    candidatos.sort(key=lambda x: x["data"], reverse=True)
    return candidatos[0]

# =========================================================
# Página + filtro (uma janela de ANO)
# =========================================================
def coletar_ano(ano: int, usar_prefiltro: bool, limite_final: int, df_acum: pd.DataFrame) -> pd.DataFrame:
    q = build_query_ano(ano, usar_prefiltro)
    last_sort = None
    total_docs = 0

    while len(df_acum) < limite_final and total_docs < CONFIG["MAX_DOCS_HARD"]:
        q_page = json.loads(json.dumps(q))
        if last_sort:
            q_page["search_after"] = last_sort

        res = post_datajud(q_page)
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            break

        rows = []
        for h in hits:
            src = h.get("_source", {}) or {}
            total_docs += 1
            grau = src.get("grau")
            if grau not in graus_ok():  # segurança extra
                continue
            decisao = extrair_decisao(src.get("movimentos"), src=src)
            if decisao and src.get("numeroProcesso"):
                rows.append({
                    "numero": src.get("numeroProcesso"),
                    "grau": grau,
                    "orgao_julgador": (src.get("orgaoJulgador") or {}).get("nome"),
                    "classe": (src.get("classe") or {}).get("nome"),
                    "data_ajuizamento": src.get("dataAjuizamento"),
                    "data_decisao": decisao["data"],
                    "tipo_decisao": decisao["tipo"],
                    "assuntos": json.dumps(src.get("assuntos"), ensure_ascii=False)
                })

        if rows:
            df_batch = pd.DataFrame(rows)
            df_acum = pd.concat([df_acum, df_batch], ignore_index=True)
            df_acum = df_acum.drop_duplicates(subset=["numero"])

        if hits:
            last_sort = hits[-1]["sort"]

        if total_docs % (CONFIG["PAGE_SIZE"]) == 0:
            print(f"[{ano}] docs={total_docs:,} | sentenças acumuladas={len(df_acum):,}")

        time.sleep(CONFIG["SLEEP_BETWEEN"])

    return df_acum

# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(CONFIG["DIR_DATA"], exist_ok=True)
    saida_csv = os.path.join(CONFIG["DIR_DATA"], CONFIG["ARQ_CSV"])
    t0 = time.time()

    target = CONFIG["TARGET_MIN"]
    limite_final = target + CONFIG["TARGET_HEADROOM"]
    anos = list(range(CONFIG["ANO_INICIO"], CONFIG["ANO_FIM"] - 1, -1))

    print(f"[coleta] TJDFT | Graus={','.join(graus_ok())} | Estoque sem datas | Partições por ANO CNJ")
    print(f"[coleta] Ordenação: {CONFIG['SORT_FIELDS']} asc + search_after (padrão recomendado).")  # [1](https://dataloop.ai/library/model/spacy_pt_core_news_sm/)
    print(f"[coleta] Laço ANOS: {anos[0]} → {anos[-1]} | ES_PREFILTER_SENTENCA={CONFIG['ES_PREFILTER_SENTENCA']}")

    colunas = ["numero","grau","orgao_julgador","classe","data_ajuizamento","data_decisao","tipo_decisao","assuntos"]
    df_acum = pd.DataFrame(columns=colunas)

    # Passo 1: com pré-filtro de movimentos no ES (rápido)
    for ano in anos:
        if len(df_acum) >= limite_final: break
        df_acum = coletar_ano(ano, usar_prefiltro=True, limite_final=limite_final, df_acum=df_acum)

    # Passo 2 (fallback): sem pré-filtro, se ainda não bateu a meta
    if len(df_acum) < target and CONFIG["ES_PREFILTER_SENTENCA"]:
        print(f"[fallback] Ainda abaixo da meta ({len(df_acum)}<{target}). Rodando sem pré-filtro no ES…")
        for ano in anos:
            if len(df_acum) >= limite_final: break
            df_acum = coletar_ano(ano, usar_prefiltro=False, limite_final=limite_final, df_acum=df_acum)

    df_acum.to_csv(saida_csv, index=False, encoding="utf-8")
    print(f"[ok] CSV final salvo em: {saida_csv} | total_sentencas={len(df_acum)}")

    metrics = {
        "tribunal": "TJDFT",
        "graus": ",".join(graus_ok()),
        "prefiltro_movimentos": CONFIG["ES_PREFILTER_SENTENCA"],
        "anos_varridos": f"{anos[0]}..{anos[-1]}",
        "page_size": CONFIG["PAGE_SIZE"],
        "tiebreaker": CONFIG["SORT_FIELDS"],
        "target_min": CONFIG["TARGET_MIN"],
        "target_headroom": CONFIG["TARGET_HEADROOM"],
        "total_sentencas_final": int(len(df_acum)),
        "duracao_s": round(time.time() - t0, 2),
    }
    with open(os.path.join(CONFIG["DIR_DATA"], "metrics_coleta_tjdft_g1je_estoque.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[ok] Métricas: {metrics}")

if __name__ == "__main__":
    main()