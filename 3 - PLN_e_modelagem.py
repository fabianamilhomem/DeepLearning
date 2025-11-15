# -*- coding: utf-8 -*-
"""
03 - PLN e Modelagem de Sentimento (Aulas 05 e 06)

Fluxo:
1) Carrega 'data/tjdft_saude_sentencas.csv' (numero, magistrado, texto_dispositivo, metadados)
2) Pré-processa: tokenização, lematização (spaCy), stopwords; opcional: stemming (RSLP)
3) Inferência de gênero do(a) magistrado(a) a partir do prenome (IBGE/Brasil.IO offline CSV)
4) Representações e modelos:
   4.1) BoW e TF-IDF (baselines)
   4.2) Word2Vec (CBOW e Skip-gram) treinados no corpus + média por sentença
   4.3) GloVe (NILC pré-treinado) + média
   4.4) Transformer (BERTimbau/FinBERT-PT-BR) via pipeline (contextual)
5) Métricas, comparações por gênero e testes estatísticos
"""
import os, re, json, unicodedata, warnings, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from scipy import stats

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from gensim.models import Word2Vec
from transformers import pipeline

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42)

# === Caminhos ===
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
IN_SENTENCAS = os.path.join(DATA_DIR, "tjdft_saude_sentencas.csv")

# === Carrega dataset principal ===
df = pd.read_csv(IN_SENTENCAS)
# Filtra válidos
df = df[df["texto_dispositivo"].notna() & (df["texto_dispositivo"].str.len() > 300)].copy()

# === Aula 05: PRÉ-PROCESSAMENTO (tokenização, normalização, lemas, stopwords) ===
# spaCy pt (com lematizador); NLTK stopwords PT; RSLP opcional
try:
    nlp = spacy.load("pt_core_news_md")  # pode trocar por 'sm' se necessário
except OSError:
    raise RuntimeError("Instale o modelo spaCy PT: python -m spacy download pt_core_news_md")

nltk.download("stopwords", quiet=True)
STOP_PT = set(stopwords.words("portuguese"))
STEMMER = RSLPStemmer()  # opcional

def normalize_text(t: str) -> str:
    """Remove espaços extras e normaliza unicode."""
    t = t.replace("\r", "\n")
    t = re.sub(r"\s+", " ", t).strip()
    return unicodedata.normalize("NFKC", t)

def spacy_lemmatize(text: str, do_stem: bool=False) -> str:
    doc = nlp(text)
    toks = []
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.like_num or tok.is_stop:
            continue
        lemma = tok.lemma_.lower().strip()
        if len(lemma) < 2 or lemma in STOP_PT:
            continue
        if do_stem:
            lemma = STEMMER.stem(lemma)
        toks.append(lemma)
    return " ".join(toks)

df["texto_norm"] = df["texto_dispositivo"].astype(str).map(normalize_text)
df["texto_lemmas"] = df["texto_norm"].map(lambda s: spacy_lemmatize(s, do_stem=False))
# Se quiser comparar com stemming:
# df["texto_stems"] = df["texto_norm"].map(lambda s: spacy_lemmatize(s, do_stem=True))

# === Inferência de gênero do(a) magistrado(a) (IBGE/Brasil.IO offline) ===
# Baixe um CSV com frequências de prenomes e gênero (ex.: Brasil.IO "Gênero dos Nomes").
# Exemplo esperado de schema: nome, genero(F/M), prob_fem, prob_masc ...
# Fonte de dados: https://brasil.io/dataset/genero-nomes/nomes/ (CSV)  # (ver README)
IBGE_CSV = os.path.join(DATA_DIR, "ibge_genero_nomes.csv")
if os.path.exists(IBGE_CSV):
    ibge = pd.read_csv(IBGE_CSV)
    # normaliza chave
    ibge["nome_chave"] = ibge["Nome"].str.upper().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    ibge = ibge.drop_duplicates(subset=["nome_chave"])
    MAPA = ibge.set_index("nome_chave")["Gênero"].to_dict()
else:
    MAPA = {}

def extrai_prenome(nome: str) -> str:
    if not isinstance(nome, str): return ""
    nome = re.sub(r"\s+", " ", nome).strip()
    # remove títulos comuns
    nome = re.sub(r"^(?:Ju[ií]z(?:a)?|Doutor(?:a)?|Desembargador(?:a)?)\s+", "", nome, flags=re.I)
    return nome.split(" ")[0] if nome else ""

def genero_por_nome(nome: str) -> str:
    if not nome: return "indeterminado"
    key = unicodedata.normalize("NFKD", nome.upper()).encode("ascii", "ignore").decode("ascii")
    return MAPA.get(key, "indeterminado")

df["magistrado_prenome"] = df["magistrado"].astype(str).map(extrai_prenome)
df["genero_magistrado"] = df["magistrado_prenome"].map(genero_por_nome)

# === Preparação de rótulos de sentimento ===
# Para este TCC, usaremos um modelo contextual PT pré-treinado (FinBERT-PT-BR) para rotular (POS/NEG/NEU).
# Atenção ao MISMATCH de domínio (financeiro vs jurídico). Discutir no relatório.
clf_ctx = pipeline(
    task="text-classification",
    model="lucas-leme/FinBERT-PT-BR",
    tokenizer="lucas-leme/FinBERT-PT-BR",
    truncation=True
)
def rotula_sentimento(texto: str) -> str:
    try:
        out = clf_ctx(texto[:2000])[0]  # limita tamanho para velocidade
        return out["label"].upper()
    except Exception:
        return "NEUTRAL"

df["sent_label_ctx"] = df["texto_norm"].map(rotula_sentimento)  # labels: POSITIVE/NEGATIVE/NEUTRAL

# Cria versão numérica para análise de médias/efeitos: NEG=-1, NEU=0, POS=+1
MAP_NUM = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
df["sent_num_ctx"] = df["sent_label_ctx"].map(MAP_NUM)

# === Aula 05: Baselines com BoW e TF-IDF ===
X = df["texto_lemmas"].fillna("")
y = df["sent_label_ctx"]  # usando o rótulo contextual como "gold fraco"

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# BoW + Logistic Regression
pipe_bow = Pipeline([
    ("vect", CountVectorizer(min_df=3, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])
pipe_bow.fit(X_tr, y_tr)
pred_bow = pipe_bow.predict(X_te)

# TF-IDF + LinearSVC
pipe_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=3, ngram_range=(1,2))),
    ("clf", LinearSVC())
])
pipe_tfidf.fit(X_tr, y_tr)
pred_tfidf = pipe_tfidf.predict(X_te)

print("\n=== Baseline BoW + LR ===")
print(classification_report(y_te, pred_bow, digits=3))
print("\n=== Baseline TF-IDF + LinearSVC ===")
print(classification_report(y_te, pred_tfidf, digits=3))

# === Aula 05: Word2Vec (CBOW vs Skip-gram) ===
from gensim.utils import simple_preprocess
def tokenize_for_w2v(txt): return txt.split()

sents_tokens = [tokenize_for_w2v(t) for t in df["texto_lemmas"].fillna("").tolist()]

# CBOW (sg=0) e Skip-gram (sg=1)
w2v_cbow = Word2Vec(sentences=sents_tokens, vector_size=300, window=5, min_count=5, workers=4, sg=0, seed=42)
w2v_sg   = Word2Vec(sentences=sents_tokens, vector_size=300, window=5, min_count=5, workers=4, sg=1, seed=42)

def sent_vector_avg(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs: return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

X_emb_cbow = np.vstack([sent_vector_avg(t, w2v_cbow) for t in sents_tokens])
X_emb_sg   = np.vstack([sent_vector_avg(t, w2v_sg)   for t in sents_tokens])

# split alinhado
idx = np.arange(len(df))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

def avalia_dense(X_emb, y, train_idx, test_idx, nome):
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_emb[train_idx], y.iloc[train_idx])
    pred = lr.predict(X_emb[test_idx])
    print(f"\n=== {nome} + LogReg ===")
    print(classification_report(y.iloc[test_idx], pred, digits=3))

avalia_dense(X_emb_cbow, y, train_idx, test_idx, "Word2Vec CBOW")
avalia_dense(X_emb_sg,   y, train_idx, test_idx, "Word2Vec Skip-gram")

# === Aula 05 (opcional): GloVe PT (NILC) ===
# Se tiver o arquivo de embeddings do NILC (ex.: safetensors/text), carregar e fazer média por sentença (similar ao Word2Vec).
# Ver: https://huggingface.co/nilc-nlp/glove-100d (ou coleções NILC). (Instruções no README)
# --- placeholder para carregamento GloVe se disponível ---

# === Aula 06: Análise por gênero do(a) julgador(a) ===
# Distribuições por grupo e teste de hipóteses
def estatisticas_por_genero(col_label="sent_num_ctx", genero_col="genero_magistrado"):
    sub = df[df[genero_col].isin(["F", "M"])].copy()
    gF = sub[sub[genero_col]=="F"][col_label].dropna()
    gM = sub[sub[genero_col]=="M"][col_label].dropna()
    print("\n=== Distribuições numéricas (POS=+1, NEU=0, NEG=-1) ===")
    print("Feminino:", gF.describe())
    print("Masculino:", gM.describe())
    # Teste não-paramétrico (Mann-Whitney) por robustez
    u, p = stats.mannwhitneyu(gF, gM, alternative="two-sided")
    print(f"Mann-Whitney U={u:.0f}, p={p:.4f}")
    return gF, gM

gF, gM = estatisticas_por_genero()
print("\nPronto. Gere gráficos no notebook/apresentação (boxplot, violino, ECDF).")