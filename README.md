# Análise de Sentenças Judiciais com Processamento de Linguagem Natural (PLN)

Este repositório contém o código-fonte desenvolvido em **Jupyter Notebook** para coleta, tratamento e análise de sentenças judiciais do **Tribunal de Justiça de Alagoas (TJAL)**, com aplicação de técnicas de **Processamento de Linguagem Natural (PLN)**, **aprendizagem de máquina** e **deep learning** para classificação de sentimento jurídico.

**O código notebook (TrabalhoFinal_DeepLearning_TJAL.ipynb) deve ser executado obrigatoriamente no Jupyter Notebook**, pois contém células sequenciais, logs de execução e visualizações intermediárias que podem não funcionar corretamente fora desse ambiente.

---

## Objetivo do Projeto:

O objetivo do projeto é construir uma *pipeline* completa para:
- consultar os metadados na API pública do DataJud (CNJ);
- coletar processos judiciais de indenização por dano moral no portal de serviços TJAL;
- extrair automaticamente o dispositivo das sentenças;
- classificar o resultado jurídico (procedente, improcedente, procedente em parte);
- mapear esses resultados para rótulos de sentimento;
- comparar o desempenho de **modelos clássicos** e **modelos de deep learning** aplicados ao texto jurídico.

---

## Requisitos:
- Python 3.9 ou superior
- Jupyter Notebook
- Navegador instalado (Chrome, Edge ou Firefox)
- Conexão com a internet

## Execução:
- Abra o arquivo .ipynb (TrabalhoFinal_DeepLearning_TJAL.ipynb) no Jupyter Notebook.
- Execute as células sequencialmente, respeitando a ordem das seções.
- Ajuste o valor do parâmetro max_itens, localizado na Seção [8], para definir a quantidade de processos a serem coletados antes do início do scraping.


## Organização do Notebook:
O notebook está estruturado em **seções numeradas**, facilitando a leitura, execução sequencial e reprodutibilidade.

### Seção [1] — Instalação de pacotes e bibliotecas necessários
Instalação de pacotes e bibliotecas Python utilizados no pipeline do projeto.

---

### Seção [2] — Importação de bibliotecas e configurações iniciais
Importação das bibliotecas Python utilizadas para scraping, análise de dados, PLN, modelagem estatística e deep learning, além de configurações gerais do ambiente.

---

### Seção [3] — Configurações adotadas para a coleta de metadados via API DataJud
Nesta seção são definidas as configurações centrais para a consulta à API pública do DataJud (CNJ), incluindo o endpoint específico do TJAL, os parâmetros de autenticação e os filtros utilizados na pesquisa. São estabelecidos critérios como o grau de jurisdição (1º grau), a classe processual (Procedimento Comum Cível) e o assunto relacionado à indenização por dano moral. Essas configurações funcionam como base para a etapa subsequente de coleta automatizada dos metadados processuais, garantindo padronização e reprodutibilidade das consultas realizadas.

---

### Seção [4] — Funções auxiliares para query, paginação, normalização e filtros
Esta seção implementa funções auxiliares responsáveis por operacionalizar a comunicação com a API DataJud, incluindo a construção das consultas (queries), o tratamento da paginação via mecanismo `search_after`, a normalização dos campos retornados e a aplicação de filtros adicionais. Essas rotinas asseguram a extração completa dos registros disponíveis, evitando perdas de dados e organizando os resultados em um formato estruturado adequado para análise posterior.

---

### Seção [5] — Coleta de metadados via API DataJud (CNJ)
Conexão com a API pública do DataJud (CNJ) para coleta de metadados de processos:
- de 1º grau;
- da classe *Procedimento Comum Cível*;
- com assunto relacionado à *indenização por dano moral*.

Os dados são normalizados e exportados para:
- `tjal_datajud_normalizado.xlsx`

Filtragem dos processos elegíveis para scraping no portal e-SAJ do TJAL, gerando:
- `processos_filtrados_TJAL.xlsx`

---

### Seção [6] — Preparação de utilitários para scraping (CNJ, navegadores, robots)
Nesta etapa são preparados os utilitários necessários para o scraping no portal de serviços do TJAL, incluindo a detecção automática do navegador disponível no ambiente (Chrome, Edge ou Firefox) e a configuração do Selenium WebDriver. Também são considerados aspectos de boas práticas, como respeito às regras de navegação automatizada, controle de tempo entre requisições e tratamento de exceções, visando reduzir riscos de bloqueio e instabilidade durante a execução do scraping.

---

### Seção [7] — Funções auxiliares de scraping e extração
Definição de funções robustas para:
- navegação automatizada no portal de serviços e-SAJ (TJAL);
- tratamento de erros e instabilidades;
- extração do HTML das movimentações processuais;
- identificação do juiz que prolatou a sentença (com fallback para juiz da distribuição);
- extração do dispositivo da sentença;
- classificação do resultado jurídico (procedente, procedente em parte, improcedente ou outra).

---

### Seção [8] — Scraping das sentenças no Portal e-SAJ (TJAL)
Execução do scraping automatizado das páginas de processos (movimentação processual) no e-SAJ do TJAL (a partir do arquivo `processos_filtrados_TJAL.xlsx`), com salvamento local dos arquivos HTML no diretório:
- `SENTENCAS_HTML/`

Leitura dos HTMLs salvos localmente e consolidação das informações em:
- `resultado_sentencas_TJAL.html`

Obs.: Esta é a fase mais demorada a depender da quantidade de processos definidos no parâmetro `max_itens`.

---

### Seção [9] — Leitura e processamento da Base Final
Leitura da base final para a inferência de gênero e análise de sentimentos; Limpeza dos dados (remove duplicatas, remove textos vazios, remove casos com classificação 'OUTRA'); tokenização e lematização do trecho do dispositivo da sentença.

---

### Seção [10] — Identificação do(a) Juiz(a) e Inferência de Gênero (Base IBGE)
Esta seção concentra as rotinas responsáveis pela identificação do magistrado que prolatou a sentença, priorizando a extração do nome a partir do texto da movimentação processual. Quando essa informação não está disponível, aplica-se um mecanismo de fallback para capturar o juiz da distribuição do processo. Em seguida, é realizada a inferência de gênero com base no prenome, utilizando a base oficial de nomes do IBGE, complementada por um cache local para otimizar o desempenho. Casos em que o nome não pôde ser identificado ou inferido são mantidos como NÃO LOCALIZADO ou INDETERMINADO, preservando a consistência da base.

---

### Seção [11] — Diferença percentual no desfecho por Gênero
Nesta etapa são calculadas e apresentadas as diferenças percentuais entre os resultados das sentenças (procedente, improcedente e procedente em parte) de acordo com o gênero inferido do magistrado. A análise é estritamente descritiva e exploratória, sem pressupor causalidade, servindo como subsídio para observações empíricas sobre possíveis assimetrias estatísticas no corpus analisado.

---

### Seção [12] — Preparação Centralizada (configurações) para Modelos de Análise de Sentimentos
- Padronização da entrada de dados (acentos normalizados, espaços limpos, sem retirar informações jurídicas importantes). A classificação já vira sentimento (procedente = positivo; improcedente = negativo; procedente em parte = neutro).
- Dados divididos em treino e teste.

---

### Seção [13] — Análise de Sentimentos - Modelo 1: Naive Bayes com TF-IDF
Esta seção apresenta a aplicação do classificador Naive Bayes, utilizando como representação textual a vetorização TF-IDF. O modelo é treinado sobre o conjunto de treino (80% dos dados) e avaliado sobre o conjunto de teste (20%), sendo calculadas métricas clássicas de desempenho, como acurácia, precisão, recall e F1-score. O Naive Bayes é adotado como baseline por sua simplicidade computacional e ampla utilização em tarefas de classificação de texto.

---

### Seção [14] — Análise de Sentimentos - Modelo 2: BERT (pré-treinado sem fine-tuning)
Nesta seção é aplicado o Multilingual BERT (mBERT) em sua versão pré-treinada, sem realização de fine-tuning específico para o domínio jurídico. O modelo é utilizado para gerar representações contextuais do texto bruto das sentenças, funcionando como um baseline neural de deep learning. Os resultados obtidos permitem avaliar o desempenho de um modelo contextual moderno em comparação com abordagens clássicas baseadas em TF-IDF.

---

### Seção [15] — Análise de Sentimentos - Modelo 3: Regressão Logística (TF-IDF)
Esta etapa descreve a aplicação do classificador Regressão Logística, também baseado na representação TF-IDF do texto. O modelo é treinado e avaliado seguindo a mesma divisão de dados e métricas utilizadas nos demais experimentos, permitindo uma comparação direta com o Naive Bayes e com o modelo BERT. A Regressão Logística é empregada por sua robustez em espaços vetoriais de alta dimensionalidade e desempenho consistente em tarefas de PLN.

---

### Seção [16] — Comparação entre os Modelos
A seção final consolida os resultados obtidos pelos três modelos avaliados — Naive Bayes, Regressão Logística e BERT Multilíngue — por meio de tabelas comparativas e métricas padronizadas. Essa análise permite discutir diferenças de desempenho entre métodos clássicos e modelos de deep learning, evidenciando ganhos, limitações e implicações metodológicas no contexto da classificação de sentenças judiciais.

---

## Como alterar a quantidade de processos para scraping no portal de serviços e-SAJ (TJAL):

> ⚠️ **Esta é a principal configuração para controle do tamanho do corpus.**
>
> Na **Seção [8]**, localize o trecho abaixo:
>
> ```python
> # --- SCRAPING NOVO - QUANTIDADE DE PROCESSOS - ALTERAR AQUI! ---
> if len(a_raspar) > 0:
>     print(f"[INFO] Iniciando scraping de {len(a_raspar)} novos processos...")
>     df_scrape = scrape_lote(
>         a_raspar,
>         navegador_preferido=None,
>         max_itens=2000   # <-- ALTERE O TOTAL DE PROCESSOS AQUI
>     )
> ```
>
> Substitua o valor `2000` pelo número desejado, por exemplo: `100`.

## Licença
Este projeto é destinado a fins acadêmicos e educacionais.