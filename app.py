import concurrent.futures as cf
from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Análise de Portefólio", page_icon="📈", layout="wide")

COLUNAS = ["Ticker", "Quantidade", "Preço Médio de Compra", "Tipo de Ativo", "Setor", "País"]
TIPOS_VALIDOS = ["Ações", "ETF", "Crypto", "P2P"]
CACHE_SEGUNDOS = 300  # as cotações são reutilizadas durante 5 minutos
DIAS_UTEIS_ANO = 252
MIN_OBSERVACOES = 30  # mínimo de dias para calcular métricas de risco

BENCHMARKS = {
    "S&P 500": "^GSPC",
    "VWCE (FTSE All-World)": "VWCE.DE",
    "MSCI World (IWDA)": "IWDA.AS",
}


# ---------------------------- Formatação ----------------------------
def fmt_eur(valor: float) -> str:
    """Formata em estilo português: 1 234,56 €"""
    return f"{valor:,.2f}".replace(",", " ").replace(".", ",") + " €"


def fmt_pct(valor: float, sinal: bool = True) -> str:
    texto = f"{valor:+.2f}" if sinal else f"{valor:.2f}"
    return texto.replace(".", ",") + "%"


def fmt_num(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def cor_pl(valor):
    if pd.isna(valor) or valor == 0:
        return ""
    return "color: #2e9e5b" if valor > 0 else "color: #d64545"


# ---------------------------- Carregamento do CSV ----------------------------
def para_numero(serie: pd.Series) -> pd.Series:
    """Converte colunas numéricas aceitando vírgula decimal (formato PT)."""
    if serie.dtype == object:
        serie = serie.astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(serie, errors="coerce")


@st.cache_data
def carregar_csv(ficheiro) -> pd.DataFrame:
    # sep=None deteta automaticamente se o separador é "," ou ";" (Excel PT usa ";")
    df = pd.read_csv(ficheiro, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def validar(df: pd.DataFrame) -> pd.DataFrame:
    em_falta = [c for c in COLUNAS if c not in df.columns]
    if em_falta:
        st.error(f"Faltam colunas no ficheiro: {', '.join(em_falta)}")
        st.stop()

    # "Moeda" é opcional: moeda do preço médio de compra (por defeito, EUR)
    if "Moeda" not in df.columns:
        df["Moeda"] = "EUR"
    extras = ["Data Compra", "Data Venda", "Preço Venda"]  # opcionais: permitem a evolução real
    for e in extras:
        if e not in df.columns:
            df[e] = pd.NA
    df = df[COLUNAS + ["Moeda"] + extras].copy()
    df["Moeda"] = df["Moeda"].fillna("EUR").astype(str).str.strip().str.upper()
    df["Data Compra"] = pd.to_datetime(df["Data Compra"], dayfirst=True, errors="coerce")
    df["Data Venda"] = pd.to_datetime(df["Data Venda"], dayfirst=True, errors="coerce")
    df["Preço Venda"] = para_numero(df["Preço Venda"])

    df["Quantidade"] = para_numero(df["Quantidade"])
    df["Preço Médio de Compra"] = para_numero(df["Preço Médio de Compra"])

    invalidas = df[df[["Quantidade", "Preço Médio de Compra"]].isna().any(axis=1)]
    if not invalidas.empty:
        st.warning(f"{len(invalidas)} linha(s) com números inválidos foram ignoradas.")
        df = df.drop(invalidas.index)

    for col in ["Ticker", "Tipo de Ativo", "Setor", "País"]:
        df[col] = df[col].astype(str).str.strip()

    desconhecidos = set(df["Tipo de Ativo"]) - set(TIPOS_VALIDOS)
    if desconhecidos:
        st.info(f"Tipos de ativo fora do esperado ({', '.join(TIPOS_VALIDOS)}): {', '.join(sorted(desconhecidos))}")
    return df


# ---------------------------- Cotações atuais (Yahoo Finance) ----------------------------
def _preco(ticker: str):
    """Devolve (preço, moeda) ou (None, None) se não houver cotação."""
    try:
        info = yf.Ticker(ticker).fast_info
        preco = float(info["last_price"])
        moeda = info["currency"]
        if moeda in ("GBp", "GBX"):  # Londres cota em pence
            preco, moeda = preco / 100, "GBP"
        return preco, moeda
    except Exception:
        return None, None


def _taxa_para_eur(moeda: str):
    if moeda == "EUR":
        return 1.0
    try:
        return float(yf.Ticker(f"{moeda}EUR=X").fast_info["last_price"])
    except Exception:
        return None


@st.cache_data(ttl=CACHE_SEGUNDOS, show_spinner=False)
def obter_precos(tickers: tuple):
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        precos = dict(zip(tickers, ex.map(_preco, tickers)))
    return precos, datetime.now().strftime("%H:%M")


@st.cache_data(ttl=CACHE_SEGUNDOS, show_spinner=False)
def obter_taxas(moedas: tuple) -> dict:
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(moedas, ex.map(_taxa_para_eur, moedas)))


# ---------------------------- Notícias ----------------------------
def _para_data(valor):
    try:
        if isinstance(valor, (int, float)):
            return pd.to_datetime(valor, unit="s", utc=True).tz_convert("Europe/Lisbon")
        return pd.to_datetime(valor, utc=True).tz_convert("Europe/Lisbon")
    except Exception:
        return pd.NaT


def _noticias(ticker: str, n: int = 3) -> list:
    try:
        itens = yf.Ticker(ticker).news or []
    except Exception:
        return []
    noticias = []
    for it in itens:
        c = it.get("content", it)  # formato novo ("content") ou antigo
        titulo = c.get("title")
        fonte = (c.get("provider") or {}).get("displayName") or c.get("publisher") or "—"
        url = (
            (c.get("canonicalUrl") or {}).get("url")
            or (c.get("clickThroughUrl") or {}).get("url")
            or c.get("link")
        )
        data = _para_data(c.get("pubDate") or it.get("providerPublishTime"))
        if titulo and url:
            noticias.append({"titulo": titulo, "fonte": fonte, "url": url, "data": data})
    noticias.sort(key=lambda x: x["data"] if pd.notna(x["data"]) else pd.Timestamp(0, tz="UTC"), reverse=True)
    return noticias[:n]


@st.cache_data(ttl=CACHE_SEGUNDOS, show_spinner=False)
def obter_noticias(tickers: tuple):
    """Notícias e hora a que foram obtidas. Mesmo prazo de validade das cotações."""
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        noticias = dict(zip(tickers, ex.map(_noticias, tickers)))
    return noticias, datetime.now().strftime("%H:%M")


# ---------------------------- Dividendos ----------------------------
def _dividendos(ticker: str):
    """Dividendos por ação, indexados pela data ex-dividendo (na moeda de cotação)."""
    try:
        s = yf.Ticker(ticker).dividends
        if s is None or s.empty:
            return pd.Series(dtype=float)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def obter_dividendos(tickers: tuple) -> dict:
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(tickers, ex.map(_dividendos, tickers)))


# ---------------------------- Histórico (12 meses) ----------------------------
def _historico(ticker: str, inicio=None, ajustado: bool = True):
    """Devolve (série de fecho, moeda) ou (None, None). Sem 'inicio' usa os últimos 12 meses."""
    try:
        t = yf.Ticker(ticker)
        if inicio:
            s = t.history(start=inicio, auto_adjust=ajustado)["Close"]
        else:
            s = t.history(period="1y", auto_adjust=ajustado)["Close"]
        if s.empty:
            return None, None
        moeda = (t.history_metadata or {}).get("currency")
        if moeda in ("GBp", "GBX"):  # Londres cota em pence
            s, moeda = s / 100, "GBP"
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s, moeda
    except Exception:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def obter_historico_eur(tickers: tuple, inicio=None, ajustado: bool = True):
    """Preços de fecho convertidos para EUR. Devolve ({ticker: Series}, {moeda: taxa para EUR})."""
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        brutos = dict(zip(tickers, ex.map(lambda t: _historico(t, inicio, ajustado), tickers)))
        moedas = sorted({m for s, m in brutos.values() if s is not None and m and m != "EUR"})
        fx = dict(zip(moedas, ex.map(lambda m: _historico(f"{m}EUR=X", inicio)[0], moedas)))
    resultado = {}
    for t, (s, m) in brutos.items():
        if s is None:
            continue
        if m and m != "EUR":
            taxa = fx.get(m)
            if taxa is None:
                continue
            s = (s * taxa.reindex(s.index, method="ffill")).dropna()
        if len(s) > 1:
            resultado[t] = s
    return resultado, {m: s for m, s in fx.items() if s is not None}


def _na_grelha(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Série alinhada com a grelha de dias úteis (preenche feriados e o início)."""
    return s.reindex(s.index.union(idx)).ffill().reindex(idx).bfill()


def evolucao_real(lotes: pd.DataFrame, precos: dict, fx: dict):
    """Evolução real da carteira (em EUR) a partir dos lotes com data de compra e de venda.
    Cada lote entra ao preço de compra e sai ao preço de venda; entre as duas datas vale o fecho diário.
    Rentabilidade = ponderada no tempo (não é distorcida por compras e vendas)."""
    sem_precos = set(lotes["Ticker"]) - set(precos)  # tickers sem histórico no Yahoo
    lotes = lotes[lotes["Ticker"].isin(precos)]
    if lotes.empty:
        return None, sorted(sem_precos)
    fim = max(s.index[-1] for s in precos.values())
    idx = pd.bdate_range(lotes["Data Compra"].min(), fim)
    n = len(idx)
    if n < 2:
        return None, []
    px = {t: _na_grelha(s, idx).values for t, s in precos.items()}
    taxas = {m: _na_grelha(s, idx).values for m, s in fx.items()}

    valor, custo, pnl, capital = (np.zeros(n) for _ in range(4))
    excluidos = set()
    for l in lotes.to_dict("records"):
        t, q, m = l["Ticker"], l["Quantidade"], l["Moeda"]
        taxa = np.ones(n) if m == "EUR" else taxas.get(m)
        if taxa is None:
            excluidos.add(t)
            continue
        p0 = idx.searchsorted(l["Data Compra"])
        fechado = pd.notna(l["Data Venda"])
        p1 = min(idx.searchsorted(l["Data Venda"]), n - 1) if fechado else n - 1
        if p0 >= n or p1 < p0:
            continue
        px_eur = px[t] * taxa
        entrada = l["Preço Médio de Compra"] * taxa[p0]
        fim_dia = px_eur[p0:p1 + 1].copy()
        if fechado and pd.notna(l["Preço Venda"]):
            fim_dia[-1] = l["Preço Venda"] * taxa[p1]
        ini_dia = np.concatenate(([entrada], px_eur[p0:p1]))
        pnl[p0:p1 + 1] += q * (fim_dia - ini_dia)
        capital[p0:p1 + 1] += q * ini_dia
        mantido = slice(p0, p1 if fechado else p1 + 1)
        valor[mantido] += q * px_eur[mantido]
        custo[mantido] += q * entrada

    retorno = np.where(capital > 0, pnl / np.where(capital > 0, capital, 1), 0.0)
    out = pd.DataFrame({"valor": valor, "custo": custo, "pnl": pnl, "retorno": retorno}, index=idx)
    out["pnl_acum"] = out["pnl"].cumsum()
    out["twr"] = ((1 + out["retorno"]).cumprod() - 1) * 100
    return out, sorted(excluidos | sem_precos)


def calcular_dividendos(lotes: pd.DataFrame, divs: dict, fx: dict) -> pd.DataFrame:
    """Dividendos brutos a que cada lote teve direito.
    Regra: tinha de ter comprado antes da data ex-dividendo e não ter vendido antes dela
    (vender no próprio dia ex-dividendo ainda dá direito)."""
    registos = []
    for l in lotes.to_dict("records"):
        serie = divs.get(l["Ticker"])
        if serie is None or serie.empty:
            continue
        compra, venda, m = l["Data Compra"], l["Data Venda"], l["Moeda"]
        taxa_serie = fx.get(m) if m != "EUR" else None
        if m != "EUR" and taxa_serie is None:
            continue
        for ex_data, por_acao in serie.items():
            if ex_data <= compra:
                continue
            if pd.notna(venda) and ex_data > venda:
                continue
            taxa = 1.0
            if taxa_serie is not None:
                taxa = taxa_serie.asof(ex_data)
                if pd.isna(taxa):
                    taxa = taxa_serie.iloc[0]
            bruto = l["Quantidade"] * por_acao
            registos.append(
                {
                    "Data ex-dividendo": ex_data,
                    "Ticker": l["Ticker"],
                    "Quantidade": l["Quantidade"],
                    "Dividendo por ação": por_acao,
                    "Moeda": m,
                    "Bruto (moeda)": bruto,
                    "Bruto (€)": bruto * taxa,
                    "Posição": "Encerrada" if pd.notna(venda) else "Aberta",
                }
            )
    colunas = ["Data ex-dividendo", "Ticker", "Quantidade", "Dividendo por ação", "Moeda",
               "Bruto (moeda)", "Bruto (€)", "Posição"]
    return pd.DataFrame(registos, columns=colunas).sort_values("Data ex-dividendo", ignore_index=True)


def agregar_posicoes(df: pd.DataFrame) -> pd.DataFrame:
    """Junta os lotes abertos do mesmo ativo numa só posição (preço médio ponderado)."""
    d = df.copy()
    d["_custo"] = d["Quantidade"] * d["Preço Médio de Compra"]
    g = d.groupby(["Ticker", "Moeda"], as_index=False).agg(
        Quantidade=("Quantidade", "sum"),
        _custo=("_custo", "sum"),
        **{"Tipo de Ativo": ("Tipo de Ativo", "first"), "Setor": ("Setor", "first"), "País": ("País", "first")},
    )
    g["Preço Médio de Compra"] = g["_custo"] / g["Quantidade"]
    return g.drop(columns="_custo")[COLUNAS + ["Moeda"]]


# ---------------------------- Performance e risco ----------------------------
def retornos_carteira(historicos: dict, pesos: pd.Series):
    """Retornos diários da carteira com os pesos atuais constantes (rebalanceamento diário).
    Só dias úteis: o fim de semana das crypto entra na segunda-feira."""
    tickers = [t for t in pesos.index if t in historicos]
    if not tickers:
        return None
    precos = pd.concat({t: historicos[t] for t in tickers}, axis=1).sort_index().ffill()
    precos = precos[precos.index.dayofweek < 5]
    rets = precos / precos.shift(1) - 1
    w = pesos[tickers]
    disponivel = rets.notna()  # ativos que já tinham preço nesse dia
    den = (disponivel * w).sum(axis=1).replace(0, float("nan"))
    r = (rets.fillna(0) * w).sum(axis=1) / den
    return r.fillna(0)


def retornos_benchmark(serie: pd.Series, indice: pd.DatetimeIndex) -> pd.Series:
    """Retornos diários do benchmark alinhados com as datas da carteira."""
    p = serie.reindex(serie.index.union(indice)).ffill()
    p = p[p.index.dayofweek < 5]
    return (p / p.shift(1) - 1).reindex(indice)


def acumulada(r: pd.Series) -> pd.Series:
    """Retorno acumulado (%) a partir de retornos diários, a começar em 0."""
    c = ((1 + r).cumprod() - 1) * 100
    zero = pd.Series([0.0], index=[r.index[0] - pd.Timedelta(days=1)])
    return pd.concat([zero, c])


def metricas_risco(r: pd.Series, taxa_livre_risco: float) -> dict:
    vol = r.std() * sqrt(DIAS_UTEIS_ANO)
    excesso = r - taxa_livre_risco / DIAS_UTEIS_ANO
    sharpe = excesso.mean() / excesso.std() * sqrt(DIAS_UTEIS_ANO) if excesso.std() > 0 else float("nan")
    valor = (1 + r).cumprod()
    pico = valor.cummax().clip(lower=1.0)
    drawdown = valor / pico - 1
    return {
        "vol": vol,
        "sharpe": sharpe,
        "mdd": drawdown.min(),
        "data_mdd": drawdown.idxmin(),
        "drawdown": drawdown,
        "retorno": valor.iloc[-1] - 1,
    }


# ---------------------------- Posições e alocação ----------------------------
def calcular_posicoes(df: pd.DataFrame, precos: dict, taxas: dict) -> pd.DataFrame:
    registos = []
    for r in df.to_dict("records"):
        custo_eur = r["Preço Médio de Compra"] * (taxas.get(r["Moeda"]) or 1.0)

        preco_eur = None
        if r["Tipo de Ativo"] != "P2P":
            preco, moeda = precos.get(r["Ticker"], (None, None))
            taxa = taxas.get(moeda) if moeda else None
            if preco is not None and taxa:
                preco_eur = preco * taxa
        cotado = preco_eur is not None
        if not cotado:  # P2P ou ticker sem cotação: assume o preço de compra (P&L = 0)
            preco_eur = custo_eur

        investido = r["Quantidade"] * custo_eur
        atual = r["Quantidade"] * preco_eur
        pl = atual - investido
        registos.append(
            {
                "Ticker": r["Ticker"],
                "Quantidade": r["Quantidade"],
                "Preço Compra (€)": custo_eur,
                "Preço Atual (€)": preco_eur,
                "Valor Investido": investido,
                "Valor Atual": atual,
                "P&L (€)": pl,
                "P&L (%)": (pl / investido * 100) if investido else 0.0,
                "Tipo de Ativo": r["Tipo de Ativo"],
                "Setor": r["Setor"],
                "País": r["País"],
                "Cotação": "Ao vivo" if cotado else "Sem cotação",
            }
        )
    return pd.DataFrame(registos)


def grafico_alocacao(df: pd.DataFrame, coluna: str, coluna_valor: str, tipo: str):
    agg = df.groupby(coluna, as_index=False)[coluna_valor].sum()
    agg["Peso (%)"] = agg[coluna_valor] / agg[coluna_valor].sum() * 100
    if tipo == "Pie Chart":
        fig = px.pie(agg, names=coluna, values=coluna_valor, hole=0.4)
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}<br>%{value:,.2f} €<br>%{percent}")
    else:
        fig = px.treemap(agg, path=[coluna], values=coluna_valor, color=coluna_valor, color_continuous_scale="Blues")
        fig.update_traces(
            texttemplate="%{label}<br>%{value:,.0f} €<br>%{percentRoot:.1%}",
            hovertemplate="%{label}<br>%{value:,.2f} €<br>%{percentRoot:.1%}",
        )
        fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h"))
    return fig, agg


def mostrar_alocacao(df: pd.DataFrame, coluna: str, coluna_valor: str, tipo: str):
    fig, agg = grafico_alocacao(df, coluna, coluna_valor, tipo)
    st.plotly_chart(fig, width="stretch")
    tabela = agg.sort_values(coluna_valor, ascending=False).copy()
    tabela[coluna_valor] = tabela[coluna_valor].map(fmt_eur)
    tabela["Peso (%)"] = tabela["Peso (%)"].map(lambda x: f"{x:.1f}%".replace(".", ","))
    st.dataframe(tabela, hide_index=True, width="stretch")


# ---------------------------- Interface: barra lateral ----------------------------
st.title("📈 Análise do meu Portefólio")

with st.sidebar:
    st.header("Dados")
    ficheiro = st.file_uploader("Carrega o teu portefólio (CSV)", type=["csv"])
    tipo_grafico = st.radio("Tipo de gráfico", ["Pie Chart", "Treemap"], horizontal=True)
    base_alocacao = st.radio("Alocação com base em", ["Valor Atual", "Valor Investido"], horizontal=True)
    nome_bench = st.selectbox("Benchmark", list(BENCHMARKS))
    if st.button("🔄 Atualizar dados"):
        obter_precos.clear()
        obter_taxas.clear()
        obter_noticias.clear()
        obter_historico_eur.clear()
        obter_dividendos.clear()
    st.caption("Colunas necessárias: " + ", ".join(COLUNAS))
    st.caption("Opcional: Moeda (por defeito EUR), Data Compra, Data Venda e Preço Venda (para a evolução real da carteira).")

if ficheiro is None:
    st.info("👈 Carrega um ficheiro CSV para começar. Abre o menu lateral (seta no canto superior esquerdo) se estiveres no telemóvel.")
    st.stop()

dados = validar(carregar_csv(ficheiro))
if dados.empty:
    st.error("Não há dados válidos no ficheiro.")
    st.stop()

# Posições atuais = linhas sem data de venda, juntas por ativo; todas as linhas (lotes) servem para o histórico
abertas = dados[dados["Data Venda"].isna()]
if abertas.empty:
    st.error("Não há posições abertas no ficheiro (todas as linhas têm data de venda).")
    st.stop()
base = agregar_posicoes(abertas)
lotes = dados[dados["Data Compra"].notna()]

# ---------------------------- Cálculos globais ----------------------------
tickers_cotaveis = tuple(sorted(base.loc[base["Tipo de Ativo"] != "P2P", "Ticker"].unique()))
ticker_bench = BENCHMARKS[nome_bench]

with st.spinner("A obter cotações e histórico..."):
    precos, hora = obter_precos(tickers_cotaveis)
    moedas = {m for _, m in precos.values() if m} | set(base["Moeda"]) | {"EUR"}
    taxas = obter_taxas(tuple(sorted(moedas)))
    df = calcular_posicoes(base, precos, taxas)

    pesos = df[df["Tipo de Ativo"] != "P2P"].groupby("Ticker")["Valor Atual"].sum()
    hist, _ = obter_historico_eur(tuple(sorted(pesos.index)))
    serie_b = obter_historico_eur((ticker_bench,))[0].get(ticker_bench)

    # Evolução real (a partir das datas de compra/venda do ficheiro)
    evol, excluidos_evol, serie_b_real, divs_df = None, [], None, None
    if not lotes.empty:
        inicio_hist = (lotes["Data Compra"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        precos_h, fx_h = obter_historico_eur(tuple(sorted(lotes["Ticker"].unique())), inicio_hist, False)
        evol, excluidos_evol = evolucao_real(lotes, precos_h, fx_h)
        serie_b_real = obter_historico_eur((ticker_bench,), inicio_hist, True)[0].get(ticker_bench)
        divs_df = calcular_dividendos(lotes, obter_dividendos(tuple(sorted(lotes["Ticker"].unique()))), fx_h)

# Retornos diários da carteira e do benchmark, alinhados (usados no Dashboard e no Risco)
comum = None
sem_hist, cobertura = list(pesos.index), 0.0
r_p = retornos_carteira(hist, pesos)
if r_p is not None and serie_b is not None:
    r_b = retornos_benchmark(serie_b, r_p.index)
    comum = pd.concat([r_p, r_b], axis=1, keys=["p", "b"]).dropna()
    comum = comum[comum.index > r_p.index[0]]  # o 1.º dia não tem retorno real
    if len(comum) < MIN_OBSERVACOES:
        comum = None
    com_hist = [t for t in pesos.index if t in hist]
    sem_hist = [t for t in pesos.index if t not in hist]
    cobertura = pesos[com_hist].sum() / df["Valor Atual"].sum() * 100

nota_cobertura = f"Cobre {cobertura:.0f}% do valor da carteira".replace(".", ",")
if sem_hist:
    nota_cobertura += " (sem histórico: " + ", ".join(sem_hist) + ")"
nota_cobertura += ". P2P fica de fora. Simulação em € com os pesos atuais mantidos ao longo dos 12 meses."

sem_cotacao = df[(df["Cotação"] == "Sem cotação") & (df["Tipo de Ativo"] != "P2P")]["Ticker"].tolist()
if sem_cotacao:
    st.warning(
        "Sem cotação para: " + ", ".join(sem_cotacao) + ". Foi usado o preço de compra (P&L = 0). "
        "Confirma o símbolo no Yahoo Finance (ex.: EDP.LS, VWCE.DE, BTC-USD)."
    )

# ---------------------------- Separadores ----------------------------
tab_dash, tab_aloc, tab_risco, tab_div, tab_news = st.tabs(
    ["Dashboard Geral", "Alocação por Setor/País", "Métricas de Risco", "Dividendos", "Notícias do Mercado"]
)

# ======== Dashboard Geral ========
with tab_dash:
    st.caption(f"Cotações obtidas às {hora} · Yahoo Finance (podem ter alguns minutos de atraso) · valores em €")

    investido_total = df["Valor Investido"].sum()
    atual_total = df["Valor Atual"].sum()
    pl_total = atual_total - investido_total
    pl_total_pct = (pl_total / investido_total * 100) if investido_total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor investido", fmt_eur(investido_total))
    c2.metric("Valor atual", fmt_eur(atual_total))
    c3.metric("P&L (€)", fmt_eur(pl_total), delta=fmt_pct(pl_total_pct))
    c4.metric("Posições", len(df))

    st.subheader("Evolução da carteira ao longo do tempo")
    if evol is not None and len(evol) > 1:
        ult = evol.iloc[-1]
        bench_cum = None
        if serie_b_real is not None:
            g = _na_grelha(serie_b_real, evol.index)
            bench_cum = (g / g.iloc[0] - 1) * 100

        e1, e2, e3 = st.columns(3)
        e1.metric("Rentabilidade acumulada", fmt_pct(ult["twr"]),
                  help="Rentabilidade ponderada no tempo: não é afetada pelas tuas compras e vendas.")
        e2.metric("Resultado acumulado", fmt_eur(ult["pnl_acum"]),
                  help="Ganhos e perdas realizados (posições vendidas) e não realizados (posições abertas), em €.")
        if bench_cum is not None:
            e3.metric(nome_bench, fmt_pct(bench_cum.iloc[-1]), help="Mesmo período, desde o primeiro dia da carteira.")

        st.markdown("**Valor da carteira (€)**")
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(x=evol.index, y=evol["valor"], name="Valor da carteira",
                                     fill="tozeroy", line=dict(width=3)))
        fig_val.add_trace(go.Scatter(x=evol.index, y=evol["custo"], name="Capital investido",
                                     line=dict(width=2, dash="dot", shape="hv")))
        fig_val.update_traces(hovertemplate="%{y:,.2f} €")
        fig_val.update_layout(hovermode="x unified", yaxis=dict(ticksuffix=" €"),
                              legend=dict(orientation="h", y=1.1), margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_val, width="stretch")

        st.markdown("**Evolução em percentagem (%)**")
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(x=evol.index, y=evol["twr"], name="Carteira", line=dict(width=3)))
        if bench_cum is not None:
            fig_pct.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name=nome_bench,
                                         line=dict(width=2, dash="dash")))
        fig_pct.add_hline(y=0, line_width=1, line_color="gray")
        fig_pct.update_traces(hovertemplate="%{y:.2f}%")
        fig_pct.update_layout(hovermode="x unified", yaxis=dict(ticksuffix="%"),
                              legend=dict(orientation="h", y=1.1), margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pct, width="stretch")

        if bench_cum is not None:
            dif_real = ult["twr"] - bench_cum.iloc[-1]
            if dif_real >= 0:
                st.success(f"No período, a carteira bateu o {nome_bench} em {fmt_num(abs(dif_real), 1)} p.p.")
            else:
                st.info(f"No período, a carteira ficou {fmt_num(abs(dif_real), 1)} p.p. abaixo do {nome_bench}.")
        aviso_real = (
            f"Período: {evol.index[0]:%d/%m/%Y} a {evol.index[-1]:%d/%m/%Y}. Calculado com as tuas datas e preços de "
            "compra e venda e os fechos diários do Yahoo Finance, em €. Não inclui dividendos nem comissões."
        )
        if excluidos_evol:
            aviso_real += " Sem histórico (ignorados): " + ", ".join(excluidos_evol) + "."
        st.caption(aviso_real)
    else:
        if comum is None:
            st.warning("Não foi possível obter histórico suficiente para a carteira ou para o benchmark.")
        else:
            p, b = acumulada(comum["p"]), acumulada(comum["b"])
            ret_p, ret_b = p.iloc[-1], b.iloc[-1]
            dif = ret_p - ret_b

            m1, m2, m3 = st.columns(3)
            m1.metric("Carteira", fmt_pct(ret_p))
            m2.metric(nome_bench, fmt_pct(ret_b))
            m3.metric("Diferença", f"{dif:+.2f} p.p.".replace(".", ","))

            fig_evol = go.Figure()
            fig_evol.add_trace(go.Scatter(x=p.index, y=p.values, name="Carteira", line=dict(width=3)))
            fig_evol.add_trace(go.Scatter(x=b.index, y=b.values, name=nome_bench, line=dict(width=2, dash="dash")))
            fig_evol.add_hline(y=0, line_width=1, line_color="gray")
            fig_evol.update_traces(hovertemplate="%{y:.2f}%")
            fig_evol.update_layout(
                hovermode="x unified",
                yaxis=dict(ticksuffix="%"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_evol, width="stretch")

            if dif >= 0:
                st.success(f"Com os pesos atuais, a carteira bateu o {nome_bench} em {fmt_num(abs(dif), 1)} p.p.")
            else:
                st.info(f"Com os pesos atuais, a carteira ficou {fmt_num(abs(dif), 1)} p.p. abaixo do {nome_bench}.")
            st.caption(nota_cobertura + " Não reflete as tuas compras e vendas reais.")

    st.subheader("Alocação por tipo de ativo")
    mostrar_alocacao(df, "Tipo de Ativo", base_alocacao, tipo_grafico)

    st.subheader("Posições e P&L")
    mostrar = df.sort_values("Valor Atual", ascending=False)[
        ["Ticker", "Quantidade", "Preço Compra (€)", "Preço Atual (€)", "Valor Investido",
         "Valor Atual", "P&L (€)", "P&L (%)", "Cotação"]
    ]
    formatos = {c: fmt_eur for c in ["Preço Compra (€)", "Preço Atual (€)", "Valor Investido", "Valor Atual", "P&L (€)"]}
    formatos["P&L (%)"] = fmt_pct
    st.dataframe(
        mostrar.style.format(formatos).map(cor_pl, subset=["P&L (€)", "P&L (%)"]),
        hide_index=True,
        width="stretch",
    )

    graf = df.sort_values("P&L (%)", ascending=False).copy()
    graf["Resultado"] = graf["P&L (%)"].apply(lambda x: "Ganho" if x >= 0 else "Perda")
    fig_pl = px.bar(
        graf, x="Ticker", y="P&L (%)", color="Resultado",
        color_discrete_map={"Ganho": "#2e9e5b", "Perda": "#d64545"},
    )
    fig_pl.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pl, width="stretch")

# ======== Alocação por Setor/País ========
with tab_aloc:
    col_setor, col_pais = st.columns(2)
    with col_setor:
        st.subheader("Por setor")
        mostrar_alocacao(df, "Setor", base_alocacao, tipo_grafico)
    with col_pais:
        st.subheader("Por país")
        mostrar_alocacao(df, "País", base_alocacao, tipo_grafico)

# ======== Métricas de Risco ========
with tab_risco:
    taxa_txt = st.radio("Taxa livre de risco (para o Sharpe Ratio)", ["2%", "3%"], horizontal=True)
    taxa_lr = float(taxa_txt.strip("%")) / 100

    if comum is None:
        st.warning(
            f"Não há histórico suficiente (mínimo {MIN_OBSERVACOES} dias úteis) para calcular o risco. "
            "Confirma os símbolos no CSV e carrega em «Atualizar dados»."
        )
    else:
        rp = metricas_risco(comum["p"], taxa_lr)
        rb = metricas_risco(comum["b"], taxa_lr)
        beta = comum["p"].cov(comum["b"]) / comum["b"].var()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Volatilidade anualizada", fmt_pct(rp["vol"] * 100, sinal=False),
            help="Desvio-padrão dos retornos diários da carteira × √252.",
        )
        k2.metric(
            f"Beta vs. {nome_bench}", fmt_num(beta),
            help="1 = oscila como o benchmark; acima de 1 amplifica os movimentos; abaixo de 1 amortece.",
        )
        k3.metric(
            "Sharpe Ratio", fmt_num(rp["sharpe"]),
            help=f"Retorno em excesso sobre a taxa livre de risco ({taxa_txt}) por unidade de volatilidade.",
        )
        k4.metric(
            "Queda máxima (12M)", fmt_pct(rp["mdd"] * 100, sinal=False),
            help="Maior queda desde um máximo até ao mínimo seguinte.",
        )
        st.caption(
            f"Queda máxima atingida em {rp['data_mdd']:%d/%m/%Y}. "
            f"Período analisado: {comum.index[0]:%d/%m/%Y} a {comum.index[-1]:%d/%m/%Y} ({len(comum)} dias úteis)."
        )

        if beta > 1.05:
            st.info(f"Beta de {fmt_num(beta)}: a carteira tende a subir e a descer mais do que o {nome_bench}.")
        elif beta < 0.95:
            st.info(f"Beta de {fmt_num(beta)}: a carteira tende a oscilar menos do que o {nome_bench}.")
        else:
            st.info(f"Beta de {fmt_num(beta)}: a carteira oscila de forma semelhante ao {nome_bench}.")

        st.subheader("Queda desde o máximo (drawdown)")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=rp["drawdown"].index, y=rp["drawdown"].values * 100, name="Carteira",
            fill="tozeroy", line=dict(width=2),
        ))
        fig_dd.add_trace(go.Scatter(
            x=rb["drawdown"].index, y=rb["drawdown"].values * 100, name=nome_bench,
            line=dict(width=2, dash="dash"),
        ))
        fig_dd.update_traces(hovertemplate="%{y:.2f}%")
        fig_dd.update_layout(
            hovermode="x unified",
            yaxis=dict(ticksuffix="%"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_dd, width="stretch")

        st.subheader(f"Carteira vs. {nome_bench}")
        comparacao = pd.DataFrame(
            {
                "Métrica": ["Retorno no período", "Volatilidade anualizada", "Sharpe Ratio", "Queda máxima", "Beta"],
                "Carteira": [
                    fmt_pct(rp["retorno"] * 100), fmt_pct(rp["vol"] * 100, sinal=False),
                    fmt_num(rp["sharpe"]), fmt_pct(rp["mdd"] * 100, sinal=False), fmt_num(beta),
                ],
                nome_bench: [
                    fmt_pct(rb["retorno"] * 100), fmt_pct(rb["vol"] * 100, sinal=False),
                    fmt_num(rb["sharpe"]), fmt_pct(rb["mdd"] * 100, sinal=False), "1,00",
                ],
            }
        )
        st.dataframe(comparacao, hide_index=True, width="stretch")
        st.caption(
            nota_cobertura + " Métricas calculadas com retornos diários em dias úteis (252 por ano). "
            "Resultados passados não garantem resultados futuros."
        )

# ======== Dividendos ========
with tab_div:
    if divs_df is None:
        st.info("Para calcular os dividendos, o CSV precisa das colunas Data Compra (e Data Venda nas posições vendidas).")
    elif divs_df.empty:
        st.info("Nenhum dos teus ativos pagou dividendos durante o período em que os detiveste (segundo o Yahoo Finance).")
    else:
        total_div = divs_df["Bruto (€)"].sum()
        por_ativo = (
            divs_df.groupby("Ticker", as_index=False)
            .agg(**{"Pagamentos": ("Ticker", "size"), "Bruto (€)": ("Bruto (€)", "sum")})
            .sort_values("Bruto (€)", ascending=False)
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Dividendos brutos recebidos", fmt_eur(total_div))
        d2.metric("Pagamentos", len(divs_df))
        d3.metric("Maior pagador", por_ativo.iloc[0]["Ticker"], help=fmt_eur(por_ativo.iloc[0]["Bruto (€)"]))

        st.markdown("**Por mês (data ex-dividendo)**")
        mensal = divs_df.assign(Mês=divs_df["Data ex-dividendo"].dt.strftime("%Y-%m"))
        mensal = mensal.groupby(["Mês", "Ticker"], as_index=False)["Bruto (€)"].sum()
        fig_m = px.bar(mensal, x="Mês", y="Bruto (€)", color="Ticker")
        fig_m.update_traces(hovertemplate="%{x}<br>%{y:,.2f} €")
        fig_m.update_layout(yaxis=dict(ticksuffix=" €"), legend=dict(orientation="h", y=1.15),
                            margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_m, width="stretch")

        st.markdown("**Por ativo**")
        fig_a = px.bar(por_ativo, x="Bruto (€)", y="Ticker", orientation="h")
        fig_a.update_traces(hovertemplate="%{y}<br>%{x:,.2f} €")
        fig_a.update_layout(xaxis=dict(ticksuffix=" €"), yaxis=dict(autorange="reversed"),
                            margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_a, width="stretch")
        resumo = por_ativo.copy()
        resumo["Bruto (€)"] = resumo["Bruto (€)"].map(fmt_eur)
        st.dataframe(resumo, hide_index=True, width="stretch")

        with st.expander("Ver todos os pagamentos"):
            detalhe = divs_df.sort_values("Data ex-dividendo", ascending=False).copy()
            detalhe["Data ex-dividendo"] = detalhe["Data ex-dividendo"].dt.strftime("%d/%m/%Y")
            detalhe["Dividendo por ação"] = detalhe.apply(
                lambda r: f"{r['Dividendo por ação']:.4f} {r['Moeda']}".replace(".", ","), axis=1)
            detalhe["Bruto (moeda)"] = detalhe.apply(
                lambda r: f"{r['Bruto (moeda)']:.2f} {r['Moeda']}".replace(".", ","), axis=1)
            detalhe["Bruto (€)"] = detalhe["Bruto (€)"].map(fmt_eur)
            st.dataframe(detalhe.drop(columns="Moeda"), hide_index=True, width="stretch")

    sem_div = sorted(set(lotes["Ticker"]) - set(divs_df["Ticker"])) if divs_df is not None else []
    if sem_div:
        st.caption("Sem dividendos no período em que foram detidos: " + ", ".join(sem_div) + ".")
    st.caption(
        "Valores brutos (antes de impostos e retenção na fonte). Contam os dividendos cuja data ex-dividendo caiu "
        "entre a compra e a venda de cada lote; convertidos para € à taxa da data ex-dividendo. É uma estimativa com "
        "dados do Yahoo Finance: o dinheiro entra na data de pagamento e pode diferir do extrato da corretora."
    )

# ======== Notícias do Mercado ========
with tab_news:
    with st.spinner("A obter notícias..."):
        noticias, hora_noticias = obter_noticias(tickers_cotaveis)
    st.caption(f"Notícias atualizadas às {hora_noticias} · as 3 mais recentes de cada ativo (P2P não tem notícias).")
    for t in tickers_cotaveis:
        with st.expander(t):
            itens = noticias.get(t, [])
            if not itens:
                st.write("Sem notícias disponíveis.")
            for n in itens:
                titulo = n["titulo"].replace("[", "(").replace("]", ")")
                data_txt = "" if pd.isna(n["data"]) else f" · {n['data']:%d/%m/%Y %H:%M}"
                st.markdown(f"**[{titulo}]({n['url']})**  \n{n['fonte']}{data_txt}")
