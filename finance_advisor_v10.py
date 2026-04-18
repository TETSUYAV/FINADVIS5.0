import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.covariance import LedoitWolf

# ==========================================
# 1. CONFIGURATION GLOBALE
# ==========================================
st.set_page_config(
    page_title="Finance Advisor 10.0 — Institutional Grade",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom — Aesthetic: Dark Institutional / Bloomberg Terminal inspired
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp {
    background-color: #0a0e1a;
    color: #c8d6e5;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.03em;
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.risk-badge-green { color: #34d399; font-weight: 600; }
.risk-badge-orange { color: #fb923c; font-weight: 600; }
.risk-badge-red { color: #f87171; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] {
    background-color: #111827;
    border-bottom: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="border-bottom: 1px solid #1e3a5f; padding-bottom: 1rem; margin-bottom: 1.5rem;">
    <h1 style="color: #e2e8f0; margin:0; font-size: 1.8rem;">📐 FINANCE ADVISOR <span style="color:#38bdf8;">v10.0</span></h1>
    <p style="color: #64748b; margin: 0.3rem 0 0 0; font-size: 0.85rem; font-family: 'IBM Plex Mono', monospace;">
        Risk Parity · Black-Litterman · Ledoit-Wolf · CVaR · Max Drawdown · TER · Rebalancing · Fiscalité PEA/CTO
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. UNIVERS D'INVESTISSEMENT
# ==========================================
ASSET_UNIVERSE = {
    "CAC 40": [
        "MC.PA", "OR.PA", "TTE.PA", "RMS.PA", "SAN.PA", "AIR.PA",
        "SU.PA", "AI.PA", "BNP.PA", "SAF.PA", "CAP.PA", "DSY.PA",
        "CS.PA", "DG.PA", "EN.PA", "HO.PA"
    ],
    "DAX 40": [
        "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MBG.DE", "BMW.DE",
        "MUV2.DE", "BAS.DE", "IFX.DE", "BAYN.DE", "DB1.DE", "DBK.DE"
    ],
    "ETF Globaux": [
        "CW8.PA", "PSP5.PA", "PUST.PA", "RS2K.PA"
    ],
    "Niches / Alternatifs": [
        "PAEEM.PA", "AWAT.PA", "ENER.PA", "PTE.PA",
        "URW.PA", "IGLN.L", "BTC-EUR"
    ]
}

NON_PEA_ASSETS = {"URW.PA", "IGLN.L", "BTC-EUR"}

TICKER_NAMES = {
    "MC.PA": "LVMH", "OR.PA": "L'Oréal", "TTE.PA": "TotalEnergies",
    "RMS.PA": "Hermès", "SAN.PA": "Sanofi", "AIR.PA": "Airbus",
    "SU.PA": "Schneider Electric", "AI.PA": "Air Liquide", "BNP.PA": "BNP Paribas",
    "SAF.PA": "Safran", "CAP.PA": "Capgemini", "DSY.PA": "Dassault Systèmes",
    "CS.PA": "AXA", "DG.PA": "Vinci", "EN.PA": "Bouygues",
    "HO.PA": "Thales (Défense)", "URW.PA": "Unibail-Rodamco",
    "SAP.DE": "SAP", "SIE.DE": "Siemens", "ALV.DE": "Allianz",
    "DTE.DE": "Deutsche Telekom", "MBG.DE": "Mercedes-Benz", "BMW.DE": "BMW",
    "MUV2.DE": "Munich Re", "BAS.DE": "BASF", "IFX.DE": "Infineon",
    "BAYN.DE": "Bayer", "DB1.DE": "Deutsche Börse", "DBK.DE": "Deutsche Bank",
    "CW8.PA": "Amundi MSCI World", "PSP5.PA": "Amundi S&P 500",
    "PUST.PA": "Amundi Nasdaq-100", "RS2K.PA": "Amundi Russell 2000",
    "PAEEM.PA": "Amundi Emerging Markets", "ENER.PA": "Lyxor New Energy",
    "AWAT.PA": "Lyxor Water", "PTE.PA": "Lyxor Tech Europe",
    "IGLN.L": "Or Physique (iShares Gold)", "BTC-EUR": "Bitcoin",
    "OBLIG_SIMUL": "🛡️ Poche Sécurisée"
}

# TER annuel estimé par actif (Total Expense Ratio)
ASSET_TER = {
    "MC.PA": 0.0, "OR.PA": 0.0, "TTE.PA": 0.0, "RMS.PA": 0.0,
    "SAN.PA": 0.0, "AIR.PA": 0.0, "SU.PA": 0.0, "AI.PA": 0.0,
    "BNP.PA": 0.0, "SAF.PA": 0.0, "CAP.PA": 0.0, "DSY.PA": 0.0,
    "CS.PA": 0.0, "DG.PA": 0.0, "EN.PA": 0.0, "HO.PA": 0.0,
    "URW.PA": 0.0, "SAP.DE": 0.0, "SIE.DE": 0.0, "ALV.DE": 0.0,
    "DTE.DE": 0.0, "MBG.DE": 0.0, "BMW.DE": 0.0, "MUV2.DE": 0.0,
    "BAS.DE": 0.0, "IFX.DE": 0.0, "BAYN.DE": 0.0, "DB1.DE": 0.0,
    "DBK.DE": 0.0,
    "CW8.PA": 0.0038, "PSP5.PA": 0.0015, "PUST.PA": 0.0022, "RS2K.PA": 0.0035,
    "PAEEM.PA": 0.0045, "ENER.PA": 0.0040, "AWAT.PA": 0.0060, "PTE.PA": 0.0030,
    "IGLN.L": 0.0012, "BTC-EUR": 0.0075, "OBLIG_SIMUL": 0.0010
}

RISK_PROFILES = {
    "🛡️ Prudent": {"target": 0.05, "max_pos": 0.30},
    "⚖️ Équilibré": {"target": 0.08, "max_pos": 0.40},
    "🚀 Dynamique": {"target": 0.12, "max_pos": 0.60}
}

BENCHMARK_TICKER = "CW8.PA"  # MSCI World comme référence

# ==========================================
# 3. SIDEBAR — INTERFACE UTILISATEUR
# ==========================================
with st.sidebar:
    st.markdown("### 👤 Profil Investisseur")
    montant = st.number_input("Capital initial (€)", min_value=500, value=10000, step=500)
    horizon = st.slider("Horizon (années)", 5, 30, 10)
    profil_nom = st.selectbox("Profil de Risque", list(RISK_PROFILES.keys()))
    target_return = RISK_PROFILES[profil_nom]["target"]
    max_position = RISK_PROFILES[profil_nom]["max_pos"]

    st.markdown("---")
    st.markdown("### 🧮 Moteur d'Optimisation")
    methode_opti = st.selectbox(
        "Algorithme",
        ["Risk Parity (Parité des Risques)", "Black-Litterman", "Markowitz MV (Classique)"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Univers & Filtres")
    pea_only = st.checkbox("🟢 PEA uniquement", value=True)
    choix_univers = st.selectbox(
        "Classe d'actifs",
        ["Mix Global", "Uniquement ETF Globaux", "Uniquement Niches", "Uniquement CAC 40", "Uniquement DAX 40"]
    )
    max_actifs = st.slider("Actifs max retenus", 3, 15, 8)

    st.markdown("---")
    with st.expander("🏦 Poche Sécurisée"):
        inclure_oblig = st.checkbox("Activer", value=True)
        bond_yield = st.slider("Rendement (%)", 1.0, 8.0, 3.5, 0.1) / 100

    st.markdown("---")
    with st.expander("💸 Frais & Fiscalité"):
        frais_courtage = st.slider("Frais de courtage / réquilibrage (%)", 0.0, 0.5, 0.10, 0.01) / 100
        rebalancing_freq = st.selectbox("Fréquence de rééquilibrage", ["Annuel", "Semestriel", "Aucun"])
        enveloppe_fiscale = st.selectbox("Enveloppe fiscale", ["PEA (17.2%)", "CTO — Flat Tax (30%)", "Assurance Vie (17.2%)"])
        taux_imposition = 0.172 if "PEA" in enveloppe_fiscale or "Assurance" in enveloppe_fiscale else 0.30

    st.markdown("---")
    with st.expander("🔭 Vues Black-Litterman (optionnel)"):
        st.caption("Uniquement si BL sélectionné.")
        bl_view_asset = st.selectbox("Actif sur-performant attendu", list(TICKER_NAMES.keys())[:20])
        bl_view_alpha = st.slider("Surperformance attendue (%/an)", -10.0, 30.0, 5.0, 0.5) / 100
        bl_confidence = st.slider("Confiance dans la vue (%)", 10, 90, 50) / 100

    run_btn = st.button("⚡ Lancer l'Analyse", type="primary", use_container_width=True)

# ==========================================
# 4. FONCTIONS CORE — QUANTITATIVES
# ==========================================

@st.cache_data(ttl=86400)
def load_data(tickers: list) -> pd.DataFrame:
    """Télécharge 10 ans de prix ajustés."""
    start = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        data = raw
    return data.dropna(axis=1, how="all").ffill().bfill()


@st.cache_data(ttl=604800)
def get_company_info(tickers: list) -> dict:
    """Récupère les métadonnées fondamentales."""
    info_dict = {}
    for t in tickers:
        if t == "OBLIG_SIMUL":
            info_dict[t] = {"sector": "Monétaire / Taux", "summary": "Actif synthétique stabilisateur. Représente un OPCVM Monétaire PEA ou Fonds Euros.", "yield": f"{bond_yield*100:.2f}%"}
            continue
        try:
            info = yf.Ticker(t).info
            summary = info.get("longBusinessSummary", "Description non disponible.")
            if len(summary) > 350:
                summary = summary[:350] + "…"
            yld = info.get("dividendYield", None)
            info_dict[t] = {
                "sector": info.get("sector", info.get("category", "Fonds / Alternatif")),
                "summary": summary,
                "yield": f"{yld*100:.2f}%" if yld else "N/A"
            }
        except Exception:
            info_dict[t] = {"sector": "N/A", "summary": "Données indisponibles.", "yield": "N/A"}
    return info_dict


def compute_ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Estimateur de covariance robuste via Ledoit-Wolf shrinkage."""
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_ * 252


def black_litterman(
    mean_mkt: np.ndarray,
    cov: np.ndarray,
    market_caps: np.ndarray,
    view_asset_idx: int,
    view_alpha: float,
    confidence: float,
    tau: float = 0.05
) -> np.ndarray:
    """
    Black-Litterman Model.
    Retourne des rendements ajustés qui combinent l'équilibre du marché
    avec une vue directionnelle sur un actif.
    """
    n = len(mean_mkt)
    # Equilibre implicite (CAPM inverse)
    w_mkt = market_caps / market_caps.sum()
    risk_aversion = (np.dot(mean_mkt, w_mkt) - 0.02) / np.dot(w_mkt, np.dot(cov, w_mkt))
    pi_eq = risk_aversion * np.dot(cov, w_mkt)  # rendements implicites d'équilibre

    # Matrice de vue P (1 actif surperformant)
    P = np.zeros((1, n))
    P[0, view_asset_idx] = 1.0
    Q = np.array([view_alpha])

    # Incertitude sur la vue (Omega diagonal)
    Omega = np.diag([(1.0 - confidence) / confidence * tau * float(P[0] @ cov @ P[0].T)])

    # Formule BL
    tau_cov = tau * cov
    M1 = np.linalg.inv(tau_cov)
    M2 = P.T @ np.linalg.inv(Omega) @ P
    mu_bl = np.linalg.inv(M1 + M2) @ (M1 @ pi_eq + P.T @ np.linalg.inv(Omega) @ Q)
    return mu_bl


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """
    Risk Parity (Equal Risk Contribution).
    Chaque actif contribue à parts égales au risque total du portefeuille.
    Méthode Bridgewater / Risk Parity popularisée par Ray Dalio.
    """
    n = cov.shape[0]
    target_contrib = np.ones(n) / n

    def risk_contrib_distance(weights):
        w = np.array(weights)
        port_var = w @ cov @ w
        port_std = np.sqrt(port_var)
        mrc = (cov @ w) / port_std  # Marginal Risk Contribution
        rc = w * mrc               # Risk Contribution
        rc_pct = rc / port_std     # % du risque total
        return np.sum((rc_pct - target_contrib) ** 2)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    bounds = [(0.01, 0.60)] * n
    init = np.ones(n) / n

    res = sco.minimize(
        risk_contrib_distance,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}
    )
    return res.x / res.x.sum()


def markowitz_min_vol(mean_returns: pd.Series, cov: np.ndarray, target: float, max_w: float) -> np.ndarray:
    """Markowitz classique: minimise la volatilité sous contrainte de rendement cible."""
    n = len(mean_returns)
    bounds = [(0.0, max_w)] * n
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        {"type": "eq", "fun": lambda x: float(mean_returns.values @ x) - np.clip(target, mean_returns.min(), mean_returns.max())}
    ]
    res = sco.minimize(
        lambda w: float(np.sqrt(w @ cov @ w)),
        np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500}
    )
    return res.x


def compute_risk_metrics(port_returns: pd.Series, weights: dict, benchmark_returns: pd.Series = None) -> dict:
    """
    Calcule l'ensemble des métriques de risque institutionnelles:
    Max Drawdown, CVaR 95%, Beta, Tracking Error, Sharpe, Sortino.
    """
    r = port_returns.dropna()

    # — Sharpe Ratio (rf = 3.5%)
    rf_daily = (1 + 0.035) ** (1/252) - 1
    excess = r - rf_daily
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0.0

    # — Sortino Ratio
    downside = excess[excess < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(252) if downside.std() > 0 else 0.0

    # — Max Drawdown
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    # — CVaR 95% (Expected Shortfall)
    var_95 = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()

    # — Annualized Return & Vol
    ann_ret = (1 + r.mean()) ** 252 - 1
    ann_vol = r.std() * np.sqrt(252)

    metrics = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "cvar_95": cvar_95,
        "var_95": var_95,
    }

    # — Beta & Tracking Error vs Benchmark
    if benchmark_returns is not None:
        aligned = pd.concat([r, benchmark_returns], axis=1).dropna()
        aligned.columns = ["port", "bench"]
        if len(aligned) > 30:
            cov_pb = np.cov(aligned["port"], aligned["bench"])
            metrics["beta"] = cov_pb[0, 1] / cov_pb[1, 1]
            te = (aligned["port"] - aligned["bench"]).std() * np.sqrt(252)
            metrics["tracking_error"] = te
        else:
            metrics["beta"] = None
            metrics["tracking_error"] = None

    return metrics


def backtest_with_rebalancing(
    data: pd.DataFrame,
    target_weights: dict,
    montant: float,
    rebalancing_freq: str,
    frais_courtage: float
) -> pd.Series:
    """
    Backtest avec double comptabilité risky/bond.
    risky_value suit les holdings en prix absolus.
    bond_value capitalise quotidiennement de façon independante.
    """
    tickers_real = [t for t in target_weights if t != "OBLIG_SIMUL" and t in data.columns]
    if not tickers_real:
        return pd.Series(dtype=float)

    prices = data[tickers_real].copy()

    oblig_w = target_weights.get("OBLIG_SIMUL", 0.0)
    risky_w = 1.0 - oblig_w

    # Poids relatifs au sein de la poche risquee (normalises a 1.0)
    raw_w = np.array([target_weights.get(t, 0.0) for t in tickers_real])
    if raw_w.sum() > 0:
        raw_w = raw_w / raw_w.sum()

    risky_capital = montant * risky_w
    bond_value = montant * oblig_w

    p0 = prices.iloc[0].values
    holdings = raw_w * risky_capital / np.where(p0 > 0, p0, 1.0)

    freq_map = {"Annuel": "YE", "Semestriel": "6ME", "Aucun": None}
    freq = freq_map[rebalancing_freq]

    rebalancing_dates = set()
    if freq:
        idx_rebal = pd.date_range(start=prices.index[0], end=prices.index[-1], freq=freq)
        rebalancing_dates = set(idx_rebal.normalize())

    bond_daily_rate = (1.0 + bond_yield) ** (1.0 / 252) - 1.0

    portfolio_values = [montant]

    for i in range(1, len(prices)):
        date = prices.index[i].normalize()
        current_prices = prices.iloc[i].values

        risky_value = float(np.dot(holdings, current_prices))
        bond_value = bond_value * (1.0 + bond_daily_rate)
        total_value = risky_value + bond_value

        if date in rebalancing_dates:
            cost = total_value * risky_w * frais_courtage
            total_value -= cost
            risky_value = total_value * risky_w
            bond_value = total_value * oblig_w
            holdings = raw_w * risky_value / np.where(current_prices > 0, current_prices, 1.0)

        portfolio_values.append(total_value)

    return pd.Series(portfolio_values, index=prices.index[:len(portfolio_values)])


def monte_carlo_projection(
    ann_ret: float,
    ann_vol: float,
    montant: float,
    horizon: int,
    n_sim: int = 2000,
    taux_imposition: float = 0.172
):
    """Monte Carlo (Mouvement Brownien Géométrique) avec fiscalité appliquée à la fin."""
    drift = ann_ret - 0.5 * ann_vol ** 2
    paths = np.zeros((n_sim, horizon + 1))
    paths[:, 0] = montant

    for t in range(1, horizon + 1):
        z = np.random.standard_normal(n_sim)
        paths[:, t] = paths[:, t-1] * np.exp(drift + ann_vol * z)

    # Fiscalité: impôt sur les plus-values uniquement
    def apply_tax(final_values):
        pv = np.maximum(final_values - montant, 0)
        return final_values - pv * taux_imposition

    return {
        "years": np.arange(horizon + 1),
        "p10": np.percentile(paths, 10, axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p90": np.percentile(paths, 90, axis=0),
        "p10_net": apply_tax(np.percentile(paths, 10, axis=0)),
        "p50_net": apply_tax(np.percentile(paths, 50, axis=0)),
        "p90_net": apply_tax(np.percentile(paths, 90, axis=0)),
    }


# ==========================================
# 5. EXÉCUTION PRINCIPALE
# ==========================================

if run_btn:

    # 5.1 Construction de l'univers
    ticker_pool = []
    if choix_univers == "Mix Global":
        for v in ASSET_UNIVERSE.values():
            ticker_pool.extend(v)
    elif choix_univers == "Uniquement ETF Globaux":
        ticker_pool = ASSET_UNIVERSE["ETF Globaux"]
    elif choix_univers == "Uniquement Niches":
        ticker_pool = ASSET_UNIVERSE["Niches / Alternatifs"]
    elif choix_univers == "Uniquement CAC 40":
        ticker_pool = ASSET_UNIVERSE["CAC 40"]
    elif choix_univers == "Uniquement DAX 40":
        ticker_pool = ASSET_UNIVERSE["DAX 40"]

    if pea_only:
        ticker_pool = [t for t in ticker_pool if t not in NON_PEA_ASSETS]

    ticker_pool = list(dict.fromkeys(ticker_pool))

    if not ticker_pool:
        st.error("❌ Aucun actif disponible avec ces filtres. Élargissez votre sélection.")
        st.stop()

    with st.spinner("⏳ Téléchargement des données historiques (10 ans)…"):
        data = load_data(ticker_pool)

    available = [t for t in ticker_pool if t in data.columns]
    if len(available) < 3:
        st.error("❌ Moins de 3 actifs disponibles. Modifiez votre sélection.")
        st.stop()

    returns = data[available].pct_change().dropna()

    # 5.2 Ledoit-Wolf Covariance
    cov_lw = compute_ledoit_wolf_cov(returns)
    cov_df = pd.DataFrame(cov_lw, index=available, columns=available)
    mean_ret = returns.mean() * 252

    # Déduction des TER sur les rendements attendus
    for t in available:
        ter = ASSET_TER.get(t, 0.0)
        mean_ret[t] -= ter

    ind_vols = returns.std() * np.sqrt(252)

    # Poche sécurisée
    bond_ticker = "OBLIG_SIMUL"
    if inclure_oblig:
        mean_ret[bond_ticker] = bond_yield - ASSET_TER.get(bond_ticker, 0.001)
        ind_vols[bond_ticker] = 0.01
        n_existing = len(cov_df)
        new_row = pd.DataFrame(0.0, index=[bond_ticker], columns=cov_df.columns)
        new_col = pd.DataFrame(0.0, index=cov_df.index.tolist() + [bond_ticker], columns=[bond_ticker])
        new_col.loc[bond_ticker, bond_ticker] = 0.0001
        cov_df = pd.concat([cov_df, new_row])
        cov_df = pd.concat([cov_df, new_col], axis=1)
        all_tickers = available + [bond_ticker]
    else:
        all_tickers = available

    # Sélection top N par Sharpe individuel
    individual_sharpe = (mean_ret[all_tickers] - 0.035) / ind_vols[all_tickers]
    top_tickers = individual_sharpe.nlargest(max_actifs).index.tolist()
    if inclure_oblig and bond_ticker not in top_tickers:
        top_tickers = top_tickers[:max_actifs-1] + [bond_ticker]

    mean_sel = mean_ret[top_tickers]
    cov_sel = cov_df.loc[top_tickers, top_tickers].values
    n = len(top_tickers)

    # 5.3 Optimisation selon la méthode choisie
    with st.spinner(f"🧮 Optimisation via {methode_opti}…"):

        if "Risk Parity" in methode_opti:
            raw_w = risk_parity_weights(cov_sel)

        elif "Black-Litterman" in methode_opti:
            market_caps = np.ones(n)  # poids équipondérés comme proxy market cap si indisponibles
            view_idx = top_tickers.index(bl_view_asset) if bl_view_asset in top_tickers else 0
            mu_bl = black_litterman(
                mean_sel.values,
                cov_sel,
                market_caps,
                view_idx,
                bl_view_alpha,
                bl_confidence
            )
            mean_bl = pd.Series(mu_bl, index=top_tickers)
            raw_w = markowitz_min_vol(mean_bl, cov_sel, target_return, max_position)

        else:  # Markowitz classique
            raw_w = markowitz_min_vol(mean_sel, cov_sel, target_return, max_position)

    # Nettoyage & normalisation
    raw_w = np.clip(raw_w, 0, None)
    raw_w /= raw_w.sum()

    final_weights = {t: float(w) for t, w in zip(top_tickers, raw_w) if w > 0.005}
    # Renormalisation finale
    total_w = sum(final_weights.values())
    final_weights = {t: w/total_w for t, w in final_weights.items()}

    # Métriques portefeuille
    w_vec = np.array([final_weights.get(t, 0) for t in top_tickers])
    port_ret_annual = float(mean_sel.values @ w_vec)
    port_vol_annual = float(np.sqrt(w_vec @ cov_sel @ w_vec))

    # Backtest
    bt_series = backtest_with_rebalancing(data, final_weights, montant, rebalancing_freq, frais_courtage)

    # Benchmark
    bench_data = None
    bench_returns = None
    if BENCHMARK_TICKER in data.columns:
        bench_data = data[BENCHMARK_TICKER]
        bench_returns = bench_data.pct_change().dropna()

    # Métriques de risque avancées
    if len(bt_series) > 10:
        port_daily_ret = bt_series.pct_change().dropna()
        risk_m = compute_risk_metrics(port_daily_ret, final_weights, bench_returns)
    else:
        risk_m = {"ann_return": port_ret_annual, "ann_vol": port_vol_annual,
                  "sharpe": 0, "sortino": 0, "max_drawdown": -0.30,
                  "cvar_95": -0.025, "var_95": -0.020, "beta": 1.0, "tracking_error": 0.05}

    # Monte Carlo — vol théorique (Ledoit-Wolf) au lieu de la vol empirique du backtest
    mc = monte_carlo_projection(
        port_ret_annual,
        port_vol_annual,
        montant,
        horizon,
        n_sim=2000,
        taux_imposition=taux_imposition
    )

    # ==========================================
    # 6. DASHBOARD — AFFICHAGE
    # ==========================================

    # En-tête métriques clés
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rendement Net Espéré</div>
            <div class="metric-value">{risk_m['ann_return']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volatilité Annuelle</div>
            <div class="metric-value">{risk_m['ann_vol']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_m3:
        sharpe_color = "#34d399" if risk_m['sharpe'] > 1 else "#fb923c" if risk_m['sharpe'] > 0.5 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value" style="color:{sharpe_color}">{risk_m['sharpe']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value" style="color:#f87171">{risk_m['max_drawdown']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col_m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CVaR 95%</div>
            <div class="metric-value" style="color:#fb923c">{risk_m['cvar_95']*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Allocation & Composition",
        "⚠️ Risk Management",
        "🕰️ Backtest Historique",
        "🚀 Projections Monte-Carlo"
    ])

    # ── TAB 1 : ALLOCATION ──────────────────────────────────────────────
    with tab1:
        badge = "🟢 Portefeuille 100% éligible PEA" if pea_only else "🟠 Actifs hors PEA inclus"
        method_badge = f"Méthode : **{methode_opti}** · Covariance : **Ledoit-Wolf Shrinkage**"
        st.info(f"{badge} · {method_badge}")

        c_pie, c_table = st.columns([1, 1.3])

        with c_pie:
            df_pie = pd.DataFrame({
                "Nom": [TICKER_NAMES.get(t, t) for t in final_weights],
                "Poids": list(final_weights.values()),
                "TER": [f"{ASSET_TER.get(t,0)*100:.2f}%" for t in final_weights]
            })
            colors = px.colors.qualitative.Vivid
            fig_pie = px.pie(df_pie, values="Poids", names="Nom", hole=0.45,
                             color_discrete_sequence=colors)
            fig_pie.update_traces(textposition="outside", textfont_size=11)
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8d6e5",
                showlegend=True,
                legend=dict(font=dict(size=10)),
                margin=dict(t=20, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_table:
            df_table = pd.DataFrame({
                "Actif": [TICKER_NAMES.get(t, t) for t in final_weights],
                "Ticker": list(final_weights.keys()),
                "Poids %": [f"{v*100:.1f}%" for v in final_weights.values()],
                "Montant €": [f"{v*montant:,.0f} €" for v in final_weights.values()],
                "Volatilité": [f"{ind_vols.get(t, 0)*100:.1f}%" for t in final_weights],
                "TER/an": [f"{ASSET_TER.get(t,0)*100:.2f}%" for t in final_weights],
                "PEA": ["✅" if t not in NON_PEA_ASSETS else "❌" for t in final_weights]
            })
            st.dataframe(df_table, use_container_width=True, hide_index=True)

        st.markdown("---")
        c_corr, c_rc = st.columns(2)

        with c_corr:
            st.subheader("🔗 Corrélations (Ledoit-Wolf)")
            real_tickers_corr = [t for t in final_weights if t != bond_ticker and t in returns.columns]
            if len(real_tickers_corr) > 1:
                lw_corr = LedoitWolf()
                lw_corr.fit(returns[real_tickers_corr].values)
                lw_cov_arr = lw_corr.covariance_
                std_arr = np.sqrt(np.diag(lw_cov_arr))
                corr_arr = lw_cov_arr / np.outer(std_arr, std_arr)
                corr_df = pd.DataFrame(corr_arr,
                                       index=[TICKER_NAMES.get(t, t) for t in real_tickers_corr],
                                       columns=[TICKER_NAMES.get(t, t) for t in real_tickers_corr])
                fig_corr = px.imshow(corr_df, text_auto=".2f",
                                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
                fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e5",
                                        margin=dict(t=10, b=10))
                st.plotly_chart(fig_corr, use_container_width=True)

        with c_rc:
            st.subheader("⚖️ Contributions au Risque")
            real_t = [t for t in final_weights if t != bond_ticker and t in returns.columns]
            if len(real_t) > 1:
                w_rc = np.array([final_weights[t] for t in real_t])
                cov_rc = cov_df.loc[real_t, real_t].values
                port_std_rc = np.sqrt(w_rc @ cov_rc @ w_rc)
                mrc = (cov_rc @ w_rc) / port_std_rc
                rc = w_rc * mrc
                rc_pct = rc / rc.sum() * 100
                df_rc = pd.DataFrame({
                    "Actif": [TICKER_NAMES.get(t, t) for t in real_t],
                    "Contribution au Risque (%)": rc_pct
                }).sort_values("Contribution au Risque (%)", ascending=True)
                fig_rc = px.bar(df_rc, x="Contribution au Risque (%)", y="Actif",
                                orientation="h", color="Contribution au Risque (%)",
                                color_continuous_scale="Blues")
                fig_rc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e5",
                                      showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig_rc, use_container_width=True)

    # ── TAB 2 : RISK MANAGEMENT ────────────────────────────────────────
    with tab2:
        st.subheader("📐 Tableau de Bord des Risques Institutionnels")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Sharpe Ratio", f"{risk_m['sharpe']:.2f}", help="(Rp - Rf) / σp · Rf=3.5%")
        r2.metric("Sortino Ratio", f"{risk_m['sortino']:.2f}", help="Penalise uniquement la volatilité baissière")
        r3.metric("VaR 95% (1j)", f"{risk_m['var_95']*100:.2f}%", help="Dans 95% des jours, la perte ne dépasse pas ce seuil")
        r4.metric("CVaR 95% (1j)", f"{risk_m['cvar_95']*100:.2f}%", help="Perte moyenne dans les 5% pires jours (Expected Shortfall)")

        if risk_m.get("beta") is not None:
            r5, r6, r7, r8 = st.columns(4)
            r5.metric("Beta vs MSCI World", f"{risk_m['beta']:.2f}", help=">1 = plus risqué que le marché mondial")
            r6.metric("Tracking Error", f"{risk_m['tracking_error']*100:.1f}%/an", help="Ecart annualisé avec le benchmark MSCI World")
            r7.metric("Max Drawdown", f"{risk_m['max_drawdown']*100:.1f}%", help="Pire chute historique depuis un pic")
            r8.metric("Actifs", f"{len(final_weights)}", help="Nombre d'actifs dans le portefeuille final")

        st.markdown("---")

        # Distribution des rendements journaliers
        if len(bt_series) > 10:
            daily_r = bt_series.pct_change().dropna()
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=daily_r * 100,
                nbinsx=80,
                name="Rendements journaliers",
                marker_color="#38bdf8",
                opacity=0.7
            ))
            var_line = risk_m["var_95"] * 100
            cvar_line = risk_m["cvar_95"] * 100
            fig_dist.add_vline(x=var_line, line_color="#fb923c", line_dash="dash",
                               annotation_text=f"VaR 95%: {var_line:.2f}%",
                               annotation_position="top left")
            fig_dist.add_vline(x=cvar_line, line_color="#f87171", line_dash="dot",
                               annotation_text=f"CVaR 95%: {cvar_line:.2f}%",
                               annotation_position="top left")
            fig_dist.update_layout(
                title="Distribution des Rendements Journaliers",
                xaxis_title="Rendement (%)",
                yaxis_title="Fréquence",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8d6e5",
                bargap=0.1
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Drawdown chart
        if len(bt_series) > 10:
            cum_bt = (1 + bt_series.pct_change().fillna(0)).cumprod()
            roll_max_bt = cum_bt.cummax()
            dd_series = (cum_bt - roll_max_bt) / roll_max_bt * 100

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series.values,
                fill="tozeroy", fillcolor="rgba(248,113,113,0.2)",
                line=dict(color="#f87171", width=1.5),
                name="Drawdown (%)"
            ))
            fig_dd.update_layout(
                title="Historique des Drawdowns",
                xaxis_title="Date", yaxis_title="Drawdown (%)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8d6e5"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    # ── TAB 3 : BACKTEST ──────────────────────────────────────────────
    with tab3:
        freq_label = rebalancing_freq
        frais_label = f"{frais_courtage*100:.2f}%"
        st.info(f"📌 Backtest avec rééquilibrage **{freq_label}** · Frais de courtage : **{frais_label}** par rééquilibrage · TER déduits des rendements.")

        if len(bt_series) > 10:
            # Normalise pour comparer à 100
            bt_norm = bt_series / bt_series.iloc[0] * 100

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt_norm.index, y=bt_norm.values,
                name="Portefeuille Optimisé",
                line=dict(color="#38bdf8", width=2.5)
            ))

            # Benchmark MSCI World normalisé
            if bench_data is not None:
                bench_aligned = bench_data.reindex(bt_norm.index).ffill()
                bench_norm = bench_aligned / bench_aligned.iloc[0] * 100
                fig_bt.add_trace(go.Scatter(
                    x=bench_norm.index, y=bench_norm.values,
                    name="Benchmark MSCI World",
                    line=dict(color="#94a3b8", width=1.5, dash="dot")
                ))

            fig_bt.update_layout(
                title="Performance Historique Normalisée (base 100)",
                xaxis_title="Date",
                yaxis_title="Valeur (base 100)",
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8d6e5",
                legend=dict(orientation="h", y=1.02, x=0)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # Stats comparatives
            if bench_data is not None and len(bench_norm) > 10:
                port_total = (bt_norm.iloc[-1] - 100)
                bench_total = (bench_norm.iloc[-1] - 100)
                alpha = port_total - bench_total
                s1, s2, s3 = st.columns(3)
                s1.metric("Performance Portefeuille", f"+{port_total:.1f}%" if port_total > 0 else f"{port_total:.1f}%")
                s2.metric("Performance Benchmark", f"+{bench_total:.1f}%" if bench_total > 0 else f"{bench_total:.1f}%")
                s3.metric("Alpha généré", f"+{alpha:.1f}%" if alpha > 0 else f"{alpha:.1f}%",
                          delta=f"{alpha:.1f}%", delta_color="normal")
        else:
            st.warning("Données insuffisantes pour le backtest.")

    # ── TAB 4 : MONTE CARLO ──────────────────────────────────────────
    with tab4:
        st.info(f"📊 **2 000 simulations** · Mouvement Brownien Géométrique · Fiscalité **{enveloppe_fiscale}** ({taux_imposition*100:.1f}%) appliquée à la sortie")

        years = mc["years"]

        fig_mc = go.Figure()

        # Zone de confiance 80%
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([mc["p90"], mc["p10"][::-1]]),
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="Intervalle 80% (Brut)"
        ))

        # Zone 50% (p25-p75)
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([mc["p75"], mc["p25"][::-1]]),
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="Intervalle 50% (Brut)"
        ))

        # Médiane brute
        fig_mc.add_trace(go.Scatter(
            x=years, y=mc["p50"],
            name="⚖️ Médiane Brute",
            line=dict(color="#38bdf8", width=3)
        ))

        # Scénarios nets (après impôts)
        fig_mc.add_trace(go.Scatter(
            x=years, y=mc["p50_net"],
            name="⚖️ Médiane Nette (après impôts)",
            line=dict(color="#a78bfa", width=2.5, dash="dash")
        ))
        fig_mc.add_trace(go.Scatter(
            x=years, y=mc["p90_net"],
            name="🚀 Optimiste Net",
            line=dict(color="#34d399", width=1.5, dash="dot")
        ))
        fig_mc.add_trace(go.Scatter(
            x=years, y=mc["p10_net"],
            name="📉 Pessimiste Net",
            line=dict(color="#f87171", width=1.5, dash="dot")
        ))

        # Ligne capital initial
        fig_mc.add_hline(y=montant, line_color="#64748b", line_dash="dash",
                         annotation_text="Capital Initial")

        fig_mc.update_layout(
            title=f"Projection Monte-Carlo à {horizon} ans — {n_sim if 'n_sim' in dir() else 2000} scénarios",
            xaxis_title="Années",
            yaxis_title="Valeur Projetée (€)",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d6e5",
            legend=dict(orientation="h", y=-0.15, x=0, font=dict(size=11))
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # Tableau de synthèse
        st.markdown("### 🎯 Atterrissage estimé à l'horizon")
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Pessimiste Brut (P10)", f"{mc['p10'][-1]:,.0f} €".replace(",", " "))
        tc2.metric("Médiane Brute (P50)", f"{mc['p50'][-1]:,.0f} €".replace(",", " "))
        tc3.metric("Optimiste Brut (P90)", f"{mc['p90'][-1]:,.0f} €".replace(",", " "))
        tc4.metric(f"Médiane Nette ({enveloppe_fiscale.split(' ')[0]})", f"{mc['p50_net'][-1]:,.0f} €".replace(",", " "))

        st.markdown("---")
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1e3a5f; border-radius:8px; padding:1rem; font-size:0.8rem; color:#64748b; font-family:'IBM Plex Mono', monospace;">
        ⚠️ <strong style="color:#94a3b8;">Avertissements réglementaires</strong><br>
        Les performances passées ne préjugent pas des performances futures. Ce document est à titre purement éducatif et ne constitue pas un conseil en investissement.
        Les rendements affichés sont nets de TER mais bruts de fiscalité sauf mention contraire.
        La méthode Black-Litterman nécessite des vues prospectives et des paramètres de marché à jour pour être pertinente.
        Rendements espérés calculés sur données historiques — horizon d'estimation : 10 ans max.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #334155;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📐</div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.2rem; color: #64748b;">
            Configurez votre profil dans le panneau gauche<br>et lancez l'analyse.
        </div>
        <div style="margin-top: 1rem; font-size: 0.85rem; color: #334155;">
            Risk Parity · Black-Litterman · Ledoit-Wolf · CVaR · Max Drawdown · TER · Rebalancing · Fiscalité
        </div>
    </div>
    """, unsafe_allow_html=True)
