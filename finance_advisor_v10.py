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



@st.cache_data(ttl=86400)
def get_market_caps(tickers: list) -> dict:
    """
    Récupère les market caps réelles via yfinance pour construire
    le portefeuille de marché CAPM.  Fallback = médiane du groupe
    (pas np.ones) pour que les poids restent cohérents même si
    quelques tickers échouent.
    """
    caps = {}
    for t in tickers:
        if t == "OBLIG_SIMUL":
            caps[t] = None
            continue
        try:
            info = yf.Ticker(t).info
            # marketCap pour les actions, totalAssets pour les ETF/fonds
            cap = info.get("marketCap") or info.get("totalAssets")
            caps[t] = float(cap) if cap else None
        except Exception:
            caps[t] = None

    # Fallback : remplacer les None par la médiane des caps connues
    known = [v for v in caps.values() if v is not None and v > 0]
    fallback = float(np.median(known)) if known else 1.0
    return {t: (v if v else fallback) for t, v in caps.items()}


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




# ── SCÉNARIOS DE STRESS ──────────────────────────────────────────────────────
# Chaque scénario définit une fenêtre temporelle de crise historique.
# On calcule la performance du portefeuille ET du benchmark sur cette période,
# ainsi que le drawdown max intra-période.

STRESS_SCENARIOS = {
    "🔴 Crise 2008 (Lehman)": {
        "start": "2008-09-01",
        "end":   "2009-03-31",
        "desc":  "Faillite Lehman Brothers · Gel du crédit mondial · S&P -55% en 17 mois",
        "color": "#f87171"
    },
    "🟠 Krach Covid-19": {
        "start": "2020-02-19",
        "end":   "2020-03-23",
        "desc":  "Pandémie mondiale · Vente panique · Chute la plus rapide de l'histoire (-34% en 33 jours)",
        "color": "#fb923c"
    },
    "🟡 Choc Taux 2022": {
        "start": "2022-01-03",
        "end":   "2022-10-12",
        "desc":  "Hausse brutale des taux Fed/BCE · Krach obligataire · Tech -35%, Bonds -20%",
        "color": "#fbbf24"
    },
    "🔵 Bulle Tech 2000": {
        "start": "2000-03-10",
        "end":   "2002-10-09",
        "desc":  "Éclatement bulle dot-com · Nasdaq -78% en 2.5 ans · Récession NBER 2001",
        "color": "#60a5fa"
    },
}


def run_stress_tests(
    data: pd.DataFrame,
    final_weights: dict,
    benchmark_ticker: str,
    scenarios: dict
) -> list:
    """
    Pour chaque scénario de stress :
    1. Extrait la fenêtre temporelle du prix de chaque actif du portefeuille
    2. Simule la performance (return-based, pas holdings — pas de rebalancing intra-crise)
    3. Calcule : perf totale, max drawdown intra-période, perf benchmark, ratio protection
    Retourne une liste de dicts prêts pour l'affichage.
    """
    results = []
    tickers_real = [t for t in final_weights if t != "OBLIG_SIMUL" and t in data.columns]
    oblig_w = final_weights.get("OBLIG_SIMUL", 0.0)
    risky_w = 1.0 - oblig_w

    # Poids relatifs normalisés dans la poche risquée
    raw_w = np.array([final_weights.get(t, 0.0) for t in tickers_real])
    if raw_w.sum() > 0:
        raw_w = raw_w / raw_w.sum()

    for name, cfg in scenarios.items():
        try:
            start = pd.Timestamp(cfg["start"])
            end   = pd.Timestamp(cfg["end"])

            # Slice the price data for the crisis window
            window = data[tickers_real].loc[start:end]
            if len(window) < 5:
                results.append({
                    "name": name, "desc": cfg["desc"], "color": cfg["color"],
                    "port_perf": None, "port_dd": None,
                    "bench_perf": None, "ratio": None,
                    "available": False,
                    "start": cfg["start"], "end": cfg["end"]
                })
                continue

            # Rendements journaliers pondérés (poche risquée)
            daily_r_risky = window.pct_change().fillna(0).dot(raw_w)

            # Rendement journalier de la poche sécurisée
            bond_daily = (1.0 + bond_yield) ** (1.0 / 252) - 1.0

            # Rendement total du portefeuille = pondération des deux poches
            daily_r_port = daily_r_risky * risky_w + bond_daily * oblig_w

            # Performance cumulée
            cum = (1 + daily_r_port).cumprod()
            port_perf = float(cum.iloc[-1] - 1.0)

            # Max drawdown intra-période
            roll_max = cum.cummax()
            dd = (cum - roll_max) / roll_max
            port_dd = float(dd.min())

            # Benchmark
            bench_perf = None
            if benchmark_ticker in data.columns:
                bench_window = data[benchmark_ticker].loc[start:end]
                if len(bench_window) > 5:
                    bench_cum = bench_window / bench_window.iloc[0]
                    bench_perf = float(bench_cum.iloc[-1] - 1.0)

            # Ratio de protection = 1 - (perte port / perte bench)
            # > 0 : portefeuille protège mieux que le bench
            ratio = None
            if bench_perf is not None and bench_perf < 0 and port_perf < 0:
                ratio = 1.0 - (port_perf / bench_perf)

            results.append({
                "name": name,
                "desc": cfg["desc"],
                "color": cfg["color"],
                "port_perf": port_perf,
                "port_dd": port_dd,
                "bench_perf": bench_perf,
                "ratio": ratio,
                "available": True,
                "start": cfg["start"],
                "end": cfg["end"],
                "cum_series": cum,
                "bench_series": (
                    (data[benchmark_ticker].loc[start:end] /
                     data[benchmark_ticker].loc[start:end].iloc[0])
                    if benchmark_ticker in data.columns
                    else None
                )
            })

        except Exception as e:
            results.append({
                "name": name, "desc": cfg["desc"], "color": cfg["color"],
                "port_perf": None, "port_dd": None,
                "bench_perf": None, "ratio": None,
                "available": False,
                "start": cfg["start"], "end": cfg["end"]
            })

    return results


def compute_efficient_frontier(
    mean_returns: pd.Series,
    cov: np.ndarray,
    n_points: int = 80,
    max_w: float = 0.60
) -> pd.DataFrame:
    """
    Trace la frontière efficiente de Markowitz en résolvant n_points
    portefeuilles à variance minimale pour une grille de rendements cibles.

    Retourne un DataFrame avec colonnes : ret, vol, sharpe, weights_dict
    + les portefeuilles spéciaux (min-vol, max-sharpe, tangent).
    """
    n = len(mean_returns)
    rf = 0.035  # taux sans risque OAT 10 ans

    # Bornes sûres : on couvre 95% de l'intervalle réalisable
    r_min = mean_returns.min() * 0.95
    r_max = mean_returns.max() * 0.95
    targets = np.linspace(r_min, r_max, n_points)

    rows = []
    for target in targets:
        safe_target = np.clip(target, mean_returns.min(), mean_returns.max())
        bounds = [(0.0, max_w)] * n
        constraints = [
            {"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0},
            {"type": "eq", "fun": lambda x, t=safe_target: float(mean_returns.values @ x) - t}
        ]
        res = sco.minimize(
            lambda w: float(np.sqrt(w @ cov @ w)),
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 400}
        )
        if res.success:
            w = np.clip(res.x, 0, None)
            w /= w.sum()
            vol = float(np.sqrt(w @ cov @ w))
            ret = float(mean_returns.values @ w)
            sharpe = (ret - rf) / vol if vol > 0 else 0.0
            rows.append({
                "ret": ret, "vol": vol, "sharpe": sharpe,
                "weights": dict(zip(mean_returns.index, w))
            })

    df = pd.DataFrame(rows).sort_values("vol").reset_index(drop=True)

    # ── Portefeuilles spéciaux ──────────────────────────────────────────
    specials = {}

    # Min-vol : déjà le premier point trié par vol, mais on le recalcule proprement
    res_mv = sco.minimize(
        lambda w: float(np.sqrt(w @ cov @ w)),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, max_w)] * n,
        constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}],
        options={"ftol": 1e-10, "maxiter": 400}
    )
    if res_mv.success:
        w = np.clip(res_mv.x, 0, None); w /= w.sum()
        specials["min_vol"] = {
            "ret": float(mean_returns.values @ w),
            "vol": float(np.sqrt(w @ cov @ w)),
            "sharpe": (float(mean_returns.values @ w) - rf) / float(np.sqrt(w @ cov @ w)),
            "weights": dict(zip(mean_returns.index, w)),
            "label": "📌 Min-Volatilité"
        }

    # Max-Sharpe (portefeuille tangent)
    res_ms = sco.minimize(
        lambda w: -((float(mean_returns.values @ w) - rf) / float(np.sqrt(w @ cov @ w))),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, max_w)] * n,
        constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}],
        options={"ftol": 1e-10, "maxiter": 400}
    )
    if res_ms.success:
        w = np.clip(res_ms.x, 0, None); w /= w.sum()
        specials["max_sharpe"] = {
            "ret": float(mean_returns.values @ w),
            "vol": float(np.sqrt(w @ cov @ w)),
            "sharpe": (float(mean_returns.values @ w) - rf) / float(np.sqrt(w @ cov @ w)),
            "weights": dict(zip(mean_returns.index, w)),
            "label": "⭐ Max-Sharpe (tangent)"
        }

    return df, specials


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
            # ── Market caps réelles (CAPM equilibrium) ──────────────────
            with st.spinner("📡 Récupération des market caps…"):
                raw_caps = get_market_caps(top_tickers)
            market_caps = np.array([raw_caps.get(t, 1.0) for t in top_tickers])

            # Diagnostique : affiche la source de chaque cap
            cap_labels = []
            for t in top_tickers:
                v = raw_caps.get(t)
                if t == "OBLIG_SIMUL":
                    cap_labels.append(f"{TICKER_NAMES.get(t,t)}: synthétique")
                elif v == float(np.median([x for x in raw_caps.values() if x])):
                    cap_labels.append(f"{TICKER_NAMES.get(t,t)}: fallback médiane")
                else:
                    cap_labels.append(f"{TICKER_NAMES.get(t,t)}: {v/1e9:.1f} Md€")
            with st.expander("🏦 Market caps utilisées pour l'équilibre BL", expanded=False):
                st.caption("  ·  ".join(cap_labels))

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Allocation & Composition",
        "⚠️ Risk Management",
        "🕰️ Backtest Historique",
        "🚀 Projections Monte-Carlo",
        "📐 Frontière Efficiente"
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
                "Poids %": [round(v*100, 2) for v in final_weights.values()],
                "Montant (EUR)": [round(v*montant, 2) for v in final_weights.values()],
                "Vol %": [round(ind_vols.get(t, 0)*100, 2) for t in final_weights],
                "TER %/an": [round(ASSET_TER.get(t,0)*100, 3) for t in final_weights],
                "PEA": ["✅" if t not in NON_PEA_ASSETS else "❌" for t in final_weights]
            })
            st.dataframe(
                df_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Poids %": st.column_config.NumberColumn("Poids %", format="%.1f %%"),
                    "Montant (EUR)": st.column_config.NumberColumn("Montant (€)", format="%.0f €"),
                    "Vol %": st.column_config.NumberColumn("Volatilité %", format="%.1f %%"),
                    "TER %/an": st.column_config.NumberColumn("TER %/an", format="%.2f %%"),
                }
            )

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


        # ── SECTION PÉDAGOGIQUE ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📚 Comprendre les modèles & termes techniques")
        mode_expert = True
        edu_cols = st.columns(3)

        with edu_cols[0]:
            with st.expander("⚖️ Risk Parity (Parité des Risques)"):
                st.markdown("""
**En résumé :** Chaque actif contribue *à parts égales* au risque total du portefeuille, quels que soient ses rendements attendus.
C'est la méthode du fonds Bridgewater All Weather de Ray Dalio.

**Pourquoi ?** Dans Markowitz classique, les actions monopolisent ~90 % du risque même à 60 % du capital.
Risk Parity surpondère automatiquement les actifs défensifs (obligations, monétaire) pour équilibrer.

**Avantage :** Robuste, ne dépend d'aucune estimation de rendement futur.
**Inconvénient :** Sous-performe en bull market pur actions ; sensible aux hausses de taux.
                """)
                if mode_expert:
                    st.latex(r"MRC_i = \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}}, \quad RC_i = w_i \cdot MRC_i")
                    st.latex(r"\min_w \sum_{i=1}^{n} \left( \frac{RC_i}{\sum_j RC_j} - \frac{1}{n} \right)^2 \quad s.c. \;\sum_i w_i = 1,\; w_i \geq 0")
                    st.caption("Minimise l'écart entre contribution effective et cible 1/n. Résolu par SLSQP (gradient projeté).")

            with st.expander("🏦 Black-Litterman"):
                st.markdown("""
**En résumé :** Combine l'équilibre implicite du marché (consensus) avec vos propres convictions sur certains actifs.
Développé à Goldman Sachs (1990) — standard de l'industrie institutionnelle.

**Pourquoi ?** Le Markowitz pur amplifie les erreurs : une légère sur-estimation du rendement d'un actif lui alloue 100 % du capital.
Black-Litterman ancre sur l'équilibre de marché et n'ajuste que là où vous avez une vue explicite.

**Avantage :** Portefeuilles stables, diversifiés, moins sensibles aux inputs.
**Inconvénient :** Nécessite des vues prospectives quantifiées et calibrées.
                """)
                if mode_expert:
                    st.latex(r"\Pi = \lambda \Sigma w_{mkt}, \quad \lambda = \frac{E[R_m]-r_f}{\sigma_m^2}")
                    st.latex(r"\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1} \left[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q\right]")
                    st.caption("P : matrice de vues (1 ligne par vue). Q : rendements attendus des vues. Omega : incertitude sur les vues. tau=0.05 contrôle le poids relatif vues/équilibre.")

            with st.expander("🔬 Ledoit-Wolf Shrinkage"):
                st.markdown("""
**En résumé :** Version stabilisée de la matrice de covariance historique, résistante au sur-ajustement sur historiques courts.

**Pourquoi ?** Avec 20 actifs, la matrice brute a 210 paramètres à estimer — très sensible au bruit.
Le shrinkage la contracte vers une cible structurée, réduisant l'erreur d'estimation out-of-sample.

**Avantage :** Toujours définie positive (solveur converge garanti).
**Inconvénient :** Lisse les corrélations extrêmes, peut sous-estimer la contagion en crise.
                """)
                if mode_expert:
                    st.latex(r"\hat{\Sigma} = (1-\alpha^*) \cdot S + \alpha^* \cdot T")
                    st.latex(r"\alpha^* = \arg\min_\alpha \;\mathbb{E}\left[\|\hat{\Sigma}(\alpha) - \Sigma_{true}\|_F^2\right]")
                    st.caption("S = matrice empirique. T = cible (corrélations constantes). alpha* estimé analytiquement sans cross-validation (Oracle Approximating Shrinkage, Chen et al.).")

        with edu_cols[1]:
            with st.expander("📉 Max Drawdown"):
                st.markdown("""
**En résumé :** La pire perte subie depuis un pic historique.
Si le portefeuille atteignait 100 000 € puis tombait à 65 000 €, le Max DD est **-35 %**.

**Pourquoi c'est important ?** La volatilité annualisée est abstraite.
Le Max Drawdown est ce que votre client *ressent* réellement — c'est la mesure de douleur.

**Règle empirique :** Max DD ≈ 2× la volatilité annuelle pour des actifs actions standards.
Le **Calmar Ratio** = rendement annuel / |Max DD| mesure l'efficacité par unité de souffrance.
                """)
                if mode_expert:
                    st.latex(r"\text{MaxDD} = \min_{t \in [0,T]} \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}")
                    st.caption("V_t = valeur du portefeuille à t. Le running maximum M_t = max(V_s, s<=t) représente le dernier pic atteint. Drawdown courant DD_t = (V_t - M_t)/M_t.")

            with st.expander("🎯 VaR & CVaR (Expected Shortfall)"):
                st.markdown("""
**VaR 95% :** "Dans 95 % des jours de trading, la perte ne dépassera pas X %." C'est un quantile.

**CVaR 95% :** "Dans les 5 % des pires jours, vous perdrez en moyenne Y %."
C'est la *moyenne de la queue gauche* — bien plus informative que la VaR seule.

**Pourquoi le CVaR ?** La VaR ne dit rien sur l'*ampleur* des pertes extrêmes.
Le CVaR est **obligatoire** en reporting institutionnel (Bâle III, UCITS, Solvency II).
                """)
                if mode_expert:
                    st.latex(r"\text{VaR}_\alpha = F_R^{-1}(\alpha)")
                    st.latex(r"\text{CVaR}_\alpha = \mathbb{E}[R \mid R \leq \text{VaR}_\alpha] = \frac{1}{\alpha}\int_0^\alpha F_R^{-1}(u)\,du")
                    st.caption("Le CVaR est une mesure de risque cohérente (Artzner 1999) : sous-additive et convexe. La VaR ne l'est pas — deux portefeuilles combinés peuvent avoir une VaR supérieure à leur somme individuelle.")

            with st.expander("📊 Sharpe & Sortino Ratio"):
                st.markdown("""
**Sharpe Ratio :** Rendement excédentaire (au-dessus du taux sans risque) par unité de risque *total*.
Sharpe > 1 = bon, > 2 = excellent, < 0.5 = peu efficace.

**Sortino Ratio :** Comme le Sharpe, mais ne pénalise que la volatilité *baissière*.
Un portefeuille qui monte fort et baisse peu aura un Sortino bien supérieur à son Sharpe.

**Attention :** Ces ratios sont calculés sur l'historique disponible — ils ne prédisent pas l'avenir.
                """)
                if mode_expert:
                    st.latex(r"\text{Sharpe} = \frac{\bar{R}_p - R_f}{\sigma_p} \cdot \sqrt{252}")
                    st.latex(r"\text{Sortino} = \frac{\bar{R}_p - R_f}{\sigma_{\downarrow}} \cdot \sqrt{252}, \quad \sigma_{\downarrow} = \sqrt{\frac{1}{T}\sum_{R_t < R_f}(R_t - R_f)^2}")
                    st.caption("Rf = taux sans risque (3.5% ici, OAT 10 ans approximatif). sigma_down = semi-déviation baissière. Information Ratio = alpha / TE : un IR > 0.5 suggère une compétence de gestion réelle.")

        with edu_cols[2]:
            with st.expander("🔁 Rééquilibrage & Rebalancing Drag"):
                st.markdown("""
**Pourquoi rééquilibrer ?** Sans rééquilibrage, un actif qui surperforme finit par dominer le portefeuille.
Ex : une poche actions à 60 % peut passer à 80 % après un bull market — risque non désiré.

**Rééquilibrage annuel :** Vendre ce qui a monté, acheter ce qui a baissé.
C'est une discipline contrariante naturelle qui force un "buy low, sell high" systématique.

**Rebalancing Drag :** Chaque rééquilibrage génère des frais de transaction (courtage paramétrable)
et potentiellement des événements fiscaux (réalisés en CTO).
                """)
                if mode_expert:
                    st.latex(r"V_{t_k}^{net} = V_{t_k} \cdot (1 - w_{risky} \cdot c)")
                    st.latex(r"h_i(t_k) = \frac{w_i \cdot V_{t_k}^{net} \cdot w_{risky}}{P_i(t_k)}")
                    st.caption("c = frais de courtage. w_risky = fraction risquée (on ne rééquilibre pas la poche monétaire). La bande de tolérance optimale (Dumas & Luciano, 1991) n'est pas implémentée — on utilise une approche calendaire.")

            with st.expander("🏛️ Monte Carlo — Mouvement Brownien Géométrique"):
                st.markdown("""
**En résumé :** 2 000 "futurs alternatifs" sont simulés en ajoutant chaque année un choc aléatoire
calibré sur la volatilité théorique du portefeuille (Ledoit-Wolf).

**P10 / P50 / P90 :** Dans 80 % des scénarios, votre capital finit entre P10 et P90.
P50 est la médiane — scénario central le plus probable.

**Limites :** Le MBG suppose des rendements log-normaux et une volatilité constante.
Il ne capture pas les crises de liquidité, les corrélations de stress ni les "fat tails".
                """)
                if mode_expert:
                    st.latex(r"S_{t+1} = S_t \cdot \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right) + \sigma Z_t\right], \quad Z_t \sim \mathcal{N}(0,1)")
                    st.latex(r"S_T^{net} = S_T - \max(S_T - S_0,\, 0) \cdot \tau")
                    st.caption("Le terme -sigma²/2 est le volatility drag (inégalité de Jensen) : la moyenne arithmétique doit être ajustée en moyenne géométrique pour refléter la croissance réelle du capital. tau = taux d'imposition sur plus-values (17.2% PEA, 30% CTO).")

            with st.expander("📐 Beta & Tracking Error"):
                st.markdown("""
**Beta :** Sensibilité au marché global. Beta = 1.2 → si le MSCI World monte de 10 %,
votre portefeuille monte de 12 % en moyenne (et perd 12 % si le marché recule de 10 %).

**Tracking Error :** Ecart-type annualisé de la *différence* de performance vs benchmark.
TE = 5 % → votre performance dévie d'environ ±5 %/an du MSCI World.

**Benchmark ici :** ETF MSCI World (CW8.PA) — le marché actions global capitalisation-pondéré.
                """)
                if mode_expert:
                    st.latex(r"\beta_p = \frac{\text{Cov}(R_p, R_b)}{\text{Var}(R_b)}, \quad \text{TE} = \sigma(R_p - R_b) \cdot \sqrt{252}")
                    st.latex(r"\alpha = \bar{R}_p - \beta_p \cdot \bar{R}_b \quad \text{(Jensen's Alpha)}")
                    st.caption("Le CAPM prédit alpha=0 en marché efficient. Un alpha > 0 persistant suggère soit une compétence, soit un risque non capturé (facteurs Fama-French : taille, value, momentum, qualité).")

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


        # ── ROLLING METRICS ─────────────────────────────────────────────────
        if len(bt_series) > 60:
            st.markdown("---")
            st.subheader("📈 Métriques Glissantes")

            roll_col1, roll_col2 = st.columns([1, 3])
            with roll_col1:
                roll_window = st.select_slider(
                    "Fenêtre glissante",
                    options=[21, 42, 63, 126, 252],
                    value=126,
                    format_func=lambda x: {
                        21: "1 mois", 42: "2 mois", 63: "3 mois",
                        126: "6 mois", 252: "1 an"
                    }[x],
                    key="roll_window_slider"
                )
                rf_daily = (1 + 0.035) ** (1 / 252) - 1
                show_bench_rolling = st.checkbox(
                    "Afficher benchmark", value=True, key="roll_bench_cb"
                )

            port_r = bt_series.pct_change().dropna()

            # ── Rolling Sharpe ───────────────────────────────────────────────
            excess = port_r - rf_daily
            roll_mean   = excess.rolling(roll_window).mean()
            roll_std    = excess.rolling(roll_window).std()
            roll_sharpe = (roll_mean / roll_std) * (252 ** 0.5)
            roll_sharpe = roll_sharpe.dropna()

            bench_roll_sharpe = None
            if show_bench_rolling and bench_data is not None:
                bench_r = bench_data.pct_change().dropna()
                bench_excess = bench_r - rf_daily
                brs_mean = bench_excess.rolling(roll_window).mean()
                brs_std  = bench_excess.rolling(roll_window).std()
                bench_roll_sharpe = (brs_mean / brs_std) * (252 ** 0.5)
                bench_roll_sharpe = bench_roll_sharpe.dropna()

            fig_rs = go.Figure()

            # Zone verte (Sharpe > 1) et rouge (< 0)
            fig_rs.add_hrect(y0=1, y1=max(roll_sharpe.max() + 0.5, 3.5),
                             fillcolor="rgba(52,211,153,0.04)", line_width=0)
            fig_rs.add_hrect(y0=min(roll_sharpe.min() - 0.5, -2), y1=0,
                             fillcolor="rgba(248,113,113,0.04)", line_width=0)
            fig_rs.add_hline(y=1,  line_color="#34d399", line_dash="dot",
                             line_width=0.8, annotation_text="Sharpe = 1",
                             annotation_font_size=10, annotation_font_color="#34d399")
            fig_rs.add_hline(y=0,  line_color="#64748b", line_width=0.5)

            # Courbe principale — colorée positif/négatif via deux traces
            pos_mask = roll_sharpe >= 0
            neg_mask = roll_sharpe < 0

            if pos_mask.any():
                rs_pos = roll_sharpe.where(pos_mask)
                fig_rs.add_trace(go.Scatter(
                    x=rs_pos.index, y=rs_pos.values,
                    name="Sharpe ≥ 0",
                    line=dict(color="#38bdf8", width=2),
                    connectgaps=False
                ))
            if neg_mask.any():
                rs_neg = roll_sharpe.where(neg_mask)
                fig_rs.add_trace(go.Scatter(
                    x=rs_neg.index, y=rs_neg.values,
                    name="Sharpe < 0",
                    line=dict(color="#f87171", width=2),
                    connectgaps=False
                ))

            if bench_roll_sharpe is not None:
                fig_rs.add_trace(go.Scatter(
                    x=bench_roll_sharpe.index, y=bench_roll_sharpe.values,
                    name="Benchmark",
                    line=dict(color="#94a3b8", width=1.5, dash="dot"),
                    opacity=0.7
                ))

            fig_rs.update_layout(
                title=f"Sharpe Ratio glissant ({roll_window}j · Rf=3.5%)",
                xaxis_title=None, yaxis_title="Sharpe",
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8d6e5",
                legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
                yaxis=dict(gridcolor="rgba(100,116,139,0.12)", zeroline=False),
                xaxis=dict(gridcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_rs, use_container_width=True)

            # ── Rolling Volatility ───────────────────────────────────────────
            roll_vol = port_r.rolling(roll_window).std() * (252 ** 0.5) * 100
            roll_vol = roll_vol.dropna()

            bench_roll_vol = None
            if show_bench_rolling and bench_data is not None:
                bench_r2 = bench_data.pct_change().dropna()
                bench_roll_vol = bench_r2.rolling(roll_window).std() * (252 ** 0.5) * 100
                bench_roll_vol = bench_roll_vol.dropna()

            # Percentiles pour la bande de confiance
            vol_p25 = float(roll_vol.quantile(0.25))
            vol_p75 = float(roll_vol.quantile(0.75))

            fig_rv = go.Figure()

            # Bande inter-quartile (zone normale de vol)
            fig_rv.add_hrect(
                y0=vol_p25, y1=vol_p75,
                fillcolor="rgba(56,189,248,0.05)",
                line_width=0,
                annotation_text="Zone normale (Q25–Q75)",
                annotation_position="top right",
                annotation_font_size=9,
                annotation_font_color="#64748b"
            )

            # Fill sous la courbe
            fig_rv.add_trace(go.Scatter(
                x=roll_vol.index, y=roll_vol.values,
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.06)",
                line=dict(color="#38bdf8", width=2),
                name="Volatilité portefeuille"
            ))

            if bench_roll_vol is not None:
                fig_rv.add_trace(go.Scatter(
                    x=bench_roll_vol.index, y=bench_roll_vol.values,
                    line=dict(color="#94a3b8", width=1.5, dash="dot"),
                    name="Volatilité benchmark",
                    opacity=0.7
                ))

            # Ligne de vol moyenne
            avg_vol = float(roll_vol.mean())
            fig_rv.add_hline(
                y=avg_vol, line_color="#64748b", line_dash="dash", line_width=0.8,
                annotation_text=f"Moy. {avg_vol:.1f}%",
                annotation_font_size=10, annotation_font_color="#64748b"
            )

            fig_rv.update_layout(
                title=f"Volatilité Annualisée Glissante ({roll_window}j)",
                xaxis_title=None, yaxis_title="Vol (%/an)",
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8d6e5",
                legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
                yaxis=dict(
                    gridcolor="rgba(100,116,139,0.12)",
                    zeroline=False,
                    ticksuffix="%"
                ),
                xaxis=dict(gridcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_rv, use_container_width=True)

            # ── Rolling Beta ──────────────────────────────────────────────────
            if bench_data is not None:
                bench_r3 = bench_data.pct_change().dropna()
                aligned = pd.concat([port_r, bench_r3], axis=1).dropna()
                aligned.columns = ["port", "bench"]

                roll_beta = (
                    aligned["port"].rolling(roll_window)
                    .cov(aligned["bench"])
                    /
                    aligned["bench"].rolling(roll_window).var()
                ).dropna()

                fig_rb = go.Figure()
                fig_rb.add_hline(y=1.0, line_color="#64748b", line_dash="dot",
                                 line_width=0.8,
                                 annotation_text="Beta = 1 (= marché)",
                                 annotation_font_size=10,
                                 annotation_font_color="#64748b")
                fig_rb.add_hrect(y0=0.8, y1=1.2,
                                 fillcolor="rgba(100,116,139,0.06)", line_width=0)

                beta_pos = roll_beta.where(roll_beta >= 1)
                beta_neg = roll_beta.where(roll_beta < 1)

                if not beta_pos.dropna().empty:
                    fig_rb.add_trace(go.Scatter(
                        x=beta_pos.index, y=beta_pos.values,
                        name="Beta ≥ 1 (agressif)",
                        line=dict(color="#fb923c", width=2),
                        connectgaps=False
                    ))
                if not beta_neg.dropna().empty:
                    fig_rb.add_trace(go.Scatter(
                        x=beta_neg.index, y=beta_neg.values,
                        name="Beta < 1 (défensif)",
                        line=dict(color="#34d399", width=2),
                        connectgaps=False
                    ))

                fig_rb.update_layout(
                    title=f"Beta Glissant vs MSCI World ({roll_window}j)",
                    xaxis_title=None, yaxis_title="Beta",
                    hovermode="x unified",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c8d6e5",
                    legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
                    yaxis=dict(gridcolor="rgba(100,116,139,0.12)", zeroline=False),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)")
                )
                st.plotly_chart(fig_rb, use_container_width=True)

            # ── Mini stats glissantes ─────────────────────────────────────────
            st.markdown("##### Stats sur la fenêtre glissante")
            last_n = min(roll_window, len(roll_sharpe))
            ms1, ms2, ms3, ms4 = st.columns(4)
            ms1.metric(
                "Sharpe actuel",
                f"{roll_sharpe.iloc[-1]:.2f}",
                delta=f"{roll_sharpe.iloc[-1] - roll_sharpe.mean():.2f} vs moy.",
                delta_color="normal"
            )
            ms2.metric(
                "Vol actuelle",
                f"{roll_vol.iloc[-1]:.1f}%",
                delta=f"{roll_vol.iloc[-1] - roll_vol.mean():.1f}% vs moy.",
                delta_color="inverse"
            )
            ms3.metric("Sharpe max (historique)", f"{roll_sharpe.max():.2f}")
            ms4.metric("Vol max (historique)", f"{roll_vol.max():.1f}%")


    # ── TAB 3 : BACKTEST ──────────────────────────────────────────────
    with tab3:
        freq_label = rebalancing_freq
        frais_label = f"{frais_courtage*100:.2f}%"
        st.info(f"📌 Backtest avec rééquilibrage **{freq_label}** · Frais de courtage : **{frais_label}** par rééquilibrage · TER déduits des rendements.")

        if len(bt_series) > 10:
            # Normalise pour comparer à 100
            bt_norm = bt_series / bt_series.iloc[0] * 100

            fig_bt = go.Figure()

            # Zones de crise annotées sur le backtest principal
            crisis_shading = [
                ("2008-09-01", "2009-03-31", "rgba(248,113,113,0.08)", "2008"),
                ("2020-02-19", "2020-03-23", "rgba(251,146,60,0.12)",  "Covid"),
                ("2022-01-03", "2022-10-12", "rgba(251,191,36,0.08)",  "2022"),
            ]
            for cs_start, cs_end, cs_color, cs_label in crisis_shading:
                cs_s = pd.Timestamp(cs_start)
                cs_e = pd.Timestamp(cs_end)
                if cs_s >= bt_norm.index[0] and cs_s <= bt_norm.index[-1]:
                    fig_bt.add_vrect(
                        x0=cs_s, x1=min(cs_e, bt_norm.index[-1]),
                        fillcolor=cs_color, line_width=0,
                        annotation_text=cs_label,
                        annotation_position="top left",
                        annotation_font_size=10,
                        annotation_font_color="#94a3b8"
                    )

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

        # ── STRESS TESTS ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔥 Analyse de Stress — Crises Historiques")
        st.caption(
            "Simulation de la performance de votre portefeuille actuel sur chaque crise passée. "
            "Aucun rebalancing intra-crise. Poche sécurisée capitalisée au taux paramétré."
        )

        with st.spinner("⏳ Calcul des stress tests…"):
            stress_results = run_stress_tests(
                data, final_weights, BENCHMARK_TICKER, STRESS_SCENARIOS
            )

        # ── Cartes de résumé (1 par scénario) ───────────────────────────────
        stress_cols = st.columns(len(stress_results))
        for col, sr in zip(stress_cols, stress_results):
            with col:
                if not sr["available"]:
                    st.markdown(f"""
                    <div style="border:0.5px solid #1e3a5f;border-radius:8px;padding:12px;opacity:0.5">
                        <div style="font-size:12px;font-weight:500;color:#94a3b8">{sr['name']}</div>
                        <div style="font-size:11px;color:#64748b;margin-top:4px">Données non disponibles<br>({sr['start']} → {sr['end']})</div>
                    </div>""", unsafe_allow_html=True)
                    continue

                port_p = sr["port_perf"]
                bench_p = sr["bench_perf"]
                port_dd = sr["port_dd"]
                ratio = sr["ratio"]

                # Couleur de la perf portefeuille
                perf_color = "#34d399" if port_p >= 0 else "#f87171" if port_p < -0.15 else "#fb923c"
                ratio_txt = f"Protection : +{ratio*100:.0f}%" if ratio and ratio > 0 else (f"Sous-perf : {ratio*100:.0f}%" if ratio else "N/A")
                ratio_color = "#34d399" if ratio and ratio > 0 else "#f87171"

                st.markdown(f"""
                <div style="border:0.5px solid {sr['color']}40;border-left:3px solid {sr['color']};border-radius:8px;padding:14px 12px;background:rgba(17,24,39,0.6)">
                    <div style="font-size:11px;font-weight:500;color:{sr['color']};margin-bottom:6px">{sr['name']}</div>
                    <div style="font-size:22px;font-weight:600;color:{perf_color};font-family:monospace">{port_p*100:+.1f}%</div>
                    <div style="font-size:10px;color:#64748b;margin-top:2px">Portefeuille</div>
                    <div style="border-top:0.5px solid #1e3a5f;margin:8px 0"></div>
                    <div style="font-size:11px;color:#64748b">Benchmark : <span style="color:#94a3b8">{f"{bench_p*100:+.1f}%" if bench_p is not None else "N/A"}</span></div>
                    <div style="font-size:11px;color:#64748b">Max DD : <span style="color:#f87171">{port_dd*100:.1f}%</span></div>
                    <div style="font-size:11px;color:{ratio_color};margin-top:4px">{ratio_txt}</div>
                </div>""", unsafe_allow_html=True)

        # ── Graphiques détaillés par scénario ────────────────────────────────
        st.markdown("#### Trajectoires détaillées par crise")
        available_scenarios = [sr for sr in stress_results if sr["available"]]

        if available_scenarios:
            n_cols = min(2, len(available_scenarios))
            rows = [available_scenarios[i:i+n_cols] for i in range(0, len(available_scenarios), n_cols)]

            for row in rows:
                plot_cols = st.columns(n_cols)
                for col, sr in zip(plot_cols, row):
                    with col:
                        fig_s = go.Figure()

                        # Zone de fond colorée selon la sévérité
                        fig_s.add_hrect(
                            y0=0, y1=1,
                            fillcolor=f"{sr['color']}08",
                            line_width=0
                        )

                        # Courbe portefeuille
                        cum_vals = sr["cum_series"].values * 100 - 100
                        fig_s.add_trace(go.Scatter(
                            x=sr["cum_series"].index,
                            y=cum_vals,
                            name="Portefeuille",
                            line=dict(color="#38bdf8", width=2),
                            fill="tozeroy",
                            fillcolor="rgba(56,189,248,0.06)"
                        ))

                        # Courbe benchmark
                        if sr["bench_series"] is not None:
                            bench_vals = sr["bench_series"].values * 100 - 100
                            fig_s.add_trace(go.Scatter(
                                x=sr["bench_series"].index,
                                y=bench_vals,
                                name="MSCI World",
                                line=dict(color="#94a3b8", width=1.5, dash="dot")
                            ))

                        # Ligne zéro
                        fig_s.add_hline(y=0, line_color="#334155", line_width=0.5)

                        fig_s.update_layout(
                            title=dict(
                                text=sr["name"],
                                font=dict(size=12, color=sr["color"])
                            ),
                            xaxis_title=None,
                            yaxis_title="Variation (%)",
                            hovermode="x unified",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#c8d6e5",
                            legend=dict(
                                orientation="h", y=1.1, x=0,
                                font=dict(size=9)
                            ),
                            margin=dict(t=50, b=30, l=40, r=10),
                            height=280
                        )
                        fig_s.update_yaxes(ticksuffix="%", gridcolor="rgba(100,116,139,0.1)")
                        fig_s.update_xaxes(gridcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_s, use_container_width=True)
                        st.caption(sr["desc"])
        else:
            st.info("💡 Les crises sélectionnées sont antérieures à la fenêtre de données disponibles (10 ans). La crise 2008 nécessite des données pré-2009.")

        # ── Tableau récapitulatif ────────────────────────────────────────────
        st.markdown("#### Tableau récapitulatif")
        stress_table_rows = []
        for sr in stress_results:
            if sr["available"]:
                stress_table_rows.append({
                    "Scénario": sr["name"],
                    "Période": f"{sr['start']} → {sr['end']}",
                    "Perf Portefeuille %": round(sr["port_perf"] * 100, 2),
                    "Perf Benchmark %": round(sr["bench_perf"] * 100, 2) if sr["bench_perf"] is not None else None,
                    "Max DD %": round(sr["port_dd"] * 100, 2),
                    "Ratio Protection %": round(sr["ratio"] * 100, 1) if sr["ratio"] is not None else None,
                })
        if stress_table_rows:
            df_stress = pd.DataFrame(stress_table_rows)
            st.dataframe(
                df_stress,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Perf Portefeuille %": st.column_config.NumberColumn(format="%.2f %%"),
                    "Perf Benchmark %": st.column_config.NumberColumn(format="%.2f %%"),
                    "Max DD %": st.column_config.NumberColumn(format="%.2f %%"),
                    "Ratio Protection %": st.column_config.NumberColumn(
                        format="%.1f %%",
                        help=">0% = portefeuille perd moins que le benchmark en termes relatifs"
                    ),
                }
            )

        st.markdown("""
        <div style="font-size:11px;color:#475569;font-family:monospace;margin-top:1rem;padding:10px;border:0.5px solid #1e3a5f;border-radius:6px">
        ⚠️ Stress tests calculés sur les actifs disponibles dans yfinance pour la période concernée.
        La crise 2008 peut être indisponible si le portefeuille contient des actifs cotés après 2010.
        Simulation sans levier, sans dérivés, sans short selling.
        </div>
        """, unsafe_allow_html=True)

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

    # ── TAB 5 : FRONTIÈRE EFFICIENTE ────────────────────────────────────
    with tab5:
        st.info(
            "📐 Chaque point représente un portefeuille optimal (variance minimale) "
            "pour un niveau de rendement cible. La couleur encode le **Sharpe Ratio**. "
            "Les actifs individuels sont affichés pour référence."
        )

        with st.spinner("🧮 Calcul de la frontière efficiente (80 portefeuilles)…"):
            # Exclure la poche sécurisée du calcul frontier (vol artificielle = 0.01)
            fe_tickers = [t for t in top_tickers if t != "OBLIG_SIMUL"]
            if len(fe_tickers) >= 2:
                mean_fe = mean_sel[fe_tickers]
                cov_fe  = cov_df.loc[fe_tickers, fe_tickers].values
                df_frontier, specials = compute_efficient_frontier(
                    mean_fe, cov_fe, n_points=80, max_w=max_position
                )
            else:
                df_frontier = pd.DataFrame()
                specials = {}

        if df_frontier.empty:
            st.warning("Pas assez d'actifs pour tracer la frontière.")
        else:
            fig_ef = go.Figure()

            # ── Zone sous la frontière (fill visuel) ──────────────────
            fig_ef.add_trace(go.Scatter(
                x=df_frontier["vol"] * 100,
                y=df_frontier["ret"] * 100,
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.04)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip"
            ))

            # ── Courbe frontière colorée par Sharpe ───────────────────
            fig_ef.add_trace(go.Scatter(
                x=df_frontier["vol"] * 100,
                y=df_frontier["ret"] * 100,
                mode="markers+lines",
                marker=dict(
                    size=6,
                    color=df_frontier["sharpe"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title="Sharpe",
                        thickness=12,
                        len=0.6,
                        tickformat=".2f"
                    ),
                    line=dict(width=0)
                ),
                line=dict(color="rgba(56,189,248,0.3)", width=1.5),
                name="Frontière efficiente",
                hovertemplate=(
                    "<b>Frontière</b><br>"
                    "Rendement : %{y:.2f}%<br>"
                    "Volatilité : %{x:.2f}%<br>"
                    "Sharpe : %{marker.color:.2f}<extra></extra>"
                )
            ))

            # ── Actifs individuels ────────────────────────────────────
            ind_vols_fe  = {t: float(ind_vols[t]) for t in fe_tickers if t in ind_vols}
            ind_rets_fe  = {t: float(mean_sel[t]) for t in fe_tickers if t in mean_sel}
            ind_sharpe_fe = {
                t: (ind_rets_fe[t] - 0.035) / ind_vols_fe[t]
                for t in fe_tickers
                if t in ind_vols_fe and ind_vols_fe[t] > 0
            }

            fig_ef.add_trace(go.Scatter(
                x=[ind_vols_fe[t] * 100 for t in fe_tickers if t in ind_vols_fe],
                y=[ind_rets_fe[t]  * 100 for t in fe_tickers if t in ind_rets_fe],
                mode="markers+text",
                marker=dict(
                    size=9,
                    color=[ind_sharpe_fe.get(t, 0) for t in fe_tickers if t in ind_vols_fe],
                    colorscale="Viridis",
                    symbol="diamond",
                    line=dict(color="rgba(255,255,255,0.6)", width=1),
                    showscale=False
                ),
                text=[TICKER_NAMES.get(t, t) for t in fe_tickers if t in ind_vols_fe],
                textposition="top center",
                textfont=dict(size=9, color="#94a3b8"),
                name="Actifs individuels",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Rendement : %{y:.2f}%<br>"
                    "Volatilité : %{x:.2f}%<extra></extra>"
                )
            ))

            # ── Portefeuille optimisé (notre résultat) ────────────────
            fig_ef.add_trace(go.Scatter(
                x=[port_vol_annual * 100],
                y=[port_ret_annual * 100],
                mode="markers+text",
                marker=dict(size=16, color="#38bdf8", symbol="star",
                            line=dict(color="#ffffff", width=1.5)),
                text=["Votre portefeuille"],
                textposition="bottom right",
                textfont=dict(size=11, color="#38bdf8"),
                name="Votre portefeuille",
                hovertemplate=(
                    "<b>Votre portefeuille</b><br>"
                    "Rendement : %{y:.2f}%<br>"
                    "Volatilité : %{x:.2f}%<extra></extra>"
                )
            ))

            # ── Portefeuilles spéciaux ────────────────────────────────
            special_colors = {"min_vol": "#a78bfa", "max_sharpe": "#34d399"}
            special_symbols = {"min_vol": "circle", "max_sharpe": "star-triangle-up"}
            for key, sp in specials.items():
                fig_ef.add_trace(go.Scatter(
                    x=[sp["vol"] * 100],
                    y=[sp["ret"] * 100],
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        color=special_colors.get(key, "#fb923c"),
                        symbol=special_symbols.get(key, "circle"),
                        line=dict(color="#ffffff", width=1.5)
                    ),
                    text=[sp["label"]],
                    textposition="top right",
                    textfont=dict(size=10, color=special_colors.get(key, "#fb923c")),
                    name=sp["label"],
                    hovertemplate=(
                        f"<b>{sp['label']}</b><br>"
                        "Rendement : %{y:.2f}%<br>"
                        "Volatilité : %{x:.2f}%<br>"
                        f"Sharpe : {sp['sharpe']:.2f}<extra></extra>"
                    )
                ))

            # ── Ligne du Capital Market Line (CML) ────────────────────
            if "max_sharpe" in specials:
                sp_ms = specials["max_sharpe"]
                cml_vols = np.linspace(0, sp_ms["vol"] * 1.5, 50)
                sharpe_t = sp_ms["sharpe"]
                cml_rets = 0.035 + sharpe_t * cml_vols
                fig_ef.add_trace(go.Scatter(
                    x=cml_vols * 100,
                    y=cml_rets * 100,
                    mode="lines",
                    line=dict(color="rgba(52,211,153,0.35)", width=1.5, dash="dot"),
                    name="Capital Market Line",
                    hoverinfo="skip"
                ))

            # ── Layout ────────────────────────────────────────────────
            fig_ef.update_layout(
                xaxis_title="Volatilité annuelle (%)",
                yaxis_title="Rendement espéré net TER (%)",
                hovermode="closest",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8d6e5",
                legend=dict(
                    orientation="h", y=-0.15, x=0,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"
                ),
                xaxis=dict(gridcolor="rgba(100,116,139,0.15)", zeroline=False),
                yaxis=dict(gridcolor="rgba(100,116,139,0.15)", zeroline=False),
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_ef, use_container_width=True)

            # ── Tableau comparatif des 3 portefeuilles clés ───────────
            st.markdown("### 📊 Comparaison des portefeuilles clés")
            rows_compare = []
            # Notre portefeuille
            rows_compare.append({
                "Portefeuille": "⭐ Votre portefeuille",
                "Rendement %": round(port_ret_annual * 100, 2),
                "Volatilité %": round(port_vol_annual * 100, 2),
                "Sharpe": round((port_ret_annual - 0.035) / port_vol_annual, 2) if port_vol_annual > 0 else 0.0,
            })
            for key, sp in specials.items():
                rows_compare.append({
                    "Portefeuille": sp["label"],
                    "Rendement %": round(sp["ret"] * 100, 2),
                    "Volatilité %": round(sp["vol"] * 100, 2),
                    "Sharpe": round(sp["sharpe"], 2),
                })
            df_compare = pd.DataFrame(rows_compare)
            st.dataframe(
                df_compare, use_container_width=True, hide_index=True,
                column_config={
                    "Rendement %": st.column_config.NumberColumn(format="%.2f %%"),
                    "Volatilité %": st.column_config.NumberColumn(format="%.2f %%"),
                    "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                }
            )

            # ── Composition du max-Sharpe ─────────────────────────────
            if "max_sharpe" in specials:
                with st.expander("🔍 Composition du portefeuille Max-Sharpe"):
                    ms_w = specials["max_sharpe"]["weights"]
                    df_ms = pd.DataFrame({
                        "Actif": [TICKER_NAMES.get(t, t) for t in ms_w if ms_w[t] > 0.005],
                        "Poids %": [round(v * 100, 1) for t, v in ms_w.items() if v > 0.005],
                        "Montant (€)": [round(v * montant, 0) for t, v in ms_w.items() if v > 0.005],
                    })
                    st.dataframe(
                        df_ms, use_container_width=True, hide_index=True,
                        column_config={
                            "Poids %": st.column_config.NumberColumn(format="%.1f %%"),
                            "Montant (€)": st.column_config.NumberColumn(format="%.0f €"),
                        }
                    )


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
