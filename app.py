import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION ET UNIVERS
# ==========================================
st.set_page_config(page_title="Finance Advisor 5.0", page_icon="📈", layout="wide")
st.title("🤖 Finance Advisor V5.0 - Projections & Backtest")

ASSET_UNIVERSE = {
    "Actions Françaises (CAC40)": ["AI.PA", "AIR.PA", "OR.PA", "MC.PA", "TTE.PA", "SAN.PA", "BNP.PA", "SU.PA", "CAP.PA", "RMS.PA", "KER.PA", "SAF.PA"],
    "ETF (Fonds Indiciels)": ["CW8.PA", "ESE.PA", "RS2K.PA", "PAEEM.PA", "PUST.PA"],
    "Obligations (Bonds)": ["MTA.PA", "PRHY.PA", "OBLI.PA", "C3M.PA"]
}

TICKER_NAMES = {
    "AI.PA": "Air Liquide", "AIR.PA": "Airbus", "OR.PA": "L'Oréal", "MC.PA": "LVMH",
    "TTE.PA": "TotalEnergies", "SAN.PA": "Sanofi", "BNP.PA": "BNP Paribas",
    "SU.PA": "Schneider Electric", "CAP.PA": "Capgemini", "RMS.PA": "Hermès",
    "KER.PA": "Kering", "SAF.PA": "Safran",
    "CW8.PA": "Amundi MSCI World", "ESE.PA": "Amundi S&P 500", 
    "RS2K.PA": "Amundi Russell 2000", "PAEEM.PA": "Amundi MSCI Emerging", 
    "PUST.PA": "Amundi Nasdaq-100",
    "MTA.PA": "Lyxor Euro Gov Bond 7-10Y", "PRHY.PA": "Amundi Euro High Yield",
    "OBLI.PA": "Global Aggregate Bonds", "C3M.PA": "Amundi Trésorerie (Monétaire)"
}

RISK_PROFILES = {"🛡️ Prudent": 0.05, "⚖️ Équilibré": 0.09, "🚀 Dynamique": 0.14}

# ==========================================
# 2. INTERFACE SIDEBAR
# ==========================================
st.sidebar.header("🛠️ Paramètres")
montant = st.sidebar.number_input("Capital (€)", min_value=500, value=10000, step=500)
profil_nom = st.sidebar.selectbox("Objectif de Rendement", list(RISK_PROFILES.keys()))
target_return = RISK_PROFILES[profil_nom]

st.sidebar.markdown("---")
choix_univers = st.sidebar.radio("Univers", ("Mix", "Uniquement ETF", "Uniquement Actions", "Uniquement Obligations"))
max_actifs = st.sidebar.slider("Nombre d'actifs max", 2, 15, 6)
horizon = st.sidebar.slider("Horizon de projection (ans)", 1, 30, 10)

# ==========================================
# 3. LOGIQUE MATHÉMATIQUE
# ==========================================
@st.cache_data(ttl=86400)
def load_data(tickers):
    data = yf.download(tickers, start=(datetime.now() - timedelta(days=365*10)), end=datetime.now(), auto_adjust=True)['Close']
    return data.dropna(axis=1, how='all').ffill().bfill()

def get_metrics(weights, means, cov):
    r = np.sum(means * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return r, std

def opt_func(weights, means, cov, target):
    r, std = get_metrics(weights, means, cov)
    return std + 100 * abs(r - target)

# ==========================================
# 4. EXECUTION
# ==========================================
all_tickers = []
for cat in ASSET_UNIVERSE.values(): all_tickers.extend(cat)

if st.sidebar.button("⚡ Lancer l'Analyse", type="primary"):
    tickers_sub = []
    if "ETF" in choix_univers: tickers_sub = ASSET_UNIVERSE["ETF (Fonds Indiciels)"]
    elif "Actions" in choix_univers: tickers_sub = ASSET_UNIVERSE["Actions Françaises (CAC40)"]
    elif "Obligations" in choix_univers: tickers_sub = ASSET_UNIVERSE["Obligations (Bonds)"]
    else: tickers_sub = all_tickers

    data = load_data(tickers_sub)
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Optimisation
    num = len(mean_returns)
    res = sco.minimize(opt_func, num*[1./num], args=(mean_returns, cov_matrix, target_return), 
                       method='SLSQP', bounds=tuple((0,1) for _ in range(num)), constraints={'type':'eq','fun':lambda x:np.sum(x)-1})
    
    weights_dict = {mean_returns.index[i]: res.x[i] for i in range(num) if res.x[i] > 0.01}
    top_weights = dict(sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)[:max_actifs])
    norm_sum = sum(top_weights.values())
    final_weights = {k: v/norm_sum for k, v in top_weights.items()}

    # --- AFFICHAGE ---
    tab1, tab2, tab3 = st.tabs(["📊 Allocation", "historical Backtest", "🚀 Projections Futures"])

    with tab1:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            df_pie = pd.DataFrame({"Nom": [TICKER_NAMES.get(t, t) for t in final_weights.keys()], "Poids": list(final_weights.values())})
            st.plotly_chart(px.pie(df_pie, values='Poids', names='Nom', hole=0.4).update_traces(textposition='outside'))
        with c2:
            df_table = pd.DataFrame({
                "Ticker": list(final_weights.keys()),
                "Nom": [TICKER_NAMES.get(t, t) for t in final_weights.keys()],
                "Poids": [v*100 for v in final_weights.values()],
                "Valeur": [v*montant for v in final_weights.values()]
            })
            st.dataframe(df_table, column_config={"Poids": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)}, hide_index=True)

    with tab2:
        hist_ret = data[list(final_weights.keys())].pct_change().fillna(0).dot(list(final_weights.values()))
        st.line_chart((1 + hist_ret).cumprod() * montant)

    with tab3:
        st.subheader(f"Projection sur {horizon} ans")
        
        # Séparation des rendements
        bond_tickers = ASSET_UNIVERSE["Obligations (Bonds)"]
        w_bonds = sum([v for k,v in final_weights.items() if k in bond_tickers])
        w_equity = 1 - w_bonds
        
        # Calcul des mu et sigma pondérés pour la partie actions
        equity_keys = [k for k in final_weights.keys() if k not in bond_tickers]
        if equity_keys:
            mu_e = sum([mean_returns[k] * final_weights[k] for k in equity_keys]) / w_equity if w_equity > 0 else 0
            sig_e = real_std = np.sqrt(np.dot(np.array([final_weights[k] for k in equity_keys]).T, 
                                             np.dot(cov_matrix.loc[equity_keys, equity_keys], 
                                                    np.array([final_weights[k] for k in equity_keys])))) / w_equity if w_equity > 0 else 0
        else: mu_e, sig_e = 0, 0
            
        # Rendement obligataire (Yield moyen)
        bond_keys = [k for k in final_weights.keys() if k in bond_tickers]
        mu_b = sum([mean_returns[k] * final_weights[k] for k in bond_keys]) / w_bonds if w_bonds > 0 else 0

        years = np.arange(0, horizon + 1)
        
        def calculate_path(type_scen):
            path = [montant]
            for y in range(1, horizon + 1):
                # Partie Actions avec scénarios
                if type_scen == "Pessimiste":
                    r_e = (mu_e - 2*sig_e) if y > 1 else -0.20 # Choc an 1
                elif type_scen == "Normal": r_e = mu_e
                elif type_scen == "Bon": r_e = mu_e + 0.5*sig_e
                elif type_scen == "Excellent": r_e = mu_e + 1.5*sig_e
                
                growth = (w_equity * (1 + r_e)) + (w_bonds * (1 + mu_b))
                path.append(path[-1] * growth)
            return path

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=years, y=calculate_path("Excellent"), name="🚀 Excellent", line=dict(color='green', dash='dash')))
        fig_proj.add_trace(go.Scatter(x=years, y=calculate_path("Normal"), name="⚖️ Normal", fill='tonexty', line=dict(color='blue', width=4)))
        fig_proj.add_trace(go.Scatter(x=years, y=calculate_path("Pessimiste"), name="📉 Pessimiste (Crise)", line=dict(color='red')))
        
        fig_proj.update_layout(title="Simulation de l'évolution du capital", xaxis_title="Années", yaxis_title="Valeur (€)")
        st.plotly_chart(fig_proj, use_container_width=True)
        
        if w_bonds > 0:
            st.info(f"💡 Votre portefeuille contient {w_bonds*100:.1f}% d'obligations qui agissent comme un stabilisateur (rendement constant estimé à {mu_b*100:.2f}%).")
