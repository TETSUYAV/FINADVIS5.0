import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION DE L'APPLICATION
# ==========================================
st.set_page_config(page_title="Finance Advisor 5.0", page_icon="📈", layout="wide")
st.title("🤖 Finance Advisor V5.0 - Allocation Dynamique")
st.markdown("Optimisation de portefeuille avec marge de tolérance et diversification inter-classes.")

# ==========================================
# 2. UNIVERS D'INVESTISSEMENT (Diversifié)
# ==========================================
ASSET_UNIVERSE = {
    "Actions Françaises (CAC40)": [
        "AI.PA", "AIR.PA", "OR.PA", "MC.PA", "TTE.PA", "SAN.PA", 
        "BNP.PA", "SU.PA", "CAP.PA", "RMS.PA", "KER.PA", "SAF.PA"
    ],
    "ETF (Fonds Indiciels)": [
        "CW8.PA",   # Amundi MSCI World (Monde)
        "ESE.PA",   # Amundi S&P 500 (US)
        "RS2K.PA",  # Amundi Russell 2000 (US Small Caps)
        "PAEEM.PA", # Amundi MSCI Emerging Markets
        "PUST.PA"   # Amundi Nasdaq-100 (Tech)
    ],
    "Obligations (Bonds)": [
        "MTA.PA",   # Euro Government Bond 7-10Y
        "PRHY.PA",  # Euro High Yield Corporate
        "OBLI.PA",  # Global Aggregate Bonds
        "C3M.PA"    # Trésorerie / Monétaire (Sécurité)
    ]
}

RISK_PROFILES = {
    "🛡️ Prudent": 0.05,
    "⚖️ Équilibré": 0.09,
    "🚀 Dynamique": 0.14
}

TOLERANCE = 0.006  # Marge de tolérance de 0.6%

# ==========================================
# 3. INTERFACE UTILISATEUR (Sidebar)
# ==========================================
st.sidebar.header("🛠️ Paramètres du Portefeuille")

montant = st.sidebar.number_input("Capital à investir (€)", min_value=500, value=10000, step=500)

profil_nom = st.sidebar.selectbox("Profil de Risque (Rendement Cible)", list(RISK_PROFILES.keys()))
target_return = RISK_PROFILES[profil_nom]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Stratégie & Allocation")

choix_univers = st.sidebar.radio(
    "Classes d'actifs à inclure",
    ("Mix (Actions, ETF, Obligations)", "Uniquement ETF", "Uniquement Actions", "Uniquement Obligations")
)

max_actifs = st.sidebar.slider("Nombre maximum d'actifs retenus", min_value=2, max_value=15, value=6)

# ==========================================
# 4. FONCTIONS DE LOGIQUE (Avec Cache)
# ==========================================
def get_selected_tickers(choix):
    tickers = []
    if choix == "Uniquement Actions":
        tickers.extend(ASSET_UNIVERSE["Actions Françaises (CAC40)"])
    elif choix == "Uniquement ETF":
        tickers.extend(ASSET_UNIVERSE["ETF (Fonds Indiciels)"])
    elif choix == "Uniquement Obligations":
        tickers.extend(ASSET_UNIVERSE["Obligations (Bonds)"])
    else: # Mix total
        for cat in ASSET_UNIVERSE.values():
            tickers.extend(cat)
    return tickers

@st.cache_data(ttl=86400) # Cache de 24h pour limiter les requêtes API
def load_market_data(tickers):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    return data.dropna(axis=1, how='all').ffill().bfill()

def get_portfolio_metrics(weights, means, cov):
    r = np.sum(means * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return r, std

def minimize_volatility_with_tolerance(weights, means, cov, target, tolerance):
    r, std = get_portfolio_metrics(weights, means, cov)
    if (target - tolerance) <= r <= (target + tolerance):
        penalty = 0
    else:
        diff = min(abs(r - (target - tolerance)), abs(r - (target + tolerance)))
        penalty = 100 * diff
    return std + penalty

# ==========================================
# 5. EXÉCUTION DYNAMIQUE
# ==========================================
tickers_a_analyser = get_selected_tickers(choix_univers)

if st.sidebar.button("⚡ Optimiser le Portefeuille", type="primary"):
    
    with st.spinner(f"Analyse de {len(tickers_a_analyser)} actifs en cours..."):
        
        # Récupération et traitement des données
        data = load_market_data(tickers_a_analyser)
        returns = data.pct_change().dropna()
        
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(mean_returns)
        
        # Plafonnement mathématique de sécurité
        max_possible_return = mean_returns.max()
        if target_return > max_possible_return:
            st.warning(f"⚠️ Rendement cible inatteignable avec les actifs sélectionnés. Ajustement automatique à {max_possible_return*100:.1f}%.")
            target_return = max_possible_return

        # Optimisation SciPy
        args = (mean_returns, cov_matrix, target_return, TOLERANCE)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
        
        result = sco.minimize(minimize_volatility_with_tolerance, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Filtrage et sélection
        raw_weights_dict = {mean_returns.index[i]: result.x[i] for i in range(num_assets) if result.x[i] > 0.001}
        sorted_weights = sorted(raw_weights_dict.items(), key=lambda item: item[1], reverse=True)[:max_actifs]
        
        if not sorted_weights:
            st.error("L'optimiseur n'a pas pu trouver de pondération viable.")
            st.stop()
            
        current_sum = sum([val for key, val in sorted_weights])
        final_weights = {key: val/current_sum for key, val in sorted_weights}
        
        # Recalcul des métriques
        w_vector = np.zeros(num_assets)
        for ticker, weight in final_weights.items():
            w_vector[list(mean_returns.index).index(ticker)] = weight
        real_r, real_std = get_portfolio_metrics(w_vector, mean_returns, cov_matrix)

        # ==========================================
        # 6. AFFICHAGE DES RÉSULTATS (DASHBOARD)
        # ==========================================
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement Annuel Espéré", f"{real_r*100:.2f}%")
        col2.metric("Volatilité (Risque)", f"{real_std*100:.2f}%", delta_color="inverse")
        col3.metric("Actifs Détenus", f"{len(final_weights)} / {len(tickers_a_analyser)}")

        st.markdown("---")
        
        col_chart, col_table = st.columns([1, 1])
        
        with col_chart:
            st.subheader("📊 Allocation d'Actifs")
            df_plot = pd.DataFrame({"Actif": list(final_weights.keys()), "Poids": list(final_weights.values())})
            fig = px.pie(df_plot, values='Poids', names='Actif', hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.subheader("📋 Positions Détaillées")
            df_table = pd.DataFrame({
                "Ticker": list(final_weights.keys()),
                "Poids (%)": [w * 100 for w in final_weights.values()],
                "Investissement (€)": [w * montant for w in final_weights.values()]
            })
            st.dataframe(df_table.style.format({"Poids (%)": "{:.1f}%", "Investissement (€)": "{:.2f} €"}), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📉 Backtest sur Données Historiques")
        selected_tickers = list(final_weights.keys())
        selected_data = data[selected_tickers]
        portfolio_daily_ret = selected_data.pct_change().fillna(0).dot(list(final_weights.values()))
        cumulative_ret = (1 + portfolio_daily_ret).cumprod() * montant
        
        st.line_chart(cumulative_ret)
else:
    st.info("👈 Ajustez vos paramètres et cliquez sur 'Optimiser le Portefeuille'.")
