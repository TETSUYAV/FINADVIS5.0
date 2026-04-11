import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Finance Advisor 7.0 - Multi-Enveloppes", page_icon="📈", layout="wide")
st.title("🤖 Finance Advisor V7.0 - Modélisation Patrimoniale")
st.markdown("Optimisation Core/Satellite (PEA) + Poche Sécurisée (Obligation Synthétique)")

# ==========================================
# 2. UNIVERS D'INVESTISSEMENT (ACTIONS / ETF)
# ==========================================
ASSET_UNIVERSE = {
    "CAC 40 (Top 20)": [
        "MC.PA", "OR.PA", "TTE.PA", "RMS.PA", "SAN.PA", "AIR.PA", "SU.PA", "AI.PA", 
        "BNP.PA", "SAF.PA", "CAP.PA", "DSY.PA", "CS.PA", "DG.PA", "EN.PA"
    ],
    "DAX 40 (Top 20)": [
        "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MBG.DE", "BMW.DE", "MUV2.DE", "BAS.DE", 
        "IFX.DE", "BAYN.DE", "DB1.DE", "DBK.DE"
    ],
    "ETF & Thématiques (PEA)": [
        "CW8.PA",    # MSCI World
        "PSP5.PA",   # S&P 500
        "PUST.PA",   # Nasdaq-100
        "PAEEM.PA",  # Emerging Markets
        "RS2K.PA",   # Russell 2000
        "ENER.PA",   # New Energy
        "HLTH.PA",   # Healthcare
        "PTE.PA"     # Tech Europe
    ]
}

TICKER_NAMES = {
    "MC.PA": "LVMH", "OR.PA": "L'Oréal", "TTE.PA": "TotalEnergies", "RMS.PA": "Hermès", 
    "SAN.PA": "Sanofi", "AIR.PA": "Airbus", "SU.PA": "Schneider", "AI.PA": "Air Liquide", 
    "BNP.PA": "BNP Paribas", "SAF.PA": "Safran", "CAP.PA": "Capgemini", "DSY.PA": "Dassault", 
    "CS.PA": "AXA", "DG.PA": "Vinci", "EN.PA": "Bouygues",
    "SAP.DE": "SAP", "SIE.DE": "Siemens", "ALV.DE": "Allianz", "DTE.DE": "Deutsche Telekom", 
    "MBG.DE": "Mercedes-Benz", "BMW.DE": "BMW", "MUV2.DE": "Munich Re", "BAS.DE": "BASF", 
    "IFX.DE": "Infineon", "BAYN.DE": "Bayer", "DB1.DE": "Deutsche Börse", "DBK.DE": "Deutsche Bank",
    "CW8.PA": "Amundi MSCI World", "PSP5.PA": "Amundi S&P 500", 
    "PUST.PA": "Amundi Nasdaq-100", "PAEEM.PA": "Amundi Emerging", 
    "RS2K.PA": "Amundi Russell 2000", "ENER.PA": "Lyxor New Energy", 
    "HLTH.PA": "Lyxor Healthcare", "PTE.PA": "Lyxor Tech Europe",
    "OBLIG_SIMUL": "🛡️ Obligation Corp (Grade A)" # Notre actif synthétique
}

RISK_PROFILES = {"🛡️ Prudent (5%)": 0.05, "⚖️ Équilibré (8%)": 0.08, "🚀 Dynamique (12%)": 0.12}

# ==========================================
# 3. INTERFACE UTILISATEUR
# ==========================================
st.sidebar.header("🛠️ Paramètres du Portefeuille")
montant = st.sidebar.number_input("Capital (€)", min_value=500, value=10000, step=500)
profil_nom = st.sidebar.selectbox("Profil de Risque", list(RISK_PROFILES.keys()))
target_return = RISK_PROFILES[profil_nom]

st.sidebar.markdown("---")
st.sidebar.subheader("🏦 Paramétrage Obligataire")
st.sidebar.info("L'algorithme utilisera un actif synthétique pour stabiliser le portefeuille.")
inclure_oblig = st.sidebar.checkbox("Activer l'Obligation Grade A", value=True)
bond_yield = st.sidebar.slider("Rendement actuel de l'obligation", 1.0, 8.0, 4.0, 0.1) / 100
bond_vol = 0.02 # Volatilité fixe estimée à 2% pour une Corp Grade A

st.sidebar.markdown("---")
choix_univers = st.sidebar.radio("Sélecteur d'Actions (PEA)", 
    ("Actions & ETF", "Uniquement ETF", "Uniquement CAC 40", "Uniquement DAX 40")
)
max_actifs = st.sidebar.slider("Diversification (Actifs max)", 3, 15, 8)
horizon = st.sidebar.slider("Horizon de placement (ans)", 5, 30, 10)

# ==========================================
# 4. LOGIQUE MATHÉMATIQUE
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
# 5. EXECUTION DYNAMIQUE
# ==========================================
if st.sidebar.button("⚡ Optimiser l'Allocation", type="primary"):
    
    tickers_sub = []
    if choix_univers == "Uniquement ETF": tickers_sub = ASSET_UNIVERSE["ETF & Thématiques (PEA)"]
    elif choix_univers == "Uniquement CAC 40": tickers_sub = ASSET_UNIVERSE["CAC 40 (Top 20)"]
    elif choix_univers == "Uniquement DAX 40": tickers_sub = ASSET_UNIVERSE["DAX 40 (Top 20)"]
    else: 
        for cat in ASSET_UNIVERSE.values(): tickers_sub.extend(cat)

    with st.spinner("Téléchargement des marchés et injection du proxy obligataire..."):
        
        # A. Données réelles
        data = load_data(tickers_sub)
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        individual_vols = returns.std() * np.sqrt(252)

        # B. INJECTION DE L'OBLIGATION SYNTHÉTIQUE
        if inclure_oblig:
            bond_ticker = "OBLIG_SIMUL"
            # 1. Ajout de l'espérance de rendement
            mean_returns.loc[bond_ticker] = bond_yield
            individual_vols.loc[bond_ticker] = bond_vol
            
            # 2. Agrandissement de la matrice de covariance
            # On suppose une corrélation de 0 avec les actions (indépendance pure)
            for col in cov_matrix.columns:
                cov_matrix.loc[col, bond_ticker] = 0.0
                cov_matrix.loc[bond_ticker, col] = 0.0
            
            # 3. Ajout de la variance de l'obligation sur la diagonale
            cov_matrix.loc[bond_ticker, bond_ticker] = bond_vol ** 2
            
            # (Pour le backtest historique, on crée une colonne de rendement constant)
            data[bond_ticker] = (1 + bond_yield/252) ** np.arange(len(data)) * data.iloc[0,0] # Simulation de prix

        # C. Optimisation
        num = len(mean_returns)
        max_ret = mean_returns.max()
        if target_return > max_ret: target_return = max_ret
            
        res = sco.minimize(opt_func, num*[1./num], args=(mean_returns, cov_matrix, target_return), 
                           method='SLSQP', bounds=tuple((0,1) for _ in range(num)), constraints={'type':'eq','fun':lambda x:np.sum(x)-1})
        
        weights_dict = {mean_returns.index[i]: res.x[i] for i in range(num) if res.x[i] > 0.005}
        top_weights = dict(sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)[:max_actifs])
        
        if not top_weights: st.error("Impossible de générer le portefeuille.") ; st.stop()
        final_weights = {k: v/sum(top_weights.values()) for k, v in top_weights.items()}

        w_vector = np.zeros(num)
        for ticker, weight in final_weights.items(): w_vector[list(mean_returns.index).index(ticker)] = weight
        real_r, real_std = get_metrics(w_vector, mean_returns, cov_matrix)

        # ==========================================
        # 6. DASHBOARD
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["📊 Composition du Portefeuille", "🕰️ Backtest Historique", "🚀 Projections Futures"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rendement Espéré", f"{real_r*100:.1f}%")
            c2.metric("Volatilité (Risque)", f"{real_std*100:.1f}%")
            c3.metric("Diversification", f"{len(final_weights)} actifs")
            st.markdown("---")
            
            c_chart, c_table = st.columns([1, 1.2])
            with c_chart:
                df_pie = pd.DataFrame({"Nom": [TICKER_NAMES.get(t, t) for t in final_weights.keys()], "Poids": list(final_weights.values())})
                
                # Couleurs personnalisées (Obligation en Bleu, Actions en autres couleurs)
                color_map = {TICKER_NAMES["OBLIG_SIMUL"]: '#2E86C1'}
                fig = px.pie(df_pie, values='Poids', names='Nom', hole=0.4, color='Nom', color_discrete_map=color_map)
                st.plotly_chart(fig.update_traces(textposition='outside').update_layout(showlegend=False), use_container_width=True)
            
            with c_table:
                df_table = pd.DataFrame({
                    "Actif": [TICKER_NAMES.get(t, t) for t in final_weights.keys()],
                    "Poids": [v*100 for v in final_weights.values()],
                    "Montant": [v*montant for v in final_weights.values()],
                    "Risque": [individual_vols[t]*100 for t in final_weights.keys()]
                })
                st.dataframe(df_table, column_config={
                    "Actif": st.column_config.TextColumn(width="medium"),
                    "Poids": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "Montant": st.column_config.NumberColumn(format="%.2f €"),
                    "Risque": st.column_config.NumberColumn(format="%.1f%%")
                }, hide_index=True, use_container_width=True)

        with tab2:
            st.info("Performance incluant la simulation de la croissance constante de l'obligation.")
            # Calcul des rendements journaliers historiques
            pct_change_data = data[list(final_weights.keys())].pct_change().fillna(0)
            
            # Remplacement des données "vides" de l'obligation par son rendement journalier théorique
            if inclure_oblig and "OBLIG_SIMUL" in final_weights:
                daily_bond_yield = (1 + bond_yield) ** (1/252) - 1
                pct_change_data["OBLIG_SIMUL"] = daily_bond_yield
                
            hist_ret = pct_change_data.dot(list(final_weights.values()))
            st.line_chart((1 + hist_ret).cumprod() * montant)

        with tab3:
            st.info("Modélisation géométrique brownienne (L'obligation incluse génère des intérêts composés constants).")
            years = np.arange(0, horizon + 1)
            
            def calc_path(type_scen):
                path = [montant]
                for y in range(1, horizon + 1):
                    if type_scen == "Pessimiste": r = (real_r - 2*real_std) if y > 1 else -0.20
                    elif type_scen == "Normal": r = real_r
                    elif type_scen == "Optimiste": r = real_r + 0.8*real_std
                    path.append(path[-1] * (1 + r))
                return path

            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=years, y=calc_path("Optimiste"), name="🚀 Optimiste", line=dict(color='green', dash='dash')))
            fig_proj.add_trace(go.Scatter(x=years, y=calc_path("Normal"), name="⚖️ Normal (Attendu)", fill='tonexty', line=dict(color='blue', width=3)))
            fig_proj.add_trace(go.Scatter(x=years, y=calc_path("Pessimiste"), name="📉 Pessimiste (Krach Actions)", line=dict(color='red')))
            fig_proj.update_layout(xaxis_title="Années", yaxis_title="Valeur Projetée (€)", hovermode="x unified")
            st.plotly_chart(fig_proj, use_container_width=True)
            
            if inclure_oblig and "OBLIG_SIMUL" in final_weights:
                poids_secu = final_weights["OBLIG_SIMUL"] * 100
                st.success(f"🛡️ Le scénario pessimiste est fortement amorti car **{poids_secu:.1f}%** de votre portefeuille est protégé par l'obligation à {bond_yield*100:.1f}%.")
else:
    st.info("👈 Ajustez vos paramètres (dont le taux de l'obligation) et cliquez sur 'Optimiser'.")
