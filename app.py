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
st.set_page_config(page_title="Finance Advisor 8.1 - Analyse Profonde", page_icon="📈", layout="wide")
st.title("🤖 Finance Advisor V8.1 - Modélisation Patrimoniale")
st.markdown("Optimisation Core/Satellite (PEA) + Poche Sécurisée avec analyse de corrélation.")

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
    "OBLIG_SIMUL": "🛡️ Obligation Corp (Grade A)"
}

RISK_PROFILES = {"🛡️ Prudent (5%)": 0.05, "⚖️ Équilibré (8%)": 0.08, "🚀 Dynamique (12%)": 0.12}

# ==========================================
# 3. INTERFACE UTILISATEUR (UX Optimisée)
# ==========================================
st.sidebar.header("👤 Profil Investisseur")
montant = st.sidebar.number_input("Capital à investir (€)", min_value=500, value=10000, step=500)
horizon = st.sidebar.slider("Horizon de placement (ans)", 5, 30, 10, help="Le PEA nécessite un minimum de 5 ans pour l'avantage fiscal.")
profil_nom = st.sidebar.selectbox("Profil de Risque", list(RISK_PROFILES.keys()), help="Définit l'objectif de rendement et le niveau de volatilité toléré.")
target_return = RISK_PROFILES[profil_nom]

st.sidebar.markdown("---")

with st.sidebar.expander("⚙️ Paramètres Avancés (Expert)"):
    st.markdown("**Sélecteur d'Univers**")
    choix_univers = st.radio(
        "Filtrer les actions :", 
        ("Actions & ETF", "Uniquement ETF", "Uniquement CAC 40", "Uniquement DAX 40"),
        label_visibility="collapsed"
    )
    max_actifs = st.slider("Actifs maximum", 3, 15, 8, help="Force l'algorithme à concentrer le capital.")
    
    st.markdown("---")
    st.markdown("**Poche Sécurisée (Obligation)**")
    inclure_oblig = st.checkbox("Activer l'Obligation Grade A", value=True, help="Simule un fonds en euros ou une obligation pour stabiliser le portefeuille.")
    bond_yield = st.slider("Rendement actuel estimé", 1.0, 8.0, 4.0, 0.1) / 100
    bond_vol = 0.02

# ==========================================
# 4. FONCTIONS DE RÉCUPÉRATION DE DONNÉES
# ==========================================
@st.cache_data(ttl=86400)
def load_data(tickers):
    data = yf.download(tickers, start=(datetime.now() - timedelta(days=365*10)), end=datetime.now(), auto_adjust=True)['Close']
    return data.dropna(axis=1, how='all').ffill().bfill()

@st.cache_data(ttl=604800)
def get_company_info(tickers):
    info_dict = {}
    for t in tickers:
        if t == "OBLIG_SIMUL":
            info_dict[t] = {"sector": "Obligations / Taux", "summary": "Actif synthétique simulant une obligation d'entreprise (Grade A) ou un Fonds en Euros à rendement fixe. Assure la stabilité du portefeuille.", "yield": f"{bond_yield*100:.2f}%"}
            continue
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            summary = info.get('longBusinessSummary', 'Description détaillée non fournie par le gestionnaire du fonds.')
            if len(summary) > 400: summary = summary[:400] + "..."
            sector = info.get('sector', info.get('category', 'Fonds Indiciel (ETF)'))
            y_val = info.get('dividendYield', 'N/A')
            if y_val != 'N/A' and y_val is not None: y_val = f"{y_val*100:.2f}%"
            info_dict[t] = {"sector": sector, "summary": summary, "yield": y_val}
        except:
            info_dict[t] = {"sector": "N/A", "summary": "Données fondamentales indisponibles.", "yield": "N/A"}
    return info_dict

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

    with st.spinner("Téléchargement des marchés et analyse approfondie des actifs..."):
        
        data = load_data(tickers_sub)
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        individual_vols = returns.std() * np.sqrt(252)

        if inclure_oblig:
            bond_ticker = "OBLIG_SIMUL"
            mean_returns.loc[bond_ticker] = bond_yield
            individual_vols.loc[bond_ticker] = bond_vol
            for col in cov_matrix.columns:
                cov_matrix.loc[col, bond_ticker] = 0.0
                cov_matrix.loc[bond_ticker, col] = 0.0
            cov_matrix.loc[bond_ticker, bond_ticker] = bond_vol ** 2

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
        tab1, tab2, tab3 = st.tabs(["📊 Allocation & Analyse", "🕰️ Backtest Historique", "🚀 Projections Futures"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rendement Espéré", f"{real_r*100:.1f}%", help="Rendement moyen annualisé visé par l'algorithme.")
            c2.metric("Volatilité (Risque)", f"{real_std*100:.1f}%", help="Fluctuation attendue. Plus le chiffre est bas, plus le portefeuille est stable.", delta_color="inverse")
            c3.metric("Diversification", f"{len(final_weights)} actifs", help="Nombre d'actifs retenus par le modèle.")
            st.markdown("---")
            
            c_chart, c_table = st.columns([1, 1.2])
            with c_chart:
                df_pie = pd.DataFrame({"Nom": [TICKER_NAMES.get(t, t) for t in final_weights.keys()], "Poids": list(final_weights.values())})
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
                    "Risque": st.column_config.NumberColumn("Volatilité", format="%.1f%%")
                }, hide_index=True, use_container_width=True)

            st.markdown("---")
            c_corr, c_info = st.columns([1, 1])
            
            with c_corr:
                st.subheader("🔗 Matrice de Corrélation")
                st.info("Comprendre l'algorithme : Le modèle recherche des actifs en bleu (décorrélés) pour diminuer le risque global en cas de krach boursier.")
                real_tickers = [t for t in final_weights.keys() if t != "OBLIG_SIMUL"]
                if len(real_tickers) > 1:
                    corr_matrix_display = returns[real_tickers].corr()
                    corr_matrix_display.index = [TICKER_NAMES.get(t, t) for t in corr_matrix_display.index]
                    corr_matrix_display.columns = [TICKER_NAMES.get(t, t) for t in corr_matrix_display.columns]
                    fig_corr = px.imshow(corr_matrix_display, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto", zmin=-1, zmax=1)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Pas assez d'actifs réels pour afficher une corrélation.")

            with c_info:
                st.subheader("📖 Fiches d'Identité")
                st.markdown("Découvrez en détail les actifs sélectionnés pour vous :")
                asset_info = get_company_info(list(final_weights.keys()))
                
                for t in final_weights.keys():
                    with st.expander(f"**{TICKER_NAMES.get(t, t)}** — Poids : {final_weights[t]*100:.1f}%"):
                        st.markdown(f"**Secteur / Catégorie :** {asset_info[t]['sector']}")
                        st.markdown(f"**Rendement Distribué :** {asset_info[t]['yield']}")
                        st.markdown(f"*{asset_info[t]['summary']}*")

        with tab2:
            st.info("Performance incluant la simulation de la croissance constante de l'obligation.")
            
            # Correction cruciale : on isole d'abord les vrais actifs
            real_tickers = [t for t in final_weights.keys() if t != "OBLIG_SIMUL"]
            
            if real_tickers:
                pct_change_data = data[real_tickers].pct_change().fillna(0)
            else:
                pct_change_data = pd.DataFrame(index=data.index)
            
            # Ensuite, on ajoute l'obligation mathématique
            if inclure_oblig and "OBLIG_SIMUL" in final_weights:
                daily_bond_yield = (1 + bond_yield) ** (1/252) - 1
                pct_change_data["OBLIG_SIMUL"] = daily_bond_yield
            
            # On réordonne les colonnes pour que le produit scalaire fonctionne
            pct_change_data = pct_change_data[list(final_weights.keys())]
            
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
    st.info("👈 Ajustez vos paramètres dans le panneau de gauche et cliquez sur 'Optimiser l'Allocation'.")
