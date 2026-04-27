import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# ==========================================
# APP CONFIG
# ==========================================
st.set_page_config(page_title="Analytika trhu ojetin", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
    .metric-card { background: #1e1e2e; border-radius: 10px; padding: 16px; border-left: 4px solid #7c3aed; }
    .insight-box { background: #1a2744; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    brands = {
        'skoda': 'Škoda', 'vw': 'Volkswagen', 'audi': 'Audi',
        'bmw': 'BMW', 'ford': 'Ford', 'toyota': 'Toyota',
        'merc': 'Mercedes', 'hyundi': 'Hyundai', 'vauxhall': 'Vauxhall'
    }

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df_list = []

    for prefix, name in brands.items():
        path = os.path.join(BASE_DIR, "Car_data", f"{prefix}.csv")
        if os.path.exists(path):
            try:
                tmp = pd.read_csv(path, sep=None, engine='python')
                tmp.columns = tmp.columns.str.strip()
                tmp['Značka'] = name
                df_list.append(tmp)
            except Exception as e:
                st.warning(f"Chyba při načítání {prefix}.csv: {e}")

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    df.dropna(subset=['price', 'mileage', 'year', 'model'], inplace=True)

    df['price']      = pd.to_numeric(df['price'].astype(str).str.replace(',', '.'), errors='coerce')
    df['mileage']    = pd.to_numeric(df['mileage'].astype(str).str.replace(',', '.'), errors='coerce')
    df['engineSize'] = pd.to_numeric(df['engineSize'].astype(str).str.replace(',', '.'), errors='coerce')
    df['mpg']        = pd.to_numeric(df['mpg'].astype(str).str.replace(',', '.'), errors='coerce')
    df['tax']        = pd.to_numeric(df['tax'].astype(str).str.replace(',', '.'), errors='coerce')

    df.dropna(subset=['price', 'mileage'], inplace=True)

    # Currency & unit conversion (GBP → CZK, miles → km)
    GBP_CZK = 29.5
    MI_KM   = 1.60934
    df['price']   = (df['price']   * GBP_CZK).astype(int)
    df['mileage'] = (df['mileage'] * MI_KM).astype(int)
    df['mpg_l100km'] = 282.48 / df['mpg'].replace(0, np.nan)  # convert MPG to l/100km

    # Outlier removal
    df = df[(df['price'] > 20_000) & (df['price'] < 2_500_000) & (df['mileage'] < 400_000)]
    df = df[(df['mpg'] > 10) & (df['mpg'] < 300)]
    df = df[df['year'].between(2000, 2026)]

    # *** KEY FIX: use dataset's own max year as reference, not hardcoded 2026 ***
    REFERENCE_YEAR = int(df['year'].max())
    df['age'] = REFERENCE_YEAR - df['year']

    # Price evaluation vs. median of same model+year
    df['avg_market_price'] = df.groupby(['Značka', 'model', 'year'])['price'].transform('median')
    def price_eval(row):
        if pd.isna(row['avg_market_price']): return "Neznámé"
        r = row['price'] / row['avg_market_price']
        if r < 0.85:  return "🟢 Extrémně výhodné"
        elif r < 0.95: return "🟡 Výhodná koupě"
        elif r > 1.15: return "🔴 Předražené"
        else:          return "⚪ Férová cena"
    df['Hodnocení'] = df.apply(price_eval, axis=1)

    # Value score: lower is better (price per km per year)
    df['value_score'] = df['price'] / ((df['mileage'] + 1) * (df['age'] + 1))

    # Rename
    df.rename(columns={
        'model': 'Model', 'year': 'Rok', 'price': 'Cena (Kč)',
        'mileage': 'Nájezd (km)', 'fuelType': 'Palivo',
        'engineSize': 'Motor (l)', 'transmission': 'Převodovka',
        'mpg': 'MPG', 'tax': 'Silniční daň (£)'
    }, inplace=True)

    return df, REFERENCE_YEAR

result = load_data()
if isinstance(result, pd.DataFrame):
    df = result
    REFERENCE_YEAR = 2020
else:
    df, REFERENCE_YEAR = result

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Filtry")
if not df.empty:
    vybrane_znacky = st.sidebar.multiselect("Značky:", df['Značka'].unique(), default=df['Značka'].unique())
    max_price  = st.sidebar.slider("Max. cena (Kč):", 50_000, 2_500_000, 1_200_000, step=50_000)
    max_age    = st.sidebar.slider("Max. stáří (roky):", 0, 20, 15)
    paliva     = st.sidebar.multiselect("Typ paliva:", df['Palivo'].dropna().unique(), default=df['Palivo'].dropna().unique())

    fdf = df[
        df['Značka'].isin(vybrane_znacky) &
        (df['Cena (Kč)'] <= max_price) &
        (df['age'] <= max_age) &
        df['Palivo'].isin(paliva)
    ]
else:
    fdf = pd.DataFrame()

# ==========================================
# TITLE
# ==========================================
st.title("🏎️ Komplexní analýza trhu ojetých vozů")

if df.empty:
    st.error("❌ Nenalezeny žádné CSV soubory ve složce 'Car_data'.")
    st.stop()

znacky_list = sorted(df['Značka'].unique())
st.success(f"✅ Načteno **{len(df):,}** vozů pro **{len(znacky_list)}** značek (referenční rok: {REFERENCE_YEAR}): {', '.join(znacky_list)}")

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Přehled trhu",
    "📉 Ztráta hodnoty",
    "💎 Příplatek za značku",
    "⛽ Palivo & Efektivita",
    "🔬 Skryté korelace",
    "🏆 Nejlepší modely",
    "🛒 Vyhledávač příležitostí"
])

# ──────────────────────────────────────────
# TAB 1 — PŘEHLED TRHU
# ──────────────────────────────────────────
with tab1:
    st.header("Základní přehled a distribuce cen")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inzeráty",        f"{len(fdf):,}".replace(',', ' '))
    c2.metric("Průměrná cena",   f"{int(fdf['Cena (Kč)'].mean()):,} Kč".replace(',', ' ') if len(fdf) else "—")
    c3.metric("Průměrný nájezd", f"{int(fdf['Nájezd (km)'].mean()):,} km".replace(',', ' ') if len(fdf) else "—")
    c4.metric("Mediánová cena",  f"{int(fdf['Cena (Kč)'].median()):,} Kč".replace(',', ' ') if len(fdf) else "—")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(fdf, x='Cena (Kč)', color='Značka', nbins=80, barmode='overlay',
                           opacity=0.7, title="Distribuce cen podle značky")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pie_data = fdf['Značka'].value_counts().reset_index()
        pie_data.columns = ['Značka', 'Počet']
        fig = px.pie(pie_data, names='Značka', values='Počet', hole=0.4,
                     title="Zastoupení značek v datasetu")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Price vs mileage scatter (sampled for performance)
        sample = fdf.sample(min(5000, len(fdf)), random_state=42)
        fig = px.scatter(sample, x='Nájezd (km)', y='Cena (Kč)', color='Značka',
                         opacity=0.5, trendline='lowess',
                         title="Cena vs. Nájezd (s trendem)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(fdf, x='Značka', y='Cena (Kč)', color='Značka',
                     title="Rozptyl cen podle značky")
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────
# TAB 2 — ZTRÁTA HODNOTY
# ──────────────────────────────────────────
with tab2:
    st.header("Depreciation: Jak rychle padá hodnota?")

    dep = fdf.groupby(['age', 'Značka'])['Cena (Kč)'].mean().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(dep, x='age', y='Cena (Kč)', color='Značka', markers=True,
                      labels={'age': 'Stáří (roky)'}, title="Absolutní pokles průměrné ceny")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Relative depreciation: normalize each brand to age=0
        dep_pivot = dep.pivot(index='age', columns='Značka', values='Cena (Kč)')
        dep_norm = (dep_pivot / dep_pivot.iloc[0] * 100).reset_index().melt(id_vars='age', var_name='Značka', value_name='% původní ceny')
        fig = px.line(dep_norm, x='age', y='% původní ceny', color='Značka', markers=True,
                      labels={'age': 'Stáří (roky)'}, title="Relativní ztráta hodnoty (index: nové = 100 %)")
        fig.add_hline(y=50, line_dash='dot', line_color='red', annotation_text='50 % hodnoty')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Průměrná roční ztráta hodnoty (%) — věk 0–5 let")
    annual_dep = []
    for brand in fdf['Značka'].unique():
        b = dep[dep['Značka'] == brand].sort_values('age')
        age0 = b[b['age'] == 0]['Cena (Kč)'].values
        age5 = b[b['age'] == 5]['Cena (Kč)'].values
        if len(age0) and len(age5) and age0[0] > 0:
            pct = round((1 - (age5[0] / age0[0])) * 100 / 5, 1)
            annual_dep.append({'Značka': brand, 'Roční depreciace (%)': pct})
    if annual_dep:
        ad_df = pd.DataFrame(annual_dep).sort_values('Roční depreciace (%)', ascending=False)
        fig = px.bar(ad_df, x='Značka', y='Roční depreciace (%)', color='Značka',
                     title="Průměrná roční depreciace (první 5 let)")
        st.plotly_chart(fig, use_container_width=True)
    
# ──────────────────────────────────────────
# TAB 3 — PŘÍPLATEK ZA ZNAČKU  (FIXED)
# ──────────────────────────────────────────
with tab3:
    st.header("Brand Premium: Platíme za logo?")
    st.markdown("""
    Porovnáváme **auta stará 3–7 let** s motorem **1.4–2.0 l** — standardní segment napříč všemi značkami.
    _(Věk je počítán od posledního roku v datasetu, **{ref}**)_
    """.format(ref=REFERENCE_YEAR))

    fair = fdf[
        fdf['age'].between(3, 7) &
        fdf['Motor (l)'].between(1.4, 2.0)
    ]

    if len(fair) < 10:
        st.warning(f"Nalezeno pouze {len(fair)} vozů pro tento segment. Zkuste rozšířit filtry v postranním panelu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(fair, x='Značka', y='Cena (Kč)', color='Značka',
                         title=f"Rozptyl cen — věk 3–7 let, motor 1.4–2.0 l (n={len(fair):,})")
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            median_df = fair.groupby('Značka')['Cena (Kč)'].median().sort_values(ascending=False).reset_index()
            # Premium relative to cheapest brand
            min_price = median_df['Cena (Kč)'].min()
            median_df['Příplatek vs. nejlevnější (Kč)'] = (median_df['Cena (Kč)'] - min_price).astype(int)
            fig = px.bar(median_df, x='Značka', y='Příplatek vs. nejlevnější (Kč)', color='Značka',
                         title="Příplatek za značku vs. nejlevnější alternativa")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Mediánové ceny tohoto segmentu")
        median_df['Cena (Kč)'] = median_df['Cena (Kč)'].apply(lambda x: f"{int(x):,} Kč".replace(',', ' '))
        median_df['Příplatek vs. nejlevnější (Kč)'] = median_df['Příplatek vs. nejlevnější (Kč)'].apply(lambda x: f"+{int(x):,} Kč".replace(',', ' '))
        st.dataframe(median_df, use_container_width=True)

        # Transmission premium
        st.subheader("💡 Příplatek za automatickou převodovku")
        trans_df = fair.groupby(['Značka', 'Převodovka'])['Cena (Kč)'].median().reset_index()
        fig = px.bar(trans_df, x='Značka', y='Cena (Kč)', color='Převodovka', barmode='group',
                     title="Manuál vs. Automat — mediánová cena")
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────
# TAB 4 — PALIVO & EFEKTIVITA
# ──────────────────────────────────────────
with tab4:
    st.header("⛽ Palivo, Spotřeba a Ekologie")

    col1, col2 = st.columns(2)

    with col1:
        fuel_year = fdf.groupby(['Rok', 'Palivo']).size().reset_index(name='Počet')
        fuel_year_pct = fuel_year.copy()
        fuel_year_pct['%'] = fuel_year_pct.groupby('Rok')['Počet'].transform(lambda x: x / x.sum() * 100)
        fig = px.area(fuel_year_pct, x='Rok', y='%', color='Palivo',
                      title="Trend pohonných hmot napříč lety (%)",
                      labels={'%': 'Podíl (%)'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fuel_price = fdf.groupby('Palivo')['Cena (Kč)'].median().reset_index()
        fig = px.bar(fuel_price.sort_values('Cena (Kč)', ascending=False),
                     x='Palivo', y='Cena (Kč)', color='Palivo',
                     title="Mediánová cena vozu podle pohonu")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Efficiency by brand
        eff = fdf.groupby('Značka')['mpg_l100km'].median().reset_index().sort_values('mpg_l100km')
        fig = px.bar(eff, x='Značka', y='mpg_l100km', color='Značka',
                     labels={'mpg_l100km': 'Průměrná spotřeba (l/100 km)'},
                     title="Průměrná spotřeba podle značky (nižší = lepší)")
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        eff_fuel = fdf.groupby(['Palivo', 'Značka'])['mpg_l100km'].median().reset_index()
        fig = px.bar(eff_fuel, x='Značka', y='mpg_l100km', color='Palivo', barmode='group',
                     labels={'mpg_l100km': 'Spotřeba (l/100 km)'},
                     title="Spotřeba: Diesel vs. Benzín vs. Hybrid")
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spotřeba vs. Cena: Platí se za efektivitu?")
    sample = fdf.sample(min(4000, len(fdf)), random_state=1)
    fig = px.scatter(sample, x='mpg_l100km', y='Cena (Kč)', color='Palivo',
                     size='Motor (l)', hover_data=['Značka', 'Model'],
                     labels={'mpg_l100km': 'Spotřeba (l/100 km)'},
                     title="Vztah spotřeby a ceny (velikost bodu = objem motoru)",
                     opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────
# TAB 5 — SKRYTÉ KORELACE
# ──────────────────────────────────────────
with tab5:
    st.header("🔬 Skryté korelace a statistické vzorce")
 
    numeric = fdf[['Cena (Kč)', 'Nájezd (km)', 'age', 'Motor (l)', 'mpg_l100km', 'Silniční daň (£)']].dropna()
 
    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(numeric.corr(), text_auto='.2f', color_continuous_scale='RdBu_r',
                        title="Korelační heatmapa (Pearson)")
        st.plotly_chart(fig, use_container_width=True)
 
    with col2:
        # Engine size vs price by brand
        eng_price = fdf.groupby(['Značka', 'Motor (l)'])['Cena (Kč)'].median().reset_index()
        fig = px.scatter(eng_price, x='Motor (l)', y='Cena (Kč)', color='Značka',
                         size='Cena (Kč)', title="Objem motoru vs. cena (per značka)",
                         hover_data=['Značka'])
        st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("Silniční daň jako proxy výkonu")
    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(fdf.sample(min(3000, len(fdf)), random_state=5),
                         x='Silniční daň (£)', y='Cena (Kč)', color='Značka',
                         opacity=0.5, trendline='ols',
                         title="Silniční daň vs. cena vozu (OLS trend)")
        st.plotly_chart(fig, use_container_width=True)
 
    with col4:
        fig = px.scatter(fdf.sample(min(3000, len(fdf)), random_state=6),
                         x='Silniční daň (£)', y='mpg_l100km', color='Palivo',
                         opacity=0.5, trendline='ols',
                         title="Silniční daň vs. spotřeba (daň roste se spotřebou?)")
        st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("Cenový výkyv podle roku modelu — kde jsou anomálie?")
    yr_brand = fdf.groupby(['Rok', 'Značka'])['Cena (Kč)'].median().reset_index()
    fig = px.line(yr_brand, x='Rok', y='Cena (Kč)', color='Značka', markers=False,
                  title="Medián ceny dle roku výroby (odhalí anomální ročníky)")
    st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("📐 Multiple Regression: Co opravdu určuje cenu?")
    st.markdown("Model trénujeme na **numerických proměnných + zakódovaných kategoriích** (značka, palivo, převodovka). Výsledek ukazuje skutečný přínos každého faktoru při kontrole ostatních.")
 
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        import warnings
        warnings.filterwarnings('ignore')
 
        reg_cols = ['Cena (Kč)', 'Nájezd (km)', 'age', 'Motor (l)', 'mpg_l100km',
                    'Silniční daň (£)', 'Značka', 'Palivo', 'Převodovka']
        reg_df = fdf[reg_cols].dropna()
 
        # One-hot encode categorical columns
        reg_encoded = pd.get_dummies(reg_df, columns=['Značka', 'Palivo', 'Převodovka'], drop_first=True)
 
        X = reg_encoded.drop(columns=['Cena (Kč)'])
        y = reg_encoded['Cena (Kč)']
 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
 
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
 
        # Cross-validated R²
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        r2_mean = cv_scores.mean()
 
        # Feature importances (standardized coefficients)
        coef_df = pd.DataFrame({
            'Proměnná': X.columns,
            'Koeficient (std.)': model.coef_
        }).sort_values('Koeficient (std.)', key=abs, ascending=False).head(20)
 
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric("R² skóre modelu (5-fold CV)", f"{r2_mean:.3f}",
                      help="1.0 = perfektní predikce, 0.0 = model nic nevysvětluje")
            st.metric("Počet vzorků", f"{len(reg_df):,}".replace(',', ' '))
            st.metric("Počet prediktorů", f"{X.shape[1]}")
            st.markdown("""
            **Jak číst koeficienty:**
            - Hodnoty jsou **standardizované** → lze porovnávat napříč proměnnými
            - **Kladný** koeficient = zvyšuje cenu
            - **Záporný** koeficient = snižuje cenu
            - Čím větší absolutní hodnota, tím větší vliv
            """)
 
        with col_r2:
            coef_df['Barva'] = coef_df['Koeficient (std.)'].apply(lambda x: '🔴 Zdražuje' if x > 0 else '🔵 Zlevňuje')
            fig = px.bar(coef_df, x='Koeficient (std.)', y='Proměnná', orientation='h',
                         color='Barva', color_discrete_map={'🔴 Zdražuje': '#ef4444', '🔵 Zlevňuje': '#3b82f6'},
                         title="Top 20 faktorů ovlivňujících cenu (standardizované koeficienty)")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=550)
            st.plotly_chart(fig, use_container_width=True)
 
        # Predicted vs Actual
        st.subheader("Predikovaná vs. skutečná cena")
        y_pred = model.predict(X_scaled)
        pred_df = pd.DataFrame({'Skutečná cena (Kč)': y.values, 'Predikovaná cena (Kč)': y_pred})
        pred_sample = pred_df.sample(min(2000, len(pred_df)), random_state=7)
        fig2 = px.scatter(pred_sample, x='Skutečná cena (Kč)', y='Predikovaná cena (Kč)',
                          opacity=0.4, title=f"Predikce modelu vs. realita (R² = {r2_mean:.3f})")
        max_val = pred_df.max().max()
        fig2.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                       line=dict(color='red', dash='dot'), name='Ideální predikce')
        st.plotly_chart(fig2, use_container_width=True)
 
    except ImportError:
        st.warning("Pro tuto sekci nainstaluj: `pip install scikit-learn`")
# ──────────────────────────────────────────
# TAB 6 — NEJLEPŠÍ MODELY
# ──────────────────────────────────────────
with tab6:
    st.header("🏆 Ranking modelů — hodnota za peníze")

    brand_sel = st.selectbox("Vyberte značku:", sorted(fdf['Značka'].unique()))
    brand_data = fdf[fdf['Značka'] == brand_sel]

    col1, col2 = st.columns(2)
    with col1:
        model_counts = brand_data['Model'].value_counts().head(15).reset_index()
        model_counts.columns = ['Model', 'Počet']
        fig = px.bar(model_counts, x='Model', y='Počet', title=f"Nejčastější modely — {brand_sel}")
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        model_price = brand_data.groupby('Model')['Cena (Kč)'].median().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(model_price, x='Model', y='Cena (Kč)', title=f"Mediánová cena modelů — {brand_sel}", color='Cena (Kč)', color_continuous_scale='Viridis')
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Ztráta hodnoty per model — {brand_sel}")
    dep_model = brand_data.groupby(['Model', 'age'])['Cena (Kč)'].median().reset_index()
    top_models = brand_data['Model'].value_counts().head(8).index
    fig = px.line(dep_model[dep_model['Model'].isin(top_models)],
                  x='age', y='Cena (Kč)', color='Model', markers=True,
                  labels={'age': 'Stáří (roky)'},
                  title=f"Depreciation curve top 8 modelů — {brand_sel}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Nájezd vs. Cena per model — {brand_sel}")
    sample_brand = brand_data[brand_data['Model'].isin(top_models)].sample(min(2000, len(brand_data[brand_data['Model'].isin(top_models)])), random_state=42)
    fig = px.scatter(sample_brand, x='Nájezd (km)', y='Cena (Kč)', color='Model',
                     opacity=0.6, trendline='lowess', title=f"Cena vs. nájezd pro top modely — {brand_sel}")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────
# TAB 7 — VYHLEDÁVAČ PŘÍLEŽITOSTÍ
# ──────────────────────────────────────────
with tab7:
    st.header("🛒 Vyhledávač anomálií a příležitostí")
    st.markdown("Model porovnává každý inzerát s mediánem trhu pro stejný model a ročník. 🟢 = minimálně 15 % pod mediánem.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        typ = st.radio("Zobrazit:", ["Vše", "Pouze výhodné (🟢 a 🟡)", "Pouze předražené (🔴)"], horizontal=False)
    with col_f2:
        palivo_filter = st.multiselect("Palivo:", fdf['Palivo'].dropna().unique(), default=list(fdf['Palivo'].dropna().unique()))
    with col_f3:
        trans_filter  = st.multiselect("Převodovka:", fdf['Převodovka'].dropna().unique(), default=list(fdf['Převodovka'].dropna().unique()))

    cols_show = ['Značka', 'Model', 'Rok', 'Cena (Kč)', 'Nájezd (km)', 'Palivo', 'Převodovka', 'Motor (l)', 'Hodnocení', 'avg_market_price']
    cols_show = [c for c in cols_show if c in fdf.columns]

    result_df = fdf[fdf['Palivo'].isin(palivo_filter) & fdf['Převodovka'].isin(trans_filter)][cols_show].copy()
    result_df['avg_market_price'] = result_df['avg_market_price'].apply(lambda x: f"{int(x):,} Kč".replace(',', ' ') if not pd.isna(x) else '—')

    if typ == "Pouze výhodné (🟢 a 🟡)":
        result_df = result_df[result_df['Hodnocení'].str.contains('🟢|🟡', na=False)]
    elif typ == "Pouze předražené (🔴)":
        result_df = result_df[result_df['Hodnocení'].str.contains('🔴', na=False)]

    st.markdown(f"**Zobrazeno {len(result_df):,} inzerátů**")

    st.subheader("Distribuce hodnocení")
    rating_counts = fdf['Hodnocení'].value_counts().reset_index()
    rating_counts.columns = ['Hodnocení', 'Počet']
    fig = px.bar(rating_counts, x='Hodnocení', y='Počet', color='Hodnocení',
                 title="Kolik % inzerátů je opravdu výhodných?")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        result_df.rename(columns={'avg_market_price': 'Tržní medián'}),
        use_container_width=True,
        height=500
    )