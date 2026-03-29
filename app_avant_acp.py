import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_classification

# ---------------------------------------------------
# Configuration générale
# ---------------------------------------------------
st.set_page_config(
    page_title="Dashboard avant ACP",
    layout="wide"
)

st.title("Dashboard exploratoire avant ACP")
st.markdown("""
Ce dashboard a pour objectif de montrer les limites de la visualisation classique
lorsqu'on manipule un dataset volumineux et multidimensionnel.
""")

# ---------------------------------------------------
# Génération d'un dataset massif synthétique
# ---------------------------------------------------
@st.cache_data
def generate_data(n_samples=100000, n_features=20, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=6,
        n_repeated=0,
        n_classes=4,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=random_state
    )
    
    columns = [f"Var_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["Classe"] = y.astype(str)
    return df

df = generate_data()

numeric_cols = [c for c in df.columns if c != "Classe"]

# ---------------------------------------------------
# Sidebar : contrôles
# ---------------------------------------------------
st.sidebar.header("Paramètres")

hist_var = st.sidebar.selectbox("Variable pour l'histogramme", numeric_cols, index=0)
box_var = st.sidebar.selectbox("Variable pour le boxplot", numeric_cols, index=1)
x_var = st.sidebar.selectbox("Variable X pour le scatter plot", numeric_cols, index=2)
y_var = st.sidebar.selectbox("Variable Y pour le scatter plot", numeric_cols, index=3)

sample_scatter = st.sidebar.slider("Taille échantillon scatter", 500, 10000, 3000, 500)
sample_matrix = st.sidebar.slider("Taille échantillon scatter matrix", 200, 3000, 800, 100)
sample_parallel = st.sidebar.slider("Taille échantillon parallel coordinates", 100, 1500, 300, 50)

matrix_vars = st.sidebar.multiselect(
    "Variables pour la scatter matrix",
    numeric_cols,
    default=numeric_cols[:5]
)

parallel_vars = st.sidebar.multiselect(
    "Variables pour les coordonnées parallèles",
    numeric_cols,
    default=numeric_cols[:6]
)

# ---------------------------------------------------
# KPI Cards
# ---------------------------------------------------
st.subheader("1. Vue d'ensemble")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Nombre de lignes", f"{df.shape[0]:,}".replace(",", " "))
with col2:
    st.metric("Nombre de variables", len(numeric_cols))
with col3:
    st.metric("Nombre de classes", df["Classe"].nunique())
with col4:
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    st.metric("Taille mémoire", f"{memory_mb:.2f} MB")

# ---------------------------------------------------
# Table brute
# ---------------------------------------------------
st.subheader("2. Table brute : première limite")
st.markdown("""
Même si les données sont accessibles, un tableau avec 100 000 lignes et beaucoup de colonnes
devient très vite difficile à lire et à interpréter globalement.
""")
st.dataframe(df.head(100), use_container_width=True)

# ---------------------------------------------------
# Histogramme
# ---------------------------------------------------
st.subheader("3. Histogramme : une seule variable à la fois")
fig_hist = px.histogram(
    df,
    x=hist_var,
    color="Classe",
    nbins=60,
    barmode="overlay",
    opacity=0.6,
    title=f"Distribution de {hist_var}"
)
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------
# Boxplot
# ---------------------------------------------------
st.subheader("4. Boxplot : comparaison d'une variable selon la classe")
fig_box = px.box(
    df.sample(5000, random_state=42),
    x="Classe",
    y=box_var,
    color="Classe",
    title=f"Boxplot de {box_var} par classe"
)
st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------
# Scatter plot 2D
# ---------------------------------------------------
st.subheader("5. Scatter plot : seulement deux variables à la fois")
scatter_df = df.sample(sample_scatter, random_state=42)
fig_scatter = px.scatter(
    scatter_df,
    x=x_var,
    y=y_var,
    color="Classe",
    opacity=0.5,
    title=f"Nuage de points : {x_var} vs {y_var}"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.info(
    "Même avec ce visuel, on ne voit qu'une relation entre 2 variables. "
    "Le dataset en contient 20 : la lecture globale reste donc limitée."
)

# ---------------------------------------------------
# Heatmap de corrélation
# ---------------------------------------------------
st.subheader("6. Heatmap de corrélation : vue globale mais dense")
corr = df[numeric_cols].corr()

fig_corr = px.imshow(
    corr,
    text_auto=False,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Matrice de corrélation"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------
# Top corrélations
# ---------------------------------------------------
st.subheader("7. Paires de variables les plus corrélées")
corr_abs = corr.abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

top_corr = (
    upper.stack()
    .reset_index()
    .rename(columns={"level_0": "Variable_1", "level_1": "Variable_2", 0: "Corrélation_absolue"})
    .sort_values("Corrélation_absolue", ascending=False)
    .head(10)
)

st.dataframe(top_corr, use_container_width=True)

# ---------------------------------------------------
# Scatter matrix
# ---------------------------------------------------
st.subheader("8. Scatter matrix : très vite surchargée")
if len(matrix_vars) >= 2:
    sm_df = df.sample(sample_matrix, random_state=42)
    fig_matrix = px.scatter_matrix(
        sm_df,
        dimensions=matrix_vars,
        color="Classe",
        title="Scatter matrix"
    )
    fig_matrix.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig_matrix, use_container_width=True)
else:
    st.warning("Sélectionne au moins 2 variables pour la scatter matrix.")

# ---------------------------------------------------
# Parallel coordinates
# ---------------------------------------------------
st.subheader("9. Coordonnées parallèles : plusieurs dimensions mais lecture difficile")
if len(parallel_vars) >= 2:
    pc_df = df.sample(sample_parallel, random_state=42).copy()
    pc_df["Classe_num"] = pc_df["Classe"].astype(int)

    fig_parallel = px.parallel_coordinates(
        pc_df,
        dimensions=parallel_vars,
        color="Classe_num",
        title="Coordonnées parallèles"
    )
    st.plotly_chart(fig_parallel, use_container_width=True)
else:
    st.warning("Sélectionne au moins 2 variables pour les coordonnées parallèles.")

# ---------------------------------------------------
# Conclusion pédagogique
# ---------------------------------------------------
st.subheader("10. Conclusion avant ACP")
st.success("""
Constat :
- le tableau brut est trop volumineux ;
- les histogrammes et boxplots n'analysent qu'une variable à la fois ;
- le scatter plot n'affiche que deux dimensions ;
- la heatmap donne une vue globale mais condensée ;
- la scatter matrix et les coordonnées parallèles deviennent vite difficiles à lire.

C'est exactement pour cette raison qu'on introduit ensuite l'ACP :
elle permet de résumer plusieurs variables en quelques composantes principales
plus faciles à visualiser et à interpréter.
""")
