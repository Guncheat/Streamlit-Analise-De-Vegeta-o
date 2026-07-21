import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuração da Página
st.set_page_config(page_title="Monitoramento NDVI - Paraíba", layout="wide")
st.title("🌱 Previsão Espaço-Temporal de NDVI na Caatinga")
st.markdown("Comparação interativa entre dados reais de satélite e predições do modelo LightGBM para o estado da Paraíba.")

# 2. Carregamento dos Dados (com cache para não travar a web)
@st.cache_data
def carregar_dados():
    # Substitua pelo nome do seu arquivo final exportado
    df = pd.read_csv("df_final.csv") 
    
    # Criar uma coluna de Data para facilitar o filtro temporal
    df['Data'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Mes'].astype(str) + '-01')
    return df

df = carregar_dados()

# 3. Barra Lateral (Filtros Temporais)
st.sidebar.header("Filtros Temporais")

# Opção de visualizar por Ano ou por Mês/Ano específico
tipo_filtro = st.sidebar.radio("Filtrar por:", ["Ano Completo", "Mês Específico"])

if tipo_filtro == "Ano Completo":
    ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", df['Ano'].unique())
    df_filtrado = df[df['Ano'] == ano_selecionado]
    periodo_texto = f"Ano: {ano_selecionado}"
else:
    datas_disponiveis = df['Data'].dt.strftime('%Y-%m').unique()
    data_selecionada = st.sidebar.selectbox("Selecione o Mês/Ano:", sorted(datas_disponiveis))
    df_filtrado = df[df['Data'].dt.strftime('%Y-%m') == data_selecionada]
    periodo_texto = f"Período: {data_selecionada}"

st.sidebar.markdown("---")
st.sidebar.info("Este painel demonstra o uso de Machine Learning (LightGBM) para predição de biomassa considerando a memória hídrica da vegetação.")

# 4. Renderização dos Mapas
st.subheader(f"Visão Espacial - {periodo_texto}")

# Layout em duas colunas para comparação lado a lado
col1, col2 = st.columns(2)

# Mapa 1: NDVI Real
with col1:
    st.markdown("**NDVI Real (Observado via Satélite)**")
    fig_real = px.scatter_mapbox(
        df_filtrado, 
        lat="Latitude", 
        lon="Longitude", 
        color="NDVI",
        color_continuous_scale="RdYlGn",
        range_color=[0, 0.9],
        zoom=5.5, 
        center={"lat": -7.115, "lon": -36.5}, # Centro aproximado da Paraíba
        mapbox_style="carto-positron"
    )
    fig_real.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_real, use_container_width=True)

# Mapa 2: NDVI Predito pelo LightGBM
with col2:
    st.markdown("**NDVI Predito (Modelo LightGBM)**")
    fig_pred = px.scatter_mapbox(
        df_filtrado, 
        lat="Latitude", 
        lon="Longitude", 
        color="NDVI_Predito",
        color_continuous_scale="RdYlGn",
        range_color=[0, 0.9],
        zoom=5.5, 
        center={"lat": -7.115, "lon": -36.5},
        mapbox_style="carto-positron"
    )
    fig_pred.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_pred, use_container_width=True)

# 5. Métricas Rápidas de Avaliação no Rodapé
st.markdown("---")
st.subheader("Métricas do Período Selecionado")
erro_medio = (df_filtrado['NDVI'] - df_filtrado['NDVI_Predito']).abs().mean()
st.metric(label="Erro Médio Absoluto (MAE) da visualização atual", value=f"{erro_medio:.4f}")