import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Configuração da Página
st.set_page_config(page_title="Monitoramento NDVI - Paraíba", layout="wide")
st.title("🌱 Previsão Espaço-Temporal de NDVI na Caatinga")
st.markdown("Plataforma interativa de Explainable AI (XAI) comparando dados reais de satélite e predições do modelo LightGBM para o estado da Paraíba.")

# 2. Carregamento e Preparação dos Dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("df_final.csv") 
    
    # Criar coluna de Data para o filtro temporal
    df['Data'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Mes'].astype(str) + '-01')
    
    # Criar divisão regional aproximada baseada na Longitude da Paraíba (Leste -> Oeste)
    if 'Regiao' not in df.columns:
        condicoes = [
            (df['Longitude'] >= -35.5),
            (df['Longitude'] < -35.5) & (df['Longitude'] >= -36.5),
            (df['Longitude'] < -36.5) & (df['Longitude'] >= -37.5),
            (df['Longitude'] < -37.5)
        ]
        escolhas = ['Mata Paraibana (João Pessoa)', 'Agreste (Campina Grande)', 'Borborema (Patos)', 'Sertão (Sousa-Cajazeiras)']
        df['Regiao'] = np.select(condicoes, escolhas, default='Outros')
        
    return df

df = carregar_dados()

# 3. Barra Lateral (Filtros Temporais)
st.sidebar.header("Filtros Temporais")

tipo_filtro = st.sidebar.radio(
    "Filtrar por:", 
    ["Ano Único", "Mês Específico", "Período (Intervalo)"]
)

# Correção da ordem cronológica usando sorted()
anos_disponiveis = sorted(df['Ano'].unique())
datas_disponiveis = sorted(df['Data'].dt.strftime('%Y-%m').unique())

if tipo_filtro == "Ano Único":
    ano_selecionado = st.sidebar.selectbox("Selecione o Ano:", anos_disponiveis)
    df_filtrado = df[df['Ano'] == ano_selecionado]
    periodo_texto = f"Ano: {ano_selecionado}"
    
elif tipo_filtro == "Mês Específico":
    data_selecionada = st.sidebar.selectbox("Selecione o Mês/Ano:", datas_disponiveis)
    df_filtrado = df[df['Data'].dt.strftime('%Y-%m') == data_selecionada]
    periodo_texto = f"Mês: {data_selecionada}"
    
else: # Período (Intervalo)
    inicio, fim = st.sidebar.select_slider(
        "Selecione o Intervalo de Meses:",
        options=datas_disponiveis,
        value=(datas_disponiveis[0], datas_disponiveis[-1])
    )
    df_filtrado = df[(df['Data'].dt.strftime('%Y-%m') >= inicio) & (df['Data'].dt.strftime('%Y-%m') <= fim)]
    periodo_texto = f"Período: {inicio} a {fim}"

st.sidebar.markdown("---")
st.sidebar.info("Utilize os filtros acima para atualizar dinamicamente os mapas, gráficos e métricas.")

# 4. Renderização dos Mapas
st.header(f"Visão Espacial - {periodo_texto}")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**NDVI Real (Observado via Satélite)**")
    fig_real = px.scatter_mapbox(
        df_filtrado, lat="Latitude", lon="Longitude", color="NDVI",
        color_continuous_scale="RdYlGn", range_color=[0, 0.9],
        zoom=5.5, center={"lat": -7.115, "lon": -36.5}, mapbox_style="carto-positron"
    )
    fig_real.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_real, use_container_width=True)

with col2:
    st.markdown("**NDVI Predito (Modelo LightGBM)**")
    fig_pred = px.scatter_mapbox(
        df_filtrado, lat="Latitude", lon="Longitude", color="NDVI_Predito",
        color_continuous_scale="RdYlGn", range_color=[0, 0.9],
        zoom=5.5, center={"lat": -7.115, "lon": -36.5}, mapbox_style="carto-positron"
    )
    fig_pred.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("---")

# 5. Análise Descritiva e Séries Temporais
st.header("Análise Descritiva e Dinâmica Regional")
col_grafico, col_tabela = st.columns([2, 1])

with col_grafico:
    st.subheader("Série Temporal de Chuvas por Região")
    # Agrupando dados para o gráfico de linhas
    df_chuva_regiao = df_filtrado.groupby(['Data', 'Regiao'])['Precipitacao'].mean().reset_index()
    
    fig_chuva = px.line(
        df_chuva_regiao, x="Data", y="Precipitacao", color="Regiao",
        labels={"Precipitacao": "Precipitação Média (mm)", "Data": "Período"},
        markers=True, template="simple_white"
    )
    st.plotly_chart(fig_chuva, use_container_width=True)

with col_tabela:
    st.subheader("Resumo Estatístico")
    st.markdown("Comparativo geral das variáveis no período selecionado:")
    resumo_estatistico = df_filtrado[['NDVI', 'NDVI_Predito', 'Precipitacao', 'Temperatura_Solo']].describe().round(2).T[['mean', 'min', 'max']]
    resumo_estatistico.columns = ['Média', 'Mínimo', 'Máximo']
    st.dataframe(resumo_estatistico, use_container_width=True)

st.markdown("---")

# 6. Métricas do Modelo e Explicação Pedagógica
st.header("Desempenho do Modelo LightGBM")

if len(df_filtrado) > 0:
    mae = mean_absolute_error(df_filtrado['NDVI'], df_filtrado['NDVI_Predito'])
    rmse = np.sqrt(mean_squared_error(df_filtrado['NDVI'], df_filtrado['NDVI_Predito']))
    r2 = r2_score(df_filtrado['NDVI'], df_filtrado['NDVI_Predito'])

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label="R² (Coeficiente de Determinação)", value=f"{r2:.4f}")
    col_m2.metric(label="MAE (Erro Médio Absoluto)", value=f"{mae:.4f}")
    col_m3.metric(label="RMSE (Raiz do Erro Quadrático)", value=f"{rmse:.4f}")
    
    with st.expander("📚 Entenda o que cada métrica significa"):
        st.markdown("""
        * **R² (Coeficiente de Determinação):** Indica o quão bem o modelo explica a variabilidade do NDVI. Um R² de 0.85, por exemplo, significa que 85% das variações na vegetação (NDVI) são explicadas pelas variáveis que fornecemos (chuva, relevo, temperatura, etc). Quanto mais próximo de 1, melhor.
        * **MAE (Erro Médio Absoluto):** Representa a diferença média "real" entre a predição do modelo e o satélite. Se o MAE for 0.05, significa que, em média, o modelo erra o valor do NDVI em 0.05 pontos (para mais ou para menos). É uma métrica excelente por ser fácil de interpretar.
        * **RMSE (Raiz do Erro Quadrático Médio):** Semelhante ao MAE, mas penaliza erros grandes. Se o RMSE estiver muito maior que o MAE, significa que o modelo tem alguns "erros graves" em pontos específicos do mapa, mesmo acertando a maioria.
        """)
else:
    st.warning("Sem dados suficientes para calcular métricas neste período.")

st.markdown("---")

# 7. Exportação de Resultados
st.header("Exportar Dados")
st.markdown("Faça o download do recorte temporal e espacial selecionado em formato CSV para análises externas.")

csv_export = df_filtrado.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Baixar Dados Filtrados (CSV)",
    data=csv_export,
    file_name=f"dados_ndvi_paraiba_{periodo_texto.replace(':', '').replace(' ', '_')}.csv",
    mime="text/csv",
)