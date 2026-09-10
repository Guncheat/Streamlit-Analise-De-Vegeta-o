import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Configuração da Página
st.set_page_config(page_title="Monitoramento NDVI - Paraíba",page_icon="🌱", layout="wide")
st.title("🌱 Previsão Espaço-Temporal de NDVI na Caatinga")
st.markdown("Compararativo dados reais de satélite e predições do modelo de Aprendizado de Máquina para o estado da Paraíba.")
st.markdown("---")
with st.expander("**Sobre o Aplicativo:**"):
    st.markdown("Este aplicativo permite analisar a vegetação da Paraíba utilizando o NDVI (Índice de Vegetação por Diferença Normalizada), comparando dados reais de satélite com predições de um modelo de Aprendizado de Máquina (LightGBM).")
    with st.expander("**O que é NDVI?**"):
        st.markdown("O NDVI (Normalized Difference Vegetation Index) é um índice que mede a densidade e saúde da vegetação com base na reflexão da luz em diferentes comprimentos de onda. Ele é calculado a partir das bandas do vermelho e do infravermelho próximo, sendo amplamente utilizado em estudos ambientais, agricultura e monitoramento de ecossistemas.")
        st.markdown("O NDVI varia de -1 a 1, onde valores próximos a 1 indicam vegetação densa e saudável, valores próximos a 0 indicam áreas sem vegetação (como solo exposto ou construções), e valores negativos geralmente correspondem a água ou superfícies não vegetadas.")
with st.expander("**Instruções de Uso:**"):
    st.markdown("""
    1. Utilize a barra lateral para selecionar o período de interesse (Ano Único, Mês Específico ou Intervalo de Meses).
    2. Visualize os mapas de NDVI real e predito, comparando a vegetação observada com a estimada pelo modelo.
    3. Analise as séries temporais de precipitação por região e consulte o resumo estatístico das variáveis.
    4. Confira as métricas de desempenho do modelo (R², MAE, RMSE) e entenda seu significado.
    5. Faça o download dos dados filtrados em formato CSV para análises externas. """)
    st.markdown("---")
    st.markdown("**Observação:** Este aplicativo é uma ferramenta educacional e de análise exploratória. As predições do modelo devem ser interpretadas com cautela e não substituem análises detalhadas de campo ou estudos científicos aprofundados.")    
    st.markdown("---")
st.markdown("---")
with st.expander("Metodologia"):
    st.markdown("""Os dados utilizados para treinar o modelo de Aprendizado de Máquina (LightGBM) foram coletados a partir de imagens de satélite e registros meteorológicos históricos da Paraíba. O NDVI foi calculado a partir das bandas do vermelho e do infravermelho próximo, enquanto as variáveis ambientais, como precipitação, temperatura do solo e relevo, foram obtidas Google Earth Engine. O pré-processamento dos dados envolveu a limpeza, normalização e transformação das variáveis para garantir a qualidade e consistência dos dados utilizados no treinamento do modelo. A divisão dos dados em conjuntos de treinamento e teste permitiu avaliar o desempenho do modelo em diferentes cenários, garantindo sua robustez e capacidade de generalização. As métricas de avaliação, como R², MAE e RMSE, foram utilizadas para quantificar a precisão das predições do modelo em relação aos valores observados de NDVI. A metodologia adotada visa fornecer uma ferramenta confiável para monitoramento da vegetação na Paraíba, contribuindo para estudos ambientais e tomada de decisões informadas.""")
    st.markdown("O modelo de Aprendizado de Máquina (LightGBM) foi treinado utilizando dados históricos de NDVI, precipitação, temperatura do solo e relevo da Paraíba. O objetivo é prever o NDVI em diferentes regiões do estado com base nas variáveis ambientais disponíveis. A metodologia envolveu a coleta e pré-processamento dos dados, a divisão em conjuntos de treinamento e teste, o ajuste do modelo e a avaliação de seu desempenho por meio de métricas como R², MAE e RMSE.")
    st.markdown("---")
    st.markdown("Qualquer dúvida ou sugestão, entre em contato: igor.negreiros2@ufrpe.br")
    st.markdown("---")
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
#with st.sidebar.expander("Simule diferentes cenários"):
#    st.markdown("Você pode simular diferentes cenários de precipitação e temperatura do solo para observar como o modelo de Aprendizado de Máquina (LightGBM) prevê o NDVI em diferentes regiões da Paraíba. Ajuste os sliders abaixo para alterar os valores das variáveis ambientais e veja como isso impacta as predições do modelo.")
#    precipitacao_simulada = st.slider("Precipitação Simulada (mm)", min_value=0, max_value=300, value=100)
#    temperatura_solo_simulada = st.slider("Temperatura do Solo Simulada (°C)", min_value=15, max_value=45, value=30)
#    st.markdown("Após ajustar os valores simulados, observe as mudanças nos mapas e gráficos apresentados no aplicativo. Isso permite entender melhor a sensibilidade do modelo às variáveis ambientais e como elas influenciam a vegetação na região semiárida da Paraíba.")

# 4. Renderização dos Mapas
st.header(f"Visão Espacial - {periodo_texto}")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**NDVI Real (Observado via Satélite)**")
    fig_real = px.scatter_map(
        df_filtrado, lat="Latitude", lon="Longitude", color="NDVI",
        color_continuous_scale="RdYlGn", range_color=[0, 0.9],
        zoom=5.5, center={"lat": -7.115, "lon": -36.5}, map_style="carto-positron"
    )
    fig_real.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_real, width="content")

with col2:
    st.markdown("**NDVI Predito (Modelo LightGBM)**")
    fig_pred = px.scatter_map(
        df_filtrado, lat="Latitude", lon="Longitude", color="NDVI_Predito",
        color_continuous_scale="RdYlGn", range_color=[0, 0.9],
        zoom=5.5, center={"lat": -7.115, "lon": -36.5}, map_style="carto-positron"
    )
    fig_pred.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_pred, width="content")

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
    st.plotly_chart(fig_chuva, width="content")

with col_tabela:
    st.subheader("Resumo Estatístico")
    st.markdown("Comparativo geral das variáveis no período selecionado:")
    resumo_estatistico = df_filtrado[['NDVI', 'NDVI_Predito', 'Precipitacao', 'Temperatura_Solo']].describe().round(2).T[['mean', 'min', 'max']]
    resumo_estatistico.columns = ['Média', 'Mínimo', 'Máximo']
    st.dataframe(resumo_estatistico, width="content")

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
        * **R² (Coeficiente de Determinação):** Indica o quão bem o modelo explica a variabilidade do NDVI. Um R² de 0.71, por exemplo, significa que 71% das variações na vegetação (NDVI) são explicadas pelas variáveis que fornecemos (chuva, relevo, temperatura, etc). Quanto mais próximo de 1, melhor.
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
with st.expander("**Referências:**"):
    st.markdown("**Fonte dos Dados:** Os dados utilizados neste aplicativo foram obtidos de fontes públicas (Google Earth Engine) e confiáveis (https://developers.google.com/earth-engine), incluindo imagens de satélite e registros meteorológicos. O modelo de Aprendizado de Máquina (LightGBM) foi treinado utilizando essas informações para fornecer estimativas do NDVI em diferentes regiões da Paraíba.")
    st.markdown("**Equipe de Desenvolvimento:** Este aplicativo foi desenvolvido por Igor Barbosa Negreiros, Mestrando em Biometria e Estatística Aplicada pela Universidade Federal Rural de Pernambuco (UFRPE), sob orientação do Prof. Dr. Wilson Rosa De Oliveira Junior , com o objetivo de fornecer uma ferramenta interativa para análise da vegetação na Paraíba. Agradecemos a todos que contribuíram com sugestões e feedbacks durante o desenvolvimento.")
    st.markdown("Esse trabalho faz parte do projeto de pesquisa financiado pela CAPES (Coordenação de Aperfeiçoamento de Pessoal de Nível Superior) e visa promover a compreensão da dinâmica da vegetação na região semiárida do Nordeste brasileiro, contribuindo para estudos ambientais e de sustentabilidade.")
    st.markdown("“O presente trabalho foi realizado com apoio da Coordenação de Aperfeiçoamento de Pessoal de Nível Superior – Brasil (CAPES) – Código de Financiamento 001”.")