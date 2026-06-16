#===============================================
#             CEFET-MG - PPGMMC                 
#===============================================
#      Métodos Matemáticos Computacionais       
# Aluno: Jeancarlo Campos Leão
#-----------------------------------------------
# DATA: 17/06/2026.
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# Config
st.set_page_config(page_title="MMQ LAB", layout="wide")

st.title("Laboratório de Regressões Comparativas")

# DADOS
st.sidebar.header("Entrada de dados")

modo = st.sidebar.radio("Modo", [
    "Exemplo 1",
    "Exemplo 2",
    "Gerador sintético",
    "CSV"
])

def dataset(tipo):
    if tipo == "Linear":
        x = np.linspace(0, 10, 30)
        y = 2*x + np.random.normal(0, 2, len(x))
    elif tipo == "Quadrático":
        x = np.linspace(-5, 5, 30)
        y = x**2 + np.random.normal(0, 5, len(x))
    else:
        x = np.linspace(0, 5, 30)
        y = np.exp(x/2) + np.random.normal(0, 2, len(x))
    return x, y

x, y = [], []

if modo == "Exemplo 1":
    #ex = st.sidebar.selectbox("Dataset", ["Linear", "Quadrático", "Exponencial"])
    #x, y = dataset(ex)
    x = [0, 2, 4, 6]
    y = [5.3, 7, 9.4, 12.3]
elif modo == "Exemplo 2":
    x = [0, 0.25, 0.5, 0.75, 1]
    y = [1, 1.284, 1.6487, 2.117, 2.7183]
elif modo == "Gerador sintético":
    n = st.sidebar.slider("Pontos", 10, 1000, 30)
    tipo = st.sidebar.selectbox("Tipo", ["Linear", "Quadrático"])
    outlier_frac = st.sidebar.slider("Fração de outliers", 0.0, 0.3, 0.1)
    n_outliers = int(n * outlier_frac)
    outlier_strength = st.sidebar.slider("Força dos outliers", 5.0, 50.0, 15.0)

    x = np.linspace(-10, 10, n)
    noise = st.sidebar.slider("Ruído", 0.0, 10.0, 2.0)
    if tipo == "Linear":
        a = st.sidebar.number_input("a", value=2.0)
        b = st.sidebar.number_input("b", value=0.0)
        y = a*x + b + np.random.normal(0, noise, n)
    else:
        a = st.sidebar.number_input("a", value=1.0)
        y = a*x**2 + np.random.normal(0, noise, n)
    # cria outliers
    idx = np.random.choice(n, n_outliers, replace=False)
    y[idx] += np.random.normal(0, noise * outlier_strength, n_outliers)
elif modo == "CSV":
    file = st.sidebar.file_uploader("CSV (x,y)", type="csv")
    if file:
        df = pd.read_csv(file)
        x = df.iloc[:,0].values
        y = df.iloc[:,1].values

x = np.array(x)
y = np.array(y)

df = pd.DataFrame({"x": x, "y": y})
def aplicar_linearizacao(x, y, tipo):
    if tipo == "Log(y)":
        return x, np.log(y)
    elif tipo == "Log(x)":
        return np.log(x), y
    elif tipo == "Log-log":
        return np.log(x), np.log(y)
    elif tipo == "Recíproca (1/x)":
        return 1/x, y
    elif tipo == "Exponencial (y -> log)":
        return x, np.log(y)
    else:
        return x, y
# Linearização:
show_linearizacao = st.sidebar.checkbox("Linearização (transformações)", False)

tipo_linearizacao = st.sidebar.selectbox(
    "Tipo de linearização",
    [
        "Nenhuma",
        "Log(y)",
        "Log(x)",
        "Log-log",
        "Recíproca (1/x)",
        "Exponencial (y -> log)"
    ]
)    
x_lin, y_lin = aplicar_linearizacao(x, y, tipo_linearizacao)

df_lin = pd.DataFrame({
    "x": x_lin,
    "y": y_lin
})
# TOGGLES DOS MODELOS
st.sidebar.header("Modelos")

show_ols = st.sidebar.checkbox("MMQ (OLS)", True)
show_poly = st.sidebar.checkbox("Polinomial", True)
show_minmax = st.sidebar.checkbox("MinMax + OLS", True)
show_minimax = st.sidebar.checkbox("Minimax", True)
show_l1 = st.sidebar.checkbox("Regressão L1 (valor absoluto)", True)

grau = st.sidebar.slider("Grau polinomial", 1, 5, 2)

try:
    # MODELO OLS (LINEAR)
    model_ols = smf.ols("y ~ x", data=df).fit()
    y_ols = model_ols.predict(df)

    # MODELO POLINOMIAL (NUMPY)
    coef = np.polyfit(x, y, grau)
    p = np.poly1d(coef)
    y_poly = p(x)

    # MINMAX + OLS
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x.reshape(-1, 1)).flatten()

    df_minmax = pd.DataFrame({
        "x_scaled": x_scaled,
        "y": y
    })

    model_minmax = smf.ols("y ~ x_scaled", data=df_minmax).fit()
    y_minmax = model_minmax.predict(df_minmax)

    # REGRESSÃO L1 (VALOR ABSOLUTO)
    def lad_loss(params):
        a, b = params
        return np.sum(np.abs(y - (a*x + b)))

    res = minimize(lad_loss, x0=[1, 0])
    a_l1, b_l1 = res.x
    y_l1 = a_l1*x + b_l1

    # MINIMAX - Minimiza o maior erro absoluto
    def minimax_loss(params):
        a, b = params

        erros = np.abs(
            y - (a*x + b)
        )

        return np.max(erros)

    res_minimax = minimize(
        minimax_loss,
        x0=[1, 0],
        method="Nelder-Mead"
    )

    a_minimax, b_minimax = res_minimax.x

    y_minimax = a_minimax*x + b_minimax

    # MÉTRICAS
    def metrics(y_real, y_pred):
        mse = np.mean((y_real - y_pred)**2)
        ss_res = np.sum((y_real - y_pred)**2)
        ss_tot = np.sum((y_real - np.mean(y_real))**2)
        r2 = 1 - ss_res/ss_tot
        #return mse, r2
        mae = np.mean(np.abs(y_real - y_pred))

        max_error = np.max(np.abs(y_real - y_pred))

        return mse, mae, max_error, r2
    mse_ols, mae_ols, max_ols, r2_ols = metrics(y, y_ols)
    mse_poly, mae_poly, max_poly, r2_poly = metrics(y, y_poly)
    mse_minmax, mae_minmax, max_minmax, r2_minmax = metrics(y, y_minmax)
    #mse_l1, r2_l1
    mse_l1, mae_l1, max_l1, r2_l1 = metrics(y, y_l1)
    mse_minimax, mae_minimax, max_minimax, r2_minimax = metrics(y, y_minimax)

    #mse_ols, r2_ols = metrics(y, y_ols) 
    #mse_poly, r2_poly = metrics(y, y_poly)
    #mse_minmax, r2_minmax = metrics(y, y_minmax)
    #mse_l1, r2_l1 = metrics(y, y_l1)

    # OVERFITTING
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    p_train = np.poly1d(np.polyfit(x_train, y_train, grau))

    train_err = mean_squared_error(y_train, p_train(x_train))
    test_err = mean_squared_error(y_test, p_train(x_test))

    overfit = test_err > train_err * 1.5

    n_amostras = len(x)
    minimo_recomendado = 30 * (grau + 1)

    proporcao = n_amostras / minimo_recomendado    
    #proporcao = len(x) / (grau + 1)  # Proporção entre número de pontos e parâmetros do modelo
    if proporcao < 0.5:
        overfitmsg = "🔴 Confiabilidade muito baixa: poucos dados para avaliar overfitting."
    elif proporcao < 1.0:
        overfitmsg = "🟠 Confiabilidade baixa: resultado pode variar significativamente."
    elif proporcao < 2.0:
        overfitmsg = "🟡 Confiabilidade moderada: avaliação razoavelmente estável."
    elif proporcao < 5.0:
        overfitmsg = "🟢 Confiabilidade alta: quantidade de dados adequada."
    else:
        overfitmsg = "✅ Confiabilidade muito alta: avaliação estatisticamente robusta."
    # GRÁFICO
    fig = go.Figure()
    blackwhite = st.sidebar.checkbox("Gráfico preto e branco", False)
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h"),
    )
    opacity = 0.85 - 0.5 * (n_amostras - 10) / 990
    if blackwhite:
        marker=dict(symbol="circle", size=6*opacity + 2, color="black", opacity=opacity)
        ols_style=dict(color="black", width=2, dash="dash")
        poly_style=dict(color="black", width=2, dash="solid")
        minmax_style=dict(color="black", width=2, dash="dot")
        l1_style=dict(color="black", width=2, dash="dashdot")
        minimax_style=dict(color="black", width=3, dash="longdash")
        minimax_marker=dict(symbol="x", size=6, color="black")
    else:       
        marker=dict(symbol="circle", size=6*opacity+2, color="blue", opacity=opacity)
        ols_style=dict(color="blue", width=2, dash="dash")
        poly_style=dict(color="red", width=2, dash="solid")
        minmax_style=dict(color="green", width=2, dash="dot")
        l1_style=dict(color="gray", width=2, dash="dashdot")
        minimax_style=dict(color="orange", width=3, dash="longdash")
        minimax_marker=dict(symbol="x", size=6, color="black")
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="",marker=marker))

    if show_ols:
        fig.add_trace(go.Scatter(x=x, y=y_ols, mode="lines", name="OLS (MMQ)", line=ols_style))

    if show_poly:
        fig.add_trace(go.Scatter(x=x, y=y_poly, mode="lines", name=f"Polinomial (grau {grau})", line=poly_style))

    if show_minmax:
        fig.add_trace(go.Scatter(x=x, y=y_minmax, mode="lines", name="MinMax + OLS", line=minmax_style))

    if show_l1:
        fig.add_trace(go.Scatter(x=x, y=y_l1, mode="lines", name="L1 (valor absoluto)", line=l1_style))

    if show_minimax:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_minimax,
                mode="lines",
                name="Minimax",
                line=minimax_style,
                marker=minimax_marker
            )
        )
    st.plotly_chart(fig, use_container_width=True)

    # ANÁLISE
    st.subheader("Resultados")
    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.write("### OLS")
            st.write(f"MSE: {mse_ols:.6f}")
            st.write(f"R²: {r2_ols:.6f}")
        with col2:
            st.write("### L1")
            st.write(f"MSE: {mse_l1:.6f}")
            st.write(f"R²: {r2_l1:.6f}")
        with col3:
            st.write("### Polinomial")
            st.write(f"MSE: {mse_poly:.6f}")
            st.write(f"R²: {r2_poly:.6f}")
        with col4:
            st.write("### MinMax+OLS")
            st.write(f"MSE: {mse_minmax:.6f}")
            st.write(f"R²: {r2_minmax:.6f}")
        with col5:
            st.write("### Minimax")
            st.write(f"MSE: {mse_minimax:.6f}")
            st.write(f"R²: {r2_minimax:.6f}")

    # INTERPRETAÇÃO
    st.subheader("Interpretação")
    ranking = [
        ("OLS", r2_ols),
        ("Polinomial", r2_poly),
        ("MinMax", r2_minmax),
        ("L1", r2_l1)
    ]
    melhor = max(ranking, key=lambda x: x[1])#[0]
    htmlMethods = f"""
    <ul>
        <li><strong>L1</strong> tende a ser mais robusto a outliers</li>
        <li><strong>MinMax</strong> altera escala e estabilidade numérica</li>
        <li><strong>OLS</strong> minimiza erro quadrático (sensível a outliers)</li>
    </ul>
    """
    htmlMetrics = f"""
<ul>
    <li>
        <strong>MSE</strong>
        (<i>Mean Squared Error, ou Erro Quadrático Médio</i>)
        mede a média dos quadrados das diferenças entre os valores reais e os valores previstos pelo modelo. Ao elevar os erros ao quadrado, penaliza mais fortemente erros grandes do que erros pequenos.
        <br>
        <strong>Como interpretar:</strong> Quanto menor o valor, melhor o desempenho do modelo.
        <br>
        <strong>Vantagem:</strong> É especialmente útil quando grandes erros são mais problemáticos e devem receber maior penalização.
    </li>
    <li>
        <strong>R²</strong>
        (<i>Coeficiente de Determinação</i>)
        indica a proporção da variabilidade da variável dependente que é explicada pelo modelo.
        <br>
        <strong>Como interpretar:</strong> Geralmente varia de 0 a 1, embora possa assumir valores negativos. Quanto mais próximo de 1, maior a capacidade explicativa do modelo. Um valor 0 indica que o modelo não explica mais variabilidade do que a simples previsão da média, enquanto valores negativos indicam desempenho pior que essa previsão.
        <br>
        <strong>Vantagem:</strong> Fornece uma medida intuitiva da capacidade do modelo em explicar a variabilidade dos dados.
        <br>
        <strong>Observação:</strong> Um R² elevado não garante que o modelo seja adequado nem implica causalidade.
    </li>
</ul>"""    
    texto = f"""
    📌 Melhor modelo: **{melhor[0]}**

    ⚠ Overfitting: {"Sim" if overfit else "Não"}

    📉 L1 tende a ser mais robusto a outliers  
    📏 MinMax altera escala e estabilidade numérica  
    📊 MMQ/OLS (Mínimos Quadrados Ordinários) minimiza erro quadrático (sensível a outliers)
    """
    #st.write(texto)
    #st.markdown(html, unsafe_allow_html=True)
    crossvalidation_msg = "O conjunto de dados é dividido em K partes iguais (chamadas folds).\nO modelo é treinado usando K-1 partes e testado na parte restante.\nEsse processo se repete até que todas as partes tenham sido usadas como teste.\nO erro final é calculado tirando a média das métricas de erro (como MSE - Erro Quadrático Médio ou RMSE - Raiz do Erro Quadrático Médio) de cada rodada."
    with st.container(border=True):
        gui_methods, gui_metrics = st.columns(2)
        with gui_methods:
            st.subheader("Métodos")
            st.markdown(htmlMethods, unsafe_allow_html=True)
        with gui_metrics:#st.container(border=True):
            st.subheader("Métricas")
            st.markdown(htmlMetrics, unsafe_allow_html=True)
        st.table({
            "Item": ["Melhor modelo", "R²", "Overfitting¹"],
            "Valor": [melhor[0], f"{melhor[1]:.6f}", "Sim" if overfit else "Não"]
        })
        st.markdown(
            f"<a href='#overfitting' title='{crossvalidation_msg}'>¹ {overfitmsg}</a>",
            unsafe_allow_html=True
        )        
        st.write("⚠️ Lembre-se: a escolha do melhor modelo depende do contexto, dos objetivos da análise e da natureza dos dados. Considere sempre as métricas em conjunto e avalie a adequação do modelo para o seu caso específico.")
    #with gui_methods:
    #    st.metric("Métodos", html)

    #with gui_metrics:
    #    st.metric("Métricas", texto)

    #col1, col2 = st.columns(2)

    #with col1:
    #    st.metric("Melhor modelo", melhor)

    #with col2:
    #    st.metric("Overfitting", "⚠ Sim" if overfit else "📌 Não")
except Exception as e:
    # Mostrar exceção no Streamlit para depuração
    #st.exception(e)
    st.error("Requer que carregue os dados")
