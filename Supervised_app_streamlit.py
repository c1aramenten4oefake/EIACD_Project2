import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree as tree

# Set page configuration
st.set_page_config(page_title="Exploração de Dados de Hepatocellular Carcinoma")

def outiline(dados):
    # Loop through each column in the DataFrame
    total = 0
    for column in dados.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(dados[column]):
            Q1 = dados[column].quantile(0.25)
            Q3 = dados[column].quantile(0.75)
            IQR = Q3 - Q1

            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            outliers = dados[(dados[column] < limite_inferior) | (dados[column] > limite_superior)][column]
            p10 = dados[column].quantile(0.10)
            p90 = dados[column].quantile(0.90)
            dados[column] = dados[column].apply(lambda x: p10 if x < limite_inferior else (p90 if x > limite_superior else x))
            total += len(outliers.tolist())
    print(total)
    return dados

# Function to load data from the CSV file uploaded by the user
def carregar_dados():
    arquivo = st.file_uploader("Carregar arquivo CSV", type=["csv"])
    if arquivo is not None:
        try:
            dados = pd.read_csv(arquivo)
            return dados
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")
            return None
    return None

# Function to display developer information with animation
def show_students_info():
    with st.spinner("Carregando..."):
        st.write("Grupo 17 EIACD,Turma PL3:")
        time.sleep(2)
        st.write("- Pedro Paulo Rosa Prado Basilio")
        time.sleep(1)
        st.write("- Adelino Quaresma Moniz da Vera Cruz")
        time.sleep(1)
        st.write("- Pedro Antonio Resende Goulart ")
        time.sleep(1)

# Function to render different plots based on user selection
def render_plot(dados, keey):
    if dados is not None:
        tipo_grafico = st.selectbox("Selecione o tipo de gráfico",
                                    ["Gráfico de Barras", "Gráfico de Dispersão", "Histograma",
                                     "Diagrama de Caixa (Boxplot)"], key=keey)

        with st.expander("Controles" ):
            zoom_x = st.slider(f"Zoom X ", 0.1, 2.0, 1.0, key=f"zoom_x_{keey}")
            zoom_y = st.slider(f"Zoom Y ", 0.1, 2.0, 1.0, key=f"zoom_y_{keey}")


        try:
            if tipo_grafico == "Gráfico de Barras":
                st.subheader("Gráfico de Barras")
                fig, ax = plt.subplots()
                col1, col2 = st.columns(2)
                with col1:
                    x = st.selectbox("Selecione a variável X", dados.columns, key=f"bar_x_{keey}")
                with col2:
                    y = st.selectbox("Selecione a variável Y", dados.columns, key=f"bar_y_{keey}")
                fig, ax = plt.subplots()
                sns.barplot(data=dados, x=x, y=y, palette='Set1', ax=ax)
                ax.set_xlim(ax.get_xlim()[0] * zoom_x, ax.get_xlim()[1] * zoom_x)
                ax.set_ylim(ax.get_ylim()[0] * zoom_y, ax.get_ylim()[1] * zoom_y)
                st.pyplot(fig)

            elif tipo_grafico == "Gráfico de Dispersão":
                st.subheader("Gráfico de Dispersão")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.selectbox("Selecione a variável X", dados.columns, key=f"dis_x_{keey}")
                with col2:
                    y = st.selectbox("Selecione a variável Y", dados.columns,key=f"dis_y_{keey}")
                fig, ax = plt.subplots()
                sns.scatterplot(x=dados[x], y=dados[y], ax=ax)
                ax.set_xlim(ax.get_xlim()[0] * zoom_x, ax.get_xlim()[1] * zoom_x)
                ax.set_ylim(ax.get_ylim()[0] * zoom_y, ax.get_ylim()[1] * zoom_y)
                st.pyplot(fig)

            elif tipo_grafico == "Histograma":
                st.subheader("Histograma")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.selectbox("Selecione a variável X", dados.columns,key=f"his_x_{keey}")
                with col2:
                    y = st.selectbox("Selecione a variável Y", dados.columns,key=f"his_y_{keey}")
                fig, ax = plt.subplots()
                sns.histplot(data=dados, x=x, hue=y, element="bars", bins=20, kde=True, ax=ax)
                ax.set_xlim(ax.get_xlim()[0] * zoom_x, ax.get_xlim()[1] * zoom_x)
                ax.set_ylim(ax.get_ylim()[0] * zoom_y, ax.get_ylim()[1] * zoom_y)
                st.pyplot(fig)
                
                plt.title(f'Histogram of {x} by {y}')

            elif tipo_grafico == "Diagrama de Caixa (Boxplot)":
                st.subheader("Diagrama de Caixa (Boxplot)")
                col = st.selectbox("Selecione a variável para o boxplot", dados.columns, key=f"box_{keey}")
                fig, ax = plt.subplots()
                sns.boxplot(data=dados[col], ax=ax)
                ax.set_xlim(ax.get_xlim()[0] * zoom_x, ax.get_xlim()[1] * zoom_x)
                ax.set_ylim(ax.get_ylim()[0] * zoom_y, ax.get_ylim()[1] * zoom_y)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao gerar o gráfico: {e}")

        label_encoder = LabelEncoder()
    dados_categoricos = dados.select_dtypes(include=['object'])
    for coluna in dados_categoricos.columns:
        dados[coluna] = label_encoder.fit_transform(dados[coluna])

# Function to evaluate the model
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, show_data):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    label_encoder = LabelEncoder()
    y_test = label_encoder.fit_transform(y_test)
    prediction = label_encoder.fit_transform(prediction)
    roc_auc = roc_auc_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    if show_data:
        st.write(f"Métricas do {model_name}:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"ROC AUC: {roc_auc:.2f}")
        st.write(f"Precisão: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"Pontuação F1: {f1:.2f}")
        cm = confusion_matrix(y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dies", "Lives"])
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        ax.set_title(f'{model_name} Confusion Matrix')
        st.pyplot(fig)

    return [accuracy, roc_auc, precision, recall, f1]

# Main function to configure and run the Streamlit app
def principal():
    st.title("Exploração de Dados de Hepatocellular Carcinoma")
    st.write("Este painel fornece análises exploratórias dos dados de Hepatocellular Carcinoma.")
    st.write("Por favor, carregue o arquivo CSV contendo seus dados.")

    dados = carregar_dados()

    if dados is not None:
        st.write(
            "GitHub repository: [Click here](https://github.com/c1aramenten4oefake/UP_FCUP_2023-2024_2-semestre_EIACD_Project2/blob/bd6ff9d592bea74b0e1dafa7cb4cf869d51f89e8c/testetrabalho2.py)")
        st.header("Graficos sem Pré Processamento")
        render_plot(dados,1)
        dados = outiline(dados)
        dados.fillna(dados.mode().iloc[0], inplace=True)
        dados.replace('?', dados.mode().iloc[0], inplace=True)
        dados.replace('Yes', 1, inplace=True)
        dados.replace('No', 0, inplace=True)
        x = dados
        if "Class" in x.columns:
            x = x.drop(columns=["Class"])
        string_columns = x.select_dtypes(include=['object']).columns
        x = pd.get_dummies(x, columns=string_columns)
        x = x.astype(int)
        scaler = MinMaxScaler()
        cx = scaler.fit_transform(x)
        x = pd.DataFrame(cx, columns=x.columns)
        y = dados[["Class",]]
        smote = SMOTE(random_state=123)
        x, y = smote.fit_resample(x, y)
        x = pd.concat([x, y], axis=1)
        x.replace('Lives', 1, inplace=True)
        x.replace('Dies', 0, inplace=True)
        selector = SelectKBest(score_func=f_classif, k=15)  
        x_new = selector.fit_transform(x, y.values.ravel())
        selected_features = x.columns[selector.get_support(indices=True)]
        x = pd.DataFrame(x_new, columns=selected_features)
        x = x.drop(columns='Class')
        corr_matrix = x.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        threshold = 0.81
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        x = x.drop(columns=to_drop)
        threshold = -0.81
        to_drop = [column for column in upper.columns if any(upper[column] < threshold)]
        # Drop highly correlated columns
        x = x.drop(columns=to_drop)
        x = pd.concat([x, y], axis=1)
        x.replace('Lives', 1, inplace=True)
        x.replace('Dies', 0, inplace=True)

        st.header("Graficos apos Pré Processamento")
        render_plot(x,2)
        st.header("Avaliação de Modelos")
        st.write("Nesta seção, você pode treinar e avaliar um modelo de machine learning.")
        x = x.drop(columns='Class')
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, stratify=y)

        model_name = st.selectbox("Selecione o Modelo",
                                    ["DSTree", "RandomForest", "Knn",
                                     "LogisticRegression"], key=3)
        
        
        if model_name == 'DSTree':
            model=  tree.DecisionTreeClassifier(max_leaf_nodes=5, criterion="gini", random_state=123)
        elif model_name == 'RandomForest':
            model= RandomForestClassifier(n_estimators=500, random_state=123)
        elif model_name == 'Knn':
            model= KNeighborsClassifier(n_neighbors=3, weights='distance')
        elif model_name == 'LogisticRegression':
            model= LogisticRegression() 
        show_data = st.checkbox("Mostrar métricas do modelo", value=True)

        if st.button("Avaliar Modelo"):
            evaluate_model(model, X_train, X_test, y_train, y_test, model_name, show_data)

# About page content
def show_about_page():
    st.title("Sobre o Aplicativo")
    st.write(
        """
        **Objectivo da aplicação:**
        Este é um aplicativo Demo para exploração de dados, produzido para ajudar os seus posteriores usuários a compreender, 
        analisar e extrair insights significantes de conjuntos de dados complexos para facilitar o processo de análise de dados 
        mais acessível e eficiente para uma variedade de finalidades e públicos. 
        
        **Instrução de uso:**
        - Através da página principal, faça upload do ficheiro hcc_dataset
        - Use o menu de navegação para se guiar
        - Explore as funcionalidades disponíveis nas páginas, como análise de dados, visualizações ou informações.
        """
    )

# Function for the navigation menu and rendering pages
def main():
    st.title("Informações")
    selected_page = st.sidebar.radio("Selecione a página", ["Exploração de Dados", "Sobre o Aplicativo", "Grupo de EIACD"])
    if selected_page == "Exploração de Dados":
        principal()
    elif selected_page == "Sobre o Aplicativo":
        show_about_page()
    elif selected_page == "Grupo de EIACD":
        if st.button("Mostrar informações do grupo"):
            show_students_info()

if __name__ == "__main__":
    main()
