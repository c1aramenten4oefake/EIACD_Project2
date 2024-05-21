import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_table, read_csv
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import sklearn.tree as tree
from imblearn.over_sampling import SMOTE

# Configuração do pandas para desativar avisos de casting silencioso no futuro
pd.set_option('future.no_silent_downcasting', True)

def outiline(df):
    """
    Função para detectar e substituir outliers em um DataFrame.
    
    Parâmetros:
    df (DataFrame): DataFrame a ser processado.
    
    Retorna:
    DataFrame: DataFrame com outliers substituídos pelos percentis 10 e 90.
    """
    total = 0
    # Loop através de cada coluna do DataFrame
    for column in df.columns:
        # Verifica se a coluna é numérica
        if pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            outliers = df[(df[column] < limite_inferior) | (df[column] > limite_superior)][column]
            p10 = df[column].quantile(0.10)
            p90 = df[column].quantile(0.90)
            df[column] = df[column].apply(lambda x: p10 if x < limite_inferior else (p90 if x > limite_superior else x))
            total += len(outliers.tolist())
    print(total)
    return df

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, show_data):
    """
    Função para avaliar um modelo de machine learning e exibir a matriz de confusão.
    
    Parâmetros:
    model (sklearn model): Modelo a ser avaliado.
    X_train (DataFrame): Conjunto de treinamento.
    X_test (DataFrame): Conjunto de teste.
    y_train (Series): Labels do conjunto de treinamento.
    y_test (Series): Labels do conjunto de teste.
    model_name (str): Nome do modelo.
    show_data (int): Flag para exibir métricas e matriz de confusão.
    
    Retorna:
    list: Lista contendo métricas de avaliação (accuracy, roc_auc, precision, recall, f1).
    """
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
    if show_data == 1:
        print(f"Métricas do {model}:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"ROC AUC: {roc_auc:.2f}")
        print(f"Precisão: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Pontuação F1: {f1:.2f}")
        cm = confusion_matrix(y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dies", "Lives"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'{model_name} Confusion Matrix')
        plt.show()

    return [accuracy, roc_auc, precision, recall, f1]

def execute_parameters(na_solution, smote, outliner, tt_split, k_for_cleaning,
                        clean_data_correlation, dstree, nodes_for_tree, random_frst, kapann, k_for_knn, graphs, shwdata, seed):
    """
    Função para executar várias configurações de parâmetros no modelo e avaliar o desempenho.
    
    Parâmetros:
    na_solution (str): Método de substituição de valores ausentes ('median' ou 'mean').
    smote (int): Flag para aplicar SMOTE para balanceamento de classes.
    outliner (int): Flag para detectar e substituir outliers.
    tt_split (float): Proporção de divisão entre treino e teste.
    k_for_cleaning (int): Número de atributos mais correlacionados a serem mantidos.
    clean_data_correlation (bool): Flag para limpar dados com base em correlação.
    dstree (int): Flag para habilitar o uso de árvore de decisão.
    nodes_for_tree (int): Número máximo de nós na árvore de decisão.
    random_frst (int): Flag para habilitar o uso de Random Forest.
    kapann (int): Flag para habilitar o uso de KNN.
    k_for_knn (int): Número de vizinhos para KNN.
    graphs (int): Flag para exibir gráficos.
    shwdata (int): Flag para exibir dados de avaliação.
    seed (int): Semente para geração de números aleatórios.
    
    Retorna:
    list: Lista de resultados das avaliações dos modelos.
    """
    # Carregar e preparar os dados
    df = read_csv("hcc_dataset.csv", sep=",", na_values='?')
    if outliner == 1:
        df = outiline(df)
    if na_solution == 'median':
        numerical_attributes = df.select_dtypes(include=['int64', 'float64'])
        numerical_attributes.fillna(numerical_attributes.median(), inplace=True)
        numerical_attributes.replace('?', numerical_attributes.median(), inplace=True)
        for column in numerical_attributes.columns:
            df[column] = numerical_attributes[column]
    elif na_solution == 'mean':
        numerical_attributes = df.select_dtypes(include=['int64', 'float64'])
        numerical_attributes.fillna(numerical_attributes.mean(), inplace=True)
        numerical_attributes.replace('?', numerical_attributes.mean(), inplace=True)
        for column in numerical_attributes.columns:
            df[column] = numerical_attributes[column]
    df.fillna(df.mode().iloc[0], inplace=True)
    df.replace('?', df.mode().iloc[0], inplace=True)
    
    # Substituir valores de string por valores numéricos
    df.replace('Yes', 1, inplace=True)
    df.replace('No', 0, inplace=True)
    
    x = df
    if "Class" in x.columns:
        x = x.drop(columns=["Class"])
    string_columns = x.select_dtypes(include=['object']).columns
    x = pd.get_dummies(x, columns=string_columns)
    x = x.astype(int)
    scaler = MinMaxScaler()
    cx = scaler.fit_transform(x)
    x = pd.DataFrame(cx, columns=x.columns)
    y = df[["Class",]]
    
    if smote == 1:
        smote = SMOTE(random_state=seed)
        x, y = smote.fit_resample(x, y)

    # Plotar a correlação dos atributos
    if graphs == 1:
        plt.figure(figsize=(15,10))
        sns.heatmap(x.corr(), annot=False, square=True, cmap='coolwarm')
        plt.show()
    
    if clean_data_correlation == True:
        selector = SelectKBest(score_func=f_classif, k=k_for_cleaning)  
        x_new = selector.fit_transform(x, y.values.ravel())
        selected_features = x.columns[selector.get_support(indices=True)]
        x = pd.DataFrame(x_new, columns=selected_features)
        x = pd.concat([x, y], axis=1)
        x.replace('Lives', 1, inplace=True)
        x.replace('Dies', 0, inplace=True)
        corr_matrix = x.corr()
        if graphs == 1:
            plt.figure(figsize=(15,10))
            sns.heatmap(x.corr(), annot=True, square=True, cmap='coolwarm')
            plt.show()
        x = x.drop(columns='Class')
        corr_matrix = x.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        threshold = 0.81
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        x = x.drop(columns=to_drop)
        threshold = -0.81
        to_drop = [column for column in upper.columns if any(upper[column] < threshold)]
        x = x.drop(columns=to_drop)
        x = pd.concat([x, y], axis=1)
        x.replace('Lives', 1, inplace=True)
        x.replace('Dies', 0, inplace=True)
        if graphs == 1:
            plt.figure(figsize=(15,10))
            sns.heatmap(x.corr(), annot=True, square=True, cmap='coolwarm')
            plt.show()
        if graphs == 1:
            for column in x.columns.drop('Class'):
                plt.figure(figsize=(8, 5))
                sns.histplot(data=x, x=column, hue='Class', element="bars", bins=20, kde=True)
                plt.title(f'Histogram of {column} by Class')
                plt.show()
        x = x.drop(columns='Class')

    y = y.values.ravel()
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=tt_split, random_state=seed, stratify=y)

    tresults = []
    if dstree == 1:
        # Árvore de decisão
        arv = tree.DecisionTreeClassifier(max_leaf_nodes=nodes_for_tree, criterion="gini", random_state=seed)
        tresults.append(evaluate_model(arv, X_train, X_test, y_train, y_test, "Decision Tree", shwdata))
        if shwdata == 1:
            plt.figure(figsize=(8,8))
            tree.plot_tree(arv, feature_names=x.columns.tolist(), class_names=["lives", "dies"], filled=True)
            plt.show()

    if kapann == 1:
        # KNN com k vizinhos
        knn = KNeighborsClassifier(n_neighbors=k_for_knn, weights='distance')
        tresults.append(evaluate_model(knn, X_train, X_test, y_train, y_test, "KNN", shwdata))

    if random_frst == 1:
        # Random Forest
        rf = RandomForestClassifier(n_estimators=1000, random_state=seed)
        tresults.append(evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest", shwdata))
    
    return tresults

def render(results, parameter, ax):
    """
    Função para renderizar gráficos dos resultados das avaliações dos modelos.
    
    Parâmetros:
    results (list): Lista de resultados das avaliações dos modelos.
    parameter (list): Lista de parâmetros utilizados.
    ax (matplotlib Axes): Objeto Axes para plotar o gráfico.
    """
    metrics = ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1 Score']
    algorithms = ['DecisionTree', 'Knn', 'Forest']
    results_df = pd.DataFrame(results, columns=metrics, index=algorithms)
    melted_df = results_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
    melted_df.rename(columns={'index': 'Algorithm'}, inplace=True)
    sns.barplot(data=melted_df, x='Metric', y='Value', hue='Algorithm', palette='Set1', ax=ax)
    ax.set_title(f"Parameters: {parameter}")
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.legend(title='Algorithm', loc='lower right')
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 0),  
                   textcoords = 'offset points')

def main(graphrender): 
    """
    Função principal para executar os parâmetros e renderizar os gráficos.
    
    Parâmetros:
    graphrender (int): Flag para habilitar gráficos para comparação entre algoritmos e parâmetros.
    """
    results = execute_parameters('mean', 1, 1, 0.3, 15, True, 1, 3, 1, 1, 3, 1, 1, 123)
    if graphrender == 1:
        results0 = results
        parameter0 = ['mean', 1, 1, 0.3, 15, True, 1, 3, 1, 1, 3, 1, 1, 123]
    results = execute_parameters('median', 1, 0, 0.3, 15, True, 1, 3, 1, 1, 3, 1, 1, 123)    
    if graphrender == 1:
        results1 = results
        parameter1 = ['median', 1, 0, 0.3, 15, True, 1, 3, 1, 1, 3, 1, 1, 123]
    if graphrender == 1:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        render(results0, parameter0, axs[0])
        render(results1, parameter1, axs[1])
        plt.tight_layout()
        plt.show()

# graphrender = 1 habilita os gráficos para comparação entre algoritmos e parâmetros
main(graphrender=1)

