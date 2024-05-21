# EIACD_Project2
Trabalho 2 de EIACD sobre Data exploration and enrichment for supervised classification para The Hepatocellular Carcinoma Dataset
by Pedro Paulo Rosa Prado Basilio, Adelino Quaresma Moniz da Vera Cruz, Pedro Antonio Resende Gulart

Como Executar
Certifique-se de ter o arquivo hcc_dataset.csv no mesmo diretório que o script Python.
E execute $python3 SupervisedMain.py no terminal.

A função execute_parameters aceita os seguintes parâmetros:

na_solution (str): Método de substituição de valores ausentes ('median' ou 'mean').
smote (int): Flag para aplicar SMOTE para balanceamento de classes (1 para aplicar, 0 para não aplicar).
outliner (int): Flag para detectar e substituir outliers (1 para aplicar, 0 para não aplicar).
tt_split (float): Proporção de divisão entre treino e teste (ex: 0.3 para 30% teste, 70% treino).
k_for_cleaning (int): Número de atributos mais correlacionados a serem mantidos.
clean_data_correlation (bool): Flag para limpar dados com base em correlação (True para limpar, False para não limpar).
dstree (int): Flag para habilitar o uso de árvore de decisão (1 para habilitar, 0 para não habilitar).
nodes_for_tree (int): Número máximo de nós na árvore de decisão.
random_frst (int): Flag para habilitar o uso de Random Forest (1 para habilitar, 0 para não habilitar).
kapann (int): Flag para habilitar o uso de KNN (1 para habilitar, 0 para não habilitar).
k_for_knn (int): Número de vizinhos para KNN.
graphs (int): Flag para exibir gráficos (1 para exibir, 0 para não exibir).
shwdata (int): Flag para exibir dados de avaliação (1 para exibir, 0 para não exibir).
seed (int): Semente para geração de números aleatórios.

Observações
Certifique-se de ajustar os parâmetros de acordo com suas necessidades específicas.
Os gráficos e as avaliações são exibidos apenas se graphs ou shwdata forem definidos como 1.
Os resultados dos modelos são apresentados com métricas de avaliação como Accuracy, ROC AUC, Precision, Recall e F1 Score.
Sinta-se à vontade para modificar e experimentar diferentes configurações para obter os melhores resultados para seu caso de uso.
