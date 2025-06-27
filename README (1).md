# avaliacao-modelos-ml
Pseudocódigo Detalhado para a Metodologia Experimental
# Adicionar imports para KFold e testes estatísticos no início do script
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon, friedmanchisquare
# Para testes post-hoc de Nemenyi, pode-se usar bibliotecas como 'scikit-posthocs' ou 'mlxtend'
# pip install scikit-posthocs
import scikit_posthocs as sp

# ... (Funções load_dataset, preprocess_data, plot_results como definidas anteriormente)

def train_and_evaluate_models_cv(X, y, problem_type='classification', n_splits=5):
    """
    Treina e avalia múltiplos modelos usando K-Fold Cross-Validation.
    Retorna os scores de cada modelo para cada fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = {}
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'MLP Classifier': MLPClassifier(random_state=42, max_iter=1000)
        }
        main_metric = 'accuracy'
        # Função lambda para calcular a métrica principal para classificação
        metric_func = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    else: # Regression
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
            'Random Forest Regressor': RandomForestRegressor(random_state=42),
            'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
            'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
            'MLP Regressor': MLPRegressor(random_state=42, max_iter=1000)
        }
        main_metric = 'RMSE'
        # Função lambda para calcular a métrica principal para regressão
        metric_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

    model_scores = {name: [] for name in models.keys()}
    
    print(f"Executando {n_splits}-Fold Cross-Validation para os modelos...")
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index] # .iloc para manter compatibilidade com pandas Series
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metric_func(y_test, y_pred)
            model_scores[name].append(score)
        print(f"  Fold {fold+1} concluído.")

    # Calcular médias e desvios padrão dos scores
    average_scores = {name: np.mean(scores) for name, scores in model_scores.items()}
    std_scores = {name: np.std(scores) for name, scores in model_scores.items()}

    print("\nResultados Médios da Validação Cruzada:")
    for name in average_scores:
        print(f"  {name}: {main_metric} = {average_scores[name]:.4f} +/- {std_scores[name]:.4f}")
            
    return model_scores, average_scores # Retorna scores por fold para testes estatísticos e médias


def perform_statistical_tests_robust(model_scores, dataset_name):
    """
    Realiza testes de significância estatística usando os scores de CV.
    """
    print(f"\n--- Testes de Significância Estatística para {dataset_name} ---")
    model_names = list(model_scores.keys())
    
    if len(model_names) < 2:
        print("Mínimo de 2 modelos necessários para testes de significância.")
        return

    # 1. Teste de Friedman (para múltiplos modelos em múltiplos folds)
    # Reorganizar scores para o teste de Friedman (cada linha é um fold, cada coluna é um modelo)
    data_for_friedman = np.array([model_scores[name] for name in model_names]).T
    
    if data_for_friedman.shape[0] < 2: # Friedman precisa de pelo menos 2 ranks, ou seja, 2 folds
        print("Poucos folds para o teste de Friedman.")
    else:
        stat, p_value_friedman = friedmanchisquare(*[list(model_scores[name]) for name in model_names])
        print(f"Teste de Friedman: Statistic={stat:.4f}, p-value={p_value_friedman:.4f}")
        if p_value_friedman < 0.05:
            print("  Há uma diferença estatisticamente significativa no desempenho geral dos modelos (p < 0.05).")
            # Nemenyi post-hoc test
            print("\nRealizando Nemenyi Post-Hoc Test:")
            # sp.posthoc_nemenyi_friedman requer a matriz de scores (colunas = modelos, linhas = folds)
            # ou uma lista de arrays, onde cada array são os scores de um modelo.
            posthoc_results = sp.posthoc_nemenyi_friedman(data_for_friedman)
            # Nomear as colunas e índices para melhor visualização
            posthoc_results.columns = model_names
            posthoc_results.index = model_names
            print(posthoc_results)
            print("P-values menores que 0.05 indicam diferença significativa entre os pares.")
        else:
            print("  Não há diferença estatisticamente significativa no desempenho geral dos modelos (p >= 0.05).")

    # 2. Teste de Wilcoxon (para comparações par a par)
    print("\nComparações Par a Par (Wilcoxon Signed-Rank Test):")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            try:
                stat, p_value_wilcoxon = wilcoxon(model_scores[model1], model_scores[model2])
                print(f"  {model1} vs {model2}: Statistic={stat:.4f}, p-value={p_value_wilcoxon:.4f}")
                if p_value_wilcoxon < 0.05:
                    print(f"    Diferença significativa (p < 0.05).")
                else:
                    print(f"    Não há diferença significativa (p >= 0.05).")
            except ValueError as e:
                print(f"  Não foi possível rodar Wilcoxon para {model1} vs {model2}: {e}")

# --- REVISÃO DA EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    datasets = {
        'Video Game Sales': {
            'filepath': 'vgsales.csv',
            'target': 'Global_Sales',
            'problem_type': 'regression'
        },
        'Games Ratings Sales Metacritic': {
            'filepath': 'Video_Game_Sales_as_of_Jan_2017.csv',
            'target': 'Critic_Score',
            'problem_type': 'regression'
        },
        'Steam Games': {
            'filepath': 'steam.csv',
            'target': 'positive_ratings', # Será convertida para classificação binária
            'problem_type': 'classification' # Mudei para classificação
        },
        'LoL Ranked Games': {
            'filepath': 'games.csv',
            'target': 'winner',
            'problem_type': 'classification'
        },
        'PUBG Finish Prediction': {
            'filepath': 'train_V2.csv',
            'target': 'winPlacePerc',
            'problem_type': 'regression'
        }
    }

    results_summary_per_dataset = {} # Para armazenar as médias dos scores por dataset
    all_raw_cv_scores = {} # Para armazenar os scores de CV crus para testes estatísticos globais

    for name, config in datasets.items():
        print(f"\n--- Processando Dataset: {name} ---")
        try:
            df = load_dataset(config['filepath'])
            df.dropna(subset=[config['target']], inplace=True) # Remover linhas com nulos no target

            if name == 'Steam Games' and config['problem_type'] == 'classification':
                # Criar a coluna target 'success' para classificação binária
                # Considerar sucesso se positive_ratings for significativamente maior que negative_ratings
                # Exemplo: ratio de 2 para 1
                df['success'] = (df['positive_ratings'] > 2 * df['negative_ratings']).astype(int)
                # ou simplesmente df['success'] = (df['positive_ratings'] > df['negative_ratings']).astype(int)
                config['target'] = 'success'
                # Remover as colunas originais de ratings se não forem features
                df = df.drop(columns=['positive_ratings', 'negative_ratings'], errors='ignore')

            # Pré-processar dados e obter os conjuntos X, y
            X_processed, _, y_processed, _, preprocessor = preprocess_data(df.copy(), config['target'], config['problem_type'])
            
            # Garantir que X_processed seja denso se ColumnTransformer produzir esparso
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            
            # Executar validação cruzada e obter scores por fold e médias
            current_model_cv_scores, current_average_scores = train_and_evaluate_models_cv(
                X_processed, y_processed, config['problem_type']
            )
            
            results_summary_per_dataset[name] = current_average_scores
            all_raw_cv_scores[name] = current_model_cv_scores # Guardar scores de CV para testes estatísticos

            # Plotar resultados médios para o dataset atual
            if config['problem_type'] == 'classification':
                main_metric = 'accuracy'
            else:
                main_metric = 'RMSE' # Ou 'R2' dependendo da preferência
            
            # Convertendo os scores médios para um dicionário de dicionários para plot_results
            plot_data_for_plot = {model: {main_metric: score} for model, score in current_average_scores.items()}
            plot_results(plot_data_for_plot, config['problem_type'], main_metric)

            # Realizar testes de significância para o dataset atual
            perform_statistical_tests_robust(current_model_cv_scores, name)

        except FileNotFoundError:
            print(f"Erro: Arquivo '{config['filepath']}' não encontrado. Por favor, baixe o dataset do Kaggle e coloque-o na pasta correta.")
        except Exception as e:
            print(f"Ocorreu um erro ao processar o dataset {name}: {e}")

    print("\n--- Sumário de Resultados Médios por Dataset ---")
    for dataset, avg_scores in results_summary_per_dataset.items():
        print(f"\nDataset: {dataset}")
        for model, score in avg_scores.items():
            print(f"  {model}: {score:.4f}")

    # No final, pode-se também tentar uma análise estatística global se os datasets forem comparáveis
    # Isso é mais avançado e pode ser uma 'feature' extra se o tempo permitir.
    # Por exemplo, Friedman test across datasets if algorithms are ranked per dataset.