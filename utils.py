import os
import numpy as np
import pandas as pd
import logging
import ast
from datetime import datetime

def setup_logging():
    """Настройка логирования в файл и консоль"""
    os.makedirs("logs", exist_ok=True)
    
    # Создаем имя файла с текущей датой и временем
    log_filename = f"logs/clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_traces(df: pd.DataFrame) -> pd.DataFrame:
    """Извлечение трасс из датафрейма"""
    logger = logging.getLogger(__name__)
    if 'trace' in df.columns:
        # Если trace — строка, преобразуем в список
        df['trace'] = df['trace'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        logger.info("Успешно извлечены трассы из данных")
        return df.copy()
    else:
        logger.error("Не найдены колонки 'trace' в данных")
        raise ValueError("Не найдены колонки 'trace' в данных")

def save_results(clusters: np.ndarray, 
                X_embedded: np.ndarray, 
                true_labels: np.ndarray, 
                output_dir: str, 
                suffix: str = "") -> None:
    """Сохранение результатов кластеризации"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"clusters_{suffix}.npy"), clusters)
    np.save(os.path.join(output_dir, f"embeddings_{suffix}.npy"), X_embedded)
    np.save(os.path.join(output_dir, f"true_labels_{suffix}.npy"), true_labels)

def create_pseudo_labels(traces_df: pd.DataFrame) -> np.ndarray:
    """Создание меток на основе действий с учетом временных характеристик"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Используем не только действия, но и временные характеристики
    features = []
    for _, row in traces_df.iterrows():
        # Базовые признаки
        trace_str = ' '.join(row['trace']) if isinstance(row['trace'], list) else row['trace']
        duration = row['trace_duration'] / 86400  # преобразуем в дни
        mean_interval = row['time_diff_mean'] / 3600 if 'time_diff_mean' in row else 0  # в часах
        
        # Добавляем врачей если есть
        doctors = ''
        if 'doctors' in row and isinstance(row['doctors'], list):
            doctors = ' '.join(set(row['doctors']))  # уникальные врачи
        
        features.append(f"{trace_str} {doctors} DUR{duration:.1f} INT{mean_interval:.1f}")
    
    # Векторизация с увеличенным размером признаков
    vectorizer = TfidfVectorizer(
        min_df=1, 
        token_pattern=r'(?u)\b\w+\b|\bDUR\d+\.\d+\b|\bINT\d+\.\d+\b',
        stop_words=None,
        max_features=1000
    )

    try:
        X = vectorizer.fit_transform(features)
        if X.shape[1] > 0:
            return KMeans(n_clusters=3, random_state=42).fit_predict(X.toarray())
    except ValueError as e:
        print(f"Ошибка при создании псевдо-меток: {e}")
    
    print("Возвращаем случайные метки из-за ошибки векторизации")
    return np.random.randint(0, 3, size=len(traces_df))