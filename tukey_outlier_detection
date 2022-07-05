import numpy as np
from collections import Counter


def detect_outliers(df, features, multiplier=2.2):
    """
    Toma un dataframe y una lista con nombres de columnas (features) y devuelve una lista de indices que corresponde a la posición de los outliers según el metodo de 
    “Exploratory Data Analysis” de John W. Tukey (1977), Se propone como multiplier 1.5 para outliers sauves y 3 para outliers
    fuertes, por defecto se usa 2.2 basado en “Fine-Tuning Some Resistant Rules for Outlier Labeling” de Hoaglin y
    Iglewicz (1987).
    """
    outlier_indices = []
    # Iteramos sobre los nombres de las columnas(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        outlier_step = multiplier * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)        
    return outlier_indices
