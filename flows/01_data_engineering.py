"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    SILVER,
    SILVER_FINAL_NAME,
    SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]

SILVER_NOT_NULL = [
    "Survived", "Pclass", "Age", "Fare", "Embarked", "Person", "Alone",
    "TCabin", "TTitulo", "NSex", "NEmbarked", "NCabin", "NTitulo",
]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Contrato de entrada: columnas esperadas presentes y PassengerId único."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Bronze no cumple el contrato; faltan columnas: {missing}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("Bronze no cumple el contrato: hay PassengerId duplicados")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Survived, Pclass, SibSp, Parch -> int; Age y Fare -> float."""
    df = df.copy()
    df["Survived"] = df["Survived"].astype(int)
    df["Pclass"] = df["Pclass"].astype(int)
    df["SibSp"] = df["SibSp"].astype(int)
    df["Parch"] = df["Parch"].astype(int)
    df["Age"] = df["Age"].astype(float)
    df["Fare"] = df["Fare"].astype(float)
    return df


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex.

    Regla para edad desconocida (NaN): se usa Sex. Esto subestima a los niños
    sin edad registrada; el sesgo se cuantifica en el Reto 8.1.
    """
    df = df.copy()
    df["Person"] = np.where(df["Age"] < 16, "child", df["Sex"])
    return df


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
    df = df.copy()
    df["TCabin"] = df["Cabin"].str[0].fillna("X")
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0, 'Alone' si no)."""
    df = df.copy()
    df["Familia"] = df["SibSp"] + df["Parch"]
    df["Alone"] = np.where(df["Familia"] > 0, "with Family", "Alone")
    return df


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes' (solo para etiquetar gráficos)."""
    df = df.copy()
    df["Survivor"] = df["Survived"].map({0: "No", 1: "Yes"})
    return df


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título del nombre ('Braund, Mr. Owen Harris' -> 'Mr') en 'TTitulo'."""
    df = df.copy()
    df["TTitulo"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return df


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa títulos raros con TITLE_MAP; todo lo que no es principal pasa a 'Otro'."""
    df = df.copy()
    df["TTitulo"] = df["TTitulo"].replace(TITLE_MAP)
    df["TTitulo"] = df["TTitulo"].where(df["TTitulo"].isin(MAIN_TITLES), "Otro")
    return df


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena 'Embarked' con la moda (el puerto más frecuente)."""
    df = df.copy()
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 'Age' con el promedio por 'Sex'.

    Conserva la edad original en 'AgeOriginal' (con NaN) y marca las filas
    imputadas en 'AgeImputada' (bool). Usada para EXPLORAR; el modelo (Parte 11)
    vuelve a imputar solo con datos de entrenamiento.
    """
    df = df.copy()
    df["AgeOriginal"] = df["Age"]
    df["AgeImputada"] = df["Age"].isna()
    std_antes = df["Age"].std()
    promedios = df.groupby("Sex")["Age"].mean()
    df["Age"] = df["Age"].fillna(df["Sex"].map(promedios))
    log.info("Desviación estándar de Age antes=%.2f después=%.2f (la imputación la reduce)",
             std_antes, df["Age"].std())
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Versiones numéricas de las categóricas, con mapas FIJOS (reproducibles)."""
    df = df.copy()
    df["NSex"] = df["Sex"].map({"female": 0, "male": 1})
    df["NEmbarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["NCabin"] = df["TCabin"].map({letra: i for i, letra in enumerate(DECKS)})
    df["NTitulo"] = df["TTitulo"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Otro": 3})
    return df


def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una COPIA sin Name, Sex, Embarked y Cabin."""
    return df.drop(columns=["Name", "Sex", "Embarked", "Cabin"]).copy()


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Contrato de salida: sin nulos en las columnas del modelo (AgeOriginal sí puede)."""
    nulos = {col: int(df[col].isnull().sum()) for col in SILVER_NOT_NULL if df[col].isnull().any()}
    if nulos:
        raise ValueError(f"Silver no cumple el contrato; columnas con nulos: {nulos}")
    return df


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir=BRONZE, out_dir=SILVER):
    bronze = pd.read_csv(BRONZE / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME
