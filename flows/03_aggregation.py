"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cuatro tablas descriptivas por grupo (Paso 28)."""
    return {
        "age_by_survival": df.groupby("Survived")["Age"].describe().reset_index(),
        "survival_by_sex": df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"]).reset_index(),
        "survival_by_pclass": df.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"]).reset_index(),
        "survival_by_alone": df.groupby("Alone")["Survived"].agg(["mean", "sum", "count"]).reset_index(),
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Prueba chi-cuadrado de independencia entre Sex y Survived (Paso 29)."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p, dof, _ = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia por clase y género (Paso 34)."""
    tabla = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")
    return tabla.rename(columns=str).reset_index()


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia y tamaño de cada puerto (Paso 35.1)."""
    return df.groupby("Embarked")["Survived"].agg(["mean", "count"]).reset_index()


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Puerto de embarque estratificado por clase: separa el efecto confuso (Paso 35.2)."""
    tabla = df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean")
    return tabla.rename(columns=str).reset_index()


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Cuartiles de Fare calculados DENTRO de cada Pclass (Paso 36.1). No muta la entrada."""
    df = df.copy()
    df["FareQ"] = df.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return df


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia por cuartil de tarifa, dentro de cada clase (Paso 36.2)."""
    con_quartil = add_fare_quartile(df)
    tabla = con_quartil.pivot_table(values="Survived", index="FareQ", columns="Pclass", aggfunc="mean")
    return tabla.rename(columns=str).reset_index()


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia y tamaño de cada tamaño de familia (Paso 37)."""
    return df.groupby("Familia")["Survived"].agg(["mean", "count"]).reset_index()


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Máscara booleana: True donde s queda fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (Paso 38)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cantidad de atípicos por variable y comparación de tasas para 'Fare' (Paso 38)."""
    age_col = "AgeOriginal" if "AgeOriginal" in df.columns else "Age"
    filas = []
    for variable, serie in [("Fare", df["Fare"]), ("Age", df[age_col]), ("Familia", df["Familia"])]:
        mask = outlier_mask_iqr(serie)
        fila = {"variable": variable, "n_outliers": int(mask.sum()), "n_total": int(serie.notna().sum())}
        if variable == "Fare":
            atipicos = df.loc[mask, "Survived"]
            resto = df.loc[~mask, "Survived"]
            fila["survival_rate_outliers"] = float(atipicos.mean()) if len(atipicos) else None
            fila["survival_rate_rest"] = float(resto.mean()) if len(resto) else None
        filas.append(fila)
    return pd.DataFrame(filas)


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_sex_survival(df), indent=2))
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
