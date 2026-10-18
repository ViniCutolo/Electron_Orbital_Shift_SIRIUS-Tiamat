import os
from typing import Literal

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    GroupShuffleSplit,
    cross_validate,
    cross_val_predict,
)
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def pre_treatment(
    table: pd.DataFrame, model_type: Literal["continuous", "classification"]
):

    X = table[
        [
            "phase",
            "depth",
            "axis",
            "magnitude",
            "magnitude_type",
            "epicenter_distance",
        ]
    ]

    if model_type == "continuous":
        y = table["score"].copy()

    elif model_type == "classification":
        y = (table["score"] > 0.75).astype(int)

    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    groups = table["event_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    division = splitter.split(X, y, groups=groups)
    train_index, test_index = next(division)

    X_train = X.iloc[train_index].copy()
    X_test = X.iloc[test_index].copy()

    y_train = y.iloc[train_index].copy()
    y_test = y.iloc[test_index].copy()

    groups_train = groups[train_index]

    numeric_list = ["depth", "magnitude", "epicenter_distance"]
    categoric_list = ["axis", "magnitude_type", "phase"]

    pre_processor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_list),
            ("categoric", OneHotEncoder(handle_unknown="ignore"), categoric_list),
        ]
    )

    return X_train, X_test, y_train, y_test, groups_train, pre_processor


def linear_error_plot(results, predictions, y_test):

    mae = -results["test_mae"]
    rmse = -results["test_rmse"]

    print("MAE por rodada:", mae)
    print("MAE médio:", mae.mean())
    print("RMSE médio:", rmse.mean())

    observed = np.asarray(y_test, dtype=float).ravel()
    predicted = np.asarray(predictions, dtype=float).ravel()

    plt.figure(figsize=(6, 6))
    plt.scatter(observed, predicted, alpha=0.6)

    limits = [
        min(observed.min(), predicted.min()),
        max(observed.max(), predicted.max()),
    ]

    plt.plot(limits, limits, "k--", label="Previsão perfeita")
    plt.xlabel("Score observado")
    plt.ylabel("Score previsto")
    plt.title("OLS — conjunto de teste")
    plt.legend()
    plt.tight_layout()
    plt.show()

    residuals = observed - predicted

    plt.figure(figsize=(7, 4))
    plt.scatter(predicted, residuals, alpha=0.6)
    plt.axhline(0, color="black", linestyle="--")

    plt.xlabel("Score previsto")
    plt.ylabel("Resíduo: observado − previsto")
    plt.title("OLS — resíduos no teste")
    plt.tight_layout()
    plt.show()


def classification_model_error(y, probabilities):

    predictions = (probabilities >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        labels=[0, 1],
        display_labels=["Não alto", "Alto"],
        cmap="Blues",
        values_format="d",
    )

    plt.title("Classificação — previsões do CV por terremoto")
    plt.tight_layout()
    plt.show()

    PrecisionRecallDisplay.from_predictions(
        y,
        probabilities,
        pos_label=1,
        plot_chance_level=True,
    )

    plt.title("Precisão–recall — CV por terremoto")
    plt.tight_layout()
    plt.show()


def model_selection(
    X_train,
    X_test,
    y_train,
    y_test,
    groups_train,
    pre_processor,
    model: Literal["linear", "l1", "l2", "elastic_net"],
):

    if model == "linear":
        ols = Pipeline(
            [
                ("preprocess", pre_processor),
                ("model", LinearRegression()),
            ]
        )
        results = cross_validate(
            ols,
            X_train,
            y_train,
            groups=groups_train,
            cv=GroupKFold(),
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            error_score="raise",
        )

        ols.fit(X_train, y_train)
        predictions = ols.predict(X_test)

        linear_error_plot(results, predictions, y_test)

    elif model in ["l1", "l2", "elastic_net"]:
        logistic = Pipeline(
            [
                ("preprocess", pre_processor),
                (
                    "model",
                    LogisticRegression(
                        n_jobs=-1,
                        penalty=model,
                        max_iter=2000,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        probabilities_cv = cross_val_predict(
            logistic,
            X_train,
            y_train,
            groups=groups_train,
            cv=LeaveOneGroupOut(),
            method="predict_proba",
        )[:, 1]

        classification_model_error(y_train, probabilities_cv)

    else:
        raise NameError(f"No model available with name {model}")


def union_tables(table, last_table):

    if last_table is None:
        return table

    return pd.concat([last_table, table], ignore_index=True)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    name_list = os.listdir("Data")
    name_list.remove("olog_data")
    name_list.remove("sismic_data")

    unified_table = None

    for i, name in enumerate(name_list):
        path = "Data/" + name + "/resultado.xlsx"
        df = pd.read_excel(path, sheet_name=-1)

        unified_table = union_tables(df, unified_table)

    unified_table = unified_table.dropna()

    X_train, X_test, y_train, y_test, groups_train, pre_processor = pre_treatment(
        unified_table, "classification"
    )

    model_selection(X_train, X_test, y_train, y_test, groups_train, pre_processor, "l1")
