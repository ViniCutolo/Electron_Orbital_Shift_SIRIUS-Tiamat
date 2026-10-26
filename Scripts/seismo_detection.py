import os
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from feature_analysis import (
    plot_feature_importance,
    run_feature_analysis,
    select_feature_candidates,
)
from final_models import fit_final_models
from model_searches import (
    make_model_searches,
    run_model_searches,
    search_plan,
)
from optuna_searches import run_optuna_searches
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    make_scorer,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_preprocessor(columns):
    categoric_list = [
        column for column in columns if column in ["axis", "magnitude_type", "phase"]
    ]

    numeric_list = [column for column in columns if column not in categoric_list]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_list,
            ),
            (
                "categoric",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categoric_list,
            ),
        ],
        remainder="drop",
    )


def make_threshold_scorer(threshold):
    def score(model, X, y):
        probabilities = model.predict_proba(X)[:, 1]
        y_array = np.asarray(y)

        scores = {"ap": average_precision_score(y, probabilities)}

        for limiar in threshold:
            tag = f"{limiar:.2f}"

            predictions = (probabilities >= limiar).astype(int)

            scores[f"precision_{tag}"] = precision_score(
                y_array, predictions, zero_division=0
            )

            scores[f"recall_{tag}"] = recall_score(
                y_array, predictions, zero_division=0
            )

            scores[f"fp_{tag}"] = int(np.sum((y_array == 0) & (predictions == 1)))

        return scores

    return score


def pre_treatment(
    table: pd.DataFrame, model_type: Literal["continuous", "classification"]
):

    if model_type == "continuous":
        n_table = table.dropna(subset=["score"]).copy()
        y = n_table["score"].copy()

    elif model_type == "classification":
        n_table = table.dropna(subset=["p_value"]).copy()
        y = (n_table["p_value"] <= 0.1).astype(int)

    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    X = n_table[
        [
            "phase",
            "depth",
            "axis",
            "takeoff_angle",
            "incident_angle",
            "ray_param",
            "magnitude",
            "magnitude_type",
            "epicenter_distance",
            "log_depth",
            "log_dist",
            "axis_noise",
            "expected_amp",
        ]
    ]

    groups = n_table["event_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    division = splitter.split(X, y, groups=groups)
    train_index, test_index = next(division)

    X_train = X.iloc[train_index].copy()
    X_test = X.iloc[test_index].copy()

    y_train = y.iloc[train_index].copy()
    y_test = y.iloc[test_index].copy()

    groups_train = groups[train_index]

    pre_processor = make_preprocessor(
        [
            "phase",
            "depth",
            "axis",
            "takeoff_angle",
            "incident_angle",
            "ray_param",
            "magnitude",
            "magnitude_type",
            "epicenter_distance",
            "log_depth",
            "log_dist",
            "axis_noise",
            "expected_amp",
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

    r2 = r2_score(y_test, predicted)

    print(f"R²: {r2}")

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


def classification_model_error(models, splits, X_train, y_train, limiar=0.5):

    # scoring = {
    #     "ap": "average_precision",
    #     "precision": make_scorer(precision_score, zero_division=0),
    #     "recall": make_scorer(recall_score, zero_division=0),
    # }

    thresholds = np.round(np.arange(0.20, 0.91, 0.05), 2)
    print(thresholds)
    scoring = make_threshold_scorer(thresholds)

    results_cv = {}
    prediction_cv = {}
    summary_rows = []

    for name, model in models.items():
        results = cross_validate(
            model,
            X_train,
            y_train,
            cv=splits,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            error_score="raise",
        )

        results_cv[name] = results

        predictions = np.empty(len(y_train), dtype=int)

        for fitted_model, (_, val_idx) in zip(results["estimator"], splits):
            # predictions[val_idx] = fitted_model.predict(X_train.iloc[val_idx])
            probabilities = fitted_model.predict_proba(X_train.iloc[val_idx])[:, 1]

            predictions[val_idx] = (probabilities >= limiar).astype(int)

        prediction_cv[name] = predictions

        for threshold in thresholds:
            tag = f"{threshold:.2f}"
            recall_folds = results[f"test_recall_{tag}"]

            summary_rows.append(
                {
                    "Modelo": name,
                    "Limiar": threshold,
                    "validation AP": results["test_ap"].mean(),
                    "Precision": results[f"test_precision_{tag}"].mean(),
                    "Recall": recall_folds.mean(),
                    "FP": int(results[f"test_fp_{tag}"].sum()),
                    "Folds na meta": int((recall_folds >= 0.55).sum()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    print(summary.round(3).to_string(index=False))
    candidates = summary[summary["Recall"] >= 0.55].sort_values(
        ["Precision", "FP"],
        ascending=[False, True],
    )

    print(candidates.head(10).round(3).to_string(index=False))

    fig, axes = plt.subplots(
        1 + ((len(models) - 1) // 3),
        min(len(models), 3),
        figsize=(15, 4.5),
        constrained_layout=True,
    )

    axes = np.array(axes).ravel()

    for ax, (name, predictions) in zip(axes, prediction_cv.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_train,
            predictions,
            labels=[0, 1],
            display_labels=["Não alto", "Alto"],
            cmap="Blues",
            values_format="d",
            colorbar=False,
            im_kw={"vmin": 0, "vmax": len(y_train)},
            ax=ax,
        )

        ax.set_title(name)
        ax.set_xlabel("Expected class")
        ax.set_ylabel("Observed class")

    fig.suptitle(f"Out-of-Training previsions — CV/group - limiar: {limiar}")
    plt.show()

    return summary, results_cv, prediction_cv


def model_comparator(
    X_train,
    X_test,
    y_train,
    y_test,
    groups_train,
    pre_processor,
):

    baseline = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            ("model", DummyClassifier(strategy="prior", random_state=42)),
        ]
    )
    logisticregresison = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                    C=0.005,
                    l1_ratio=0.25,
                    random_state=42,
                ),
            ),
        ]
    )
    tree = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=3,
                    min_samples_leaf=30,
                    # class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    models = {
        "baseline": baseline,
        "logistic_regression": logisticregresison,
        "decision_tree": tree,
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X_train, y_train, groups=groups_train))

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        if y_train.iloc[train_idx].nunique() < 2 or y_train.iloc[val_idx].nunique() < 2:
            raise ValueError(f"Round {fold}: train or test missing at least one class")

    # classification_model_error(models, splits)
    forest = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=30,
                    max_depth=3,
                    # class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    boosting = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.3,
                    max_depth=2,
                    min_samples_leaf=50,
                    l2_regularization=10,
                    random_state=42,
                    early_stopping=False,
                    # class_weight="balanced",
                ),
            ),
        ]
    )
    svc = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            (
                "model",
                SVC(
                    kernel="rbf",
                    gamma=0.01,
                    C=0.001,
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    xgboost = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )
    knn = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=15,
                    weights="distance",
                    metric="minkowski",
                    p=2,
                    n_jobs=1,
                ),
            ),
        ]
    )
    bayes = Pipeline(
        [
            ("preprocess", clone(pre_processor)),
            (
                "model",
                GaussianNB(var_smoothing=1.0),
            ),
        ]
    )

    models["random_forest"] = forest
    models["gradient_boosting"] = boosting
    models["SVC"] = svc
    models["XGBoost"] = xgboost
    models["KNN"] = knn
    models["Naive_Bayes"] = bayes

    n_logistic = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    [
                        "incident_angle",
                        "magnitude",
                        "log_depth",
                        "expected_amp",
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                    C=0.005,
                    l1_ratio=0.25,
                    random_state=42,
                ),
            ),
        ]
    )
    n_svc = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    [
                        "phase",
                        "depth",
                        "axis",
                        "takeoff_angle",
                        "incident_angle",
                        "ray_param",
                        "magnitude",
                        "magnitude_type",
                        "epicenter_distance",
                        "log_depth",
                        # "log_dist",
                        # "axis_noise",
                        "expected_amp",
                    ]
                ),
            ),
            (
                "model",
                SVC(
                    kernel="rbf",
                    gamma=0.01,
                    C=0.001,
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    n_bayes = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    [
                        "phase",
                        "depth",
                        "axis",
                        "takeoff_angle",
                        # "incident_angle",
                        # "ray_param",
                        "magnitude",
                        "magnitude_type",
                        "epicenter_distance",
                        "log_depth",
                        "log_dist",
                        "axis_noise",
                        "expected_amp",
                    ]
                ),
            ),
            (
                "model",
                GaussianNB(var_smoothing=1.0),
            ),
        ]
    )

    # models["logistic_regression"] = n_logistic
    # models["SVC"] = n_svc
    # models["Naive_Bayes"] = n_bayes

    # classification_model_error(models, splits, X_train, y_train)

    model_searches = make_model_searches(models)

    all_results, leaders, studies = run_optuna_searches(
        model_searches=model_searches,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        splits=splits,
        thresholds=np.round(np.arange(0.20, 0.91, 0.05), 2),
        recall_min=0.55,
        min_folds=4,
        output_dir="Data/optuna_searches",
        feature_version="atual",
        n_jobs=-1,
        min_configurations=100,
    )

    columns = [
        "Modelo",
        "Criterio",
        "Status",
        "Limiar",
        "Precision",
        "Recall",
        "FP",
        "validation AP",
    ]

    print(leaders[columns].round(3).to_string(index=False))
    print("Resultados salvos em:", all_results.attrs["output_dir"])

    # return all_results, leaders, studies

    ANALYZE_FEATURES = False

    if ANALYZE_FEATURES:
        candidates, selected = select_feature_candidates(
            results_csv="Data/model_searches/run_at2evvlj/all_results.csv",
            model_searches=model_searches,
            names=("SVC", "logistic_regression", "Naive_Bayes"),
            criterion="precisao",
            recall_min=0.55,
            min_folds=4,
        )

        print(selected[["Modelo", "Limiar", "Params"]].to_string(index=False))

        blocks = {
            "sem_profundidade": ["depth", "log_depth"],
            "sem_magnitude_derivado": ["magnitude", "expected_amp"],
            "sem_distancia_derivados": [
                "epicenter_distance",
                "log_dist",
                "expected_amp",
            ],
            "sem_geometria": ["ray_param", "takeoff_angle", "incident_angle"],
        }

        analysis = run_feature_analysis(
            candidates=candidates,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            splits=splits,
            make_preprocessor=make_preprocessor,
            feature_blocks=blocks,
            n_repeats=20,
            do_permutation=True,
            do_ablation=True,
            n_jobs=1,
            output_dir="Data/feature_analysis",
        )

        print("\nPermutation importance:")
        print(analysis["permutation_summary"].round(3).to_string(index=False))

        ablation_columns = [
            "Modelo",
            "Experimento",
            "validation AP",
            "Precision",
            "Recall",
            "FP",
            "Delta Precision",
            "Delta Recall",
            "Delta FP",
            "Folds na meta",
        ]

        print("\nAblação:")
        print(
            analysis["ablation_summary"][ablation_columns]
            .round(3)
            .to_string(index=False)
        )

        if not analysis["permutation_summary"].empty:
            plot_feature_importance(analysis["permutation_summary"])
            plt.show()

        #     return analysis

        final_candidates = {
            name: candidates[name]
            for name in ("SVC", "logistic_regression", "Naive_Bayes")
        }

        final = fit_final_models(
            candidates=final_candidates,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir="Data/final_models",
        )

        print("\nAvaliação no teste reservado:")
        print(final["metrics"].round(3).to_string(index=False))

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        matrices = {
            "SVC": np.array([[246, 93], [336, 189]]),
            "Logistic Regression": np.array([[299, 40], [347, 178]]),
            "Naive Bayes": np.array([[231, 108], [345, 180]]),
        }

        vmax = max(cm.max() for cm in matrices.values())

        for axis, (name, cm) in zip(ax, matrices.items()):
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Não alto", "Alto"],
            ).plot(
                ax=axis,
                cmap="Blues",
                values_format="d",
                colorbar=False,
                im_kw={"vmin": 0, "vmax": vmax},
            )
            axis.set_title(name)

        plt.tight_layout()
        plt.show()

        weights = final["logistic_weights"]["logistic_regression"]
        intercept = final["logistic_intercepts"]["logistic_regression"]

        print("\nPesos da regressão logística:")
        print(weights.round(6).to_string(index=False))

        print("\nIntercepto:", intercept)
        print("\nResultados salvos em:", final["output_dir"])

        return final

    plan = search_plan(
        model_searches,
        n_splits=len(splits),
    )

    print(plan.to_string(index=False))

    # all_results, leaders, searches = run_model_searches(
    #     model_searches=model_searches,
    #     X_train=X_train,
    #     y_train=y_train,
    #     groups_train=groups_train,
    #     splits=splits,
    #     thresholds=np.round(np.arange(0.20, 0.91, 0.05), 2),
    #     recall_min=0.55,
    #     min_folds=4,
    #     output_dir="Data/model_searches",
    #     feature_version="atual",
    #     n_jobs=-1,
    # )
    # columns = [
    #     "Modelo",
    #     "Status",
    #     "Limiar",
    #     "validation AP",
    #     "Precision",
    #     "Recall",
    #     "Recall minimo",
    #     "FP",
    #     "TP",
    #     "Folds na meta",
    # ]

    # for criterion in ["precisao", "menos_fp"]:
    #     print(f"\nCritério: {criterion}")

    #     table = leaders.loc[
    #         leaders["Criterio"] == criterion,
    #         columns,
    #     ]

    #     print(table.round(3).to_string(index=False))


def heat_map(
    sub_table: pd.DataFrame,
    data_type: Literal["score", "n_earthquakes", "both"],
    auxiliar_table: pd.DataFrame | None = None,
):
    """
    Function that generates heatmaps summarizing the relationship between
    seismic phases and sensor axes.

    Input:

    - sub_table: dataframe containing the data used to build the heatmap.
    Depending on the selected mode, it must contain the columns
    `phase`, `axis`, and either `score` or `event_id`

    - data_type: type of visualization to generate. Accepted values are:

        - "score": plots a heatmap of the mean score for each
        phase-axis combination

        - "n_earthquakes": plots a heatmap of the number of unique
        earthquakes (`event_id`) for each phase-axis combination

        - "both": plots both heatmaps side by side in the same figure

    - auxiliar_table: dataframe used only when `data_type="both"`. It must contain the columns `phase`, `axis`, and `event_id`

    Output:

    - None

    The function creates pivot tables from the input data and visualizes
    them using Seaborn heatmaps. Depending on the selected mode, it displays
    the average score distribution, the number of associated earthquakes,
    or both metrics simultaneously using
    """

    if data_type == "score":
        pivoted_table = pd.pivot_table(
            data=sub_table,
            index="phase",
            columns="axis",
            values="score",
            aggfunc="mean",
        )

        plt.figure(figsize=(14, 10))

        sns.heatmap(pivoted_table, cmap="viridis", annot=True, fmt=".0%")
        plt.title("Density of high scores")

    elif data_type == "n_earthquakes":
        pivoted_table = pd.pivot_table(
            data=sub_table,
            index="phase",
            columns="axis",
            values="event_id",
            aggfunc="nunique",
            fill_value=0,
        )

        plt.figure(figsize=(14, 10))

        sns.heatmap(pivoted_table, cmap="viridis", annot=True, fmt="d")
        plt.title("Associated Earthquakes")

    else:
        fig, axis = plt.subplots(1, 2, figsize=(10, 6))

        pivoted_table = pd.pivot_table(
            data=sub_table,
            index="phase",
            columns="axis",
            values="score",
            aggfunc="mean",
        )

        sns.heatmap(pivoted_table, cmap="viridis", annot=True, fmt=".0%", ax=axis[0])
        axis[0].set_title("Density of high scores")
        axis[0].set_xlabel("Axis")
        axis[0].set_ylabel("Phase")

        pivoted_table = pd.pivot_table(
            data=auxiliar_table,
            index="phase",
            columns="axis",
            values="event_id",
            aggfunc="nunique",
            fill_value=0,
        )

        sns.heatmap(pivoted_table, cmap="viridis", annot=True, fmt="d", ax=axis[1])
        axis[1].set_title("Associated Earthquakes")
        axis[1].set_xlabel("Axis")
        axis[1].set_ylabel("Phase")

    plt.show()


def scatter_earthquake(unified_table):
    names = np.unique(unified_table["event_id"].to_list())

    auxiliary_table = pd.DataFrame(
        {
            "magnitude": [],
            "epicenter_distance": [],
            "high_proportion": [],
            "number_of_phases": [],
        }
    )

    for name in names:
        event_table = unified_table[unified_table["event_id"] == name]

        mag = event_table["magnitude"].iloc[0]
        ep_dist = event_table["epicenter_distance"].iloc[0]

        high_proportions = len(
            event_table[event_table["p_value"] < 0.05]["p_value"].to_list()
        ) / len(event_table["p_value"])

        n_phase = event_table["phase"].nunique()

        auxiliary_table.loc[len(auxiliary_table)] = [
            mag,
            ep_dist,
            high_proportions,
            n_phase,
        ]

    sns.scatterplot(
        data=auxiliary_table,
        x="magnitude",
        y="epicenter_distance",
        hue="high_proportion",
        size="number_of_phases",
        palette="viridis",
    )

    plt.show()


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
        try:
            path_surface = "Data/" + name + "/Surface_resultado.xlsx"
            path_body = "Data/" + name + "/Body_resultado.xlsx"
            df_surface = pd.read_excel(path_surface, sheet_name=-1)
            df_body = pd.read_excel(path_body, sheet_name=-1)

            unified_table = union_tables(
                pd.concat([df_surface, df_body]), unified_table
            )
        except FileNotFoundError:
            pass

    # sub_table = pd.DataFrame(
    #     {
    #         "phase": unified_table["phase"].to_list(),
    #         "axis": unified_table["axis"].to_list(),
    #         "score": unified_table["score"].to_list(),
    #     }
    # )
    # sub_table_name = pd.DataFrame(
    #     {
    #         "phase": unified_table["phase"].to_list(),
    #         "axis": unified_table["axis"].to_list(),
    #         "event_id": unified_table["event_id"].to_list(),
    #     }
    # )

    # sub_table.dropna()
    # sub_table["score"] = (sub_table["score"] > 0.9).astype(int)

    # heat_map(sub_table, "both", sub_table_name)

    # scatter_earthquake(unified_table)

    model_type = "l1"

    X_train, X_test, y_train, y_test, groups_train, pre_processor = pre_treatment(
        unified_table, "classification" if model_type != "linear" else "continuous"
    )

    model_comparator(X_train, X_test, y_train, y_test, groups_train, pre_processor)
