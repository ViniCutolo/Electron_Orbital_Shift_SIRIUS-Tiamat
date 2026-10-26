"""Buscas agrupadas para os classificadores sismicos do projeto Sirius.

Compativel com os pipelines existentes, com etapas 'preprocess' e 'model'.
Nao le seus arquivos e nao executa treinamento ao importar este modulo.

COMO USAR, dentro de model_comparator, depois de montar models e splits:

    from model_searches import (
        make_model_searches, search_plan, run_model_searches,
    )

    model_searches = make_model_searches(models)
    print(search_plan(model_searches, len(splits)).to_string(index=False))

    all_results, leaders, searches = run_model_searches(
        model_searches=model_searches,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        splits=splits,
        thresholds=np.round(np.arange(0.20, 0.91, 0.05), 2),
        recall_min=0.55,
        min_folds=4,
        output_dir='Data/model_searches',
        feature_version='atual',
        n_jobs=-1,
    )

Cada busca aprende o preprocessamento dentro de cada fold. Os mesmos folds
e limiares sao usados para todos os classificadores. O conjunto de teste
reservado nao e recebido nem usado por estas funcoes.

Interpretacao: os resultados sao de selecao em CV, nao uma estimativa
independente do desempenho depois da escolha de atributos/parametros/corte.

Arquivos: uma subpasta nova por execucao, com os resultados de cada modelo,
resultados completos acumulados, lideres e protocolo. Os CSVs sao atualizados
ao concluir cada modelo. Nao ha retomada automatica e nao sao salvos modelos
treinados; o GridSearchCV usa refit=False.

Validado com scikit-learn 1.8, numpy, pandas, joblib e threadpoolctl. A configuracao do SVC
mantem probability=True para reproduzir a avaliacao por probabilidades ja usada
no projeto; versoes futuras do sklearn podem exigir migrar essa calibracao.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_config
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from threadpoolctl import threadpool_limits


def make_model_searches(models):
    """Clona os pipelines existentes e associa uma grade a cada um."""

    c_logistic = [
        0.003,
        0.005,
        0.006,
        0.008,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
        0.06,
        0.08,
        0.1,
        0.12,
        0.15,
        0.2,
        0.25,
        0.3,
        1.0,
    ]

    return {
        "baseline": {
            "pipeline": clone(models["baseline"]),
            "param_grid": {},
        },
        "logistic_regression": {
            "pipeline": clone(models["logistic_regression"]).set_params(
                model__solver="saga",
                model__max_iter=5000,
            ),
            "param_grid": {
                "model__C": c_logistic,
                "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        },
        "decision_tree": {
            "pipeline": clone(models["decision_tree"]),
            "param_grid": {
                "model__max_depth": [2, 3, 4, 6, 8],
                "model__min_samples_leaf": [5, 10, 20, 30, 50],
                "model__ccp_alpha": [0.0, 0.0001, 0.001, 0.01],
            },
        },
        "random_forest": {
            "pipeline": clone(models["random_forest"]).set_params(
                model__n_jobs=1,
            ),
            "param_grid": {
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_leaf": [5, 10, 20, 30, 50],
                "model__max_features": ["sqrt", 0.5, 0.75, 1.0],
            },
        },
        "gradient_boosting": {
            "pipeline": clone(models["gradient_boosting"]).set_params(
                model__early_stopping=False,
            ),
            "param_grid": {
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3],
                "model__max_iter": [50, 100, 200, 400, 600],
                "model__max_depth": [1, 2],
                "model__l2_regularization": [1.0, 10.0],
            },
        },
        "SVC": {
            "pipeline": clone(models["SVC"]).set_params(
                model__kernel="rbf",
                model__probability=True,
            ),
            "param_grid": {
                "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
                "model__gamma": [
                    "scale",
                    0.0001,
                    0.0003,
                    0.001,
                    0.003,
                    0.01,
                    0.03,
                    0.1,
                    0.3,
                    1.0,
                ],
            },
        },
        "XGBoost": {
            "pipeline": clone(models["XGBoost"]).set_params(
                model__n_jobs=1,
                model__tree_method="hist",
            ),
            "param_grid": {
                "model__n_estimators": [100, 200, 400, 600, 800],
                "model__max_depth": [2, 3, 4, 6, 8],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.7, 1.0],
            },
        },
        "KNN": {
            "pipeline": clone(models["KNN"]).set_params(
                model__n_jobs=1,
            ),
            "param_grid": {
                "model__n_neighbors": [3, 5, 7, 11, 15, 21, 31, 51, 75, 101],
                "model__weights": ["uniform", "distance"],
                "model__metric": [
                    "euclidean",
                    "manhattan",
                    "chebyshev",
                    "canberra",
                    "braycurtis",
                ],
            },
        },
        "Naive_Bayes": {
            "pipeline": clone(models["Naive_Bayes"]),
            "param_grid": {
                "model__var_smoothing": np.logspace(-12, 0, 100),
            },
        },
    }


def search_plan(model_searches, n_splits=5):
    """Conta combinacoes reais de parametros e ajustes externos de CV."""
    rows = []
    for name, spec in model_searches.items():
        count = len(ParameterGrid(spec["param_grid"]))
        rows.append(
            {
                "Modelo": name,
                "Combinacoes": count,
                "Ajustes CV": count * n_splits,
            }
        )
    return pd.DataFrame(rows)


def _tag(threshold):
    return format(float(threshold), ".12g")


def make_threshold_scorer(thresholds):
    """Calcula probabilidades uma vez por avaliacao e testa todos os cortes."""
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("Informe limiares finitos entre 0 e 1.")
    if len({_tag(t) for t in thresholds}) != len(thresholds):
        raise ValueError("Os limiares devem ser distintos.")

    def score(model, X, y):
        y_array = np.asarray(y)
        positive_column = np.flatnonzero(np.asarray(model.classes_) == 1)
        if len(positive_column) != 1:
            raise ValueError("O modelo precisa reconhecer a classe positiva 1.")
        probabilities = model.predict_proba(X)[:, positive_column[0]]
        values = {"ap": float(average_precision_score(y_array, probabilities))}
        for threshold in thresholds:
            tag = _tag(threshold)
            predicted = probabilities >= threshold
            values[f"precision_{tag}"] = precision_score(
                y_array,
                predicted,
                zero_division=0,
            )
            values[f"recall_{tag}"] = recall_score(
                y_array,
                predicted,
                zero_division=0,
            )
            values[f"fp_{tag}"] = int(((y_array == 0) & predicted).sum())
            values[f"tp_{tag}"] = int(((y_array == 1) & predicted).sum())
        return values

    return score


def _search_table(
    search,
    name,
    thresholds,
    n_splits,
    recall_min,
    n_positive,
    n_rows,
    elapsed,
    feature_version,
):
    """Converte cv_results_ em linhas modelo x parametros x limiar."""
    records = []
    n_configs = len(search.cv_results_["params"])
    for index, result in pd.DataFrame(search.cv_results_).iterrows():
        for threshold in thresholds:
            tag = _tag(threshold)
            recalls = np.array(
                [result[f"split{k}_test_recall_{tag}"] for k in range(n_splits)]
            )
            fp = int(sum(result[f"split{k}_test_fp_{tag}"] for k in range(n_splits)))
            tp = int(sum(result[f"split{k}_test_tp_{tag}"] for k in range(n_splits)))
            records.append(
                {
                    "Modelo": name,
                    "Atributos": feature_version,
                    "id": int(index),
                    "Params": result["params"],
                    "Limiar": float(threshold),
                    "train AP": result["mean_train_ap"],
                    "validation AP": result["mean_test_ap"],
                    "validation DP AP": result["std_test_ap"],
                    "Precision": result[f"mean_test_precision_{tag}"],
                    "Precision DP": result[f"std_test_precision_{tag}"],
                    "Recall": float(recalls.mean()),
                    "Recall minimo": float(recalls.min()),
                    "Folds na meta": int((recalls >= recall_min).sum()),
                    "FP": fp,
                    "TP": tp,
                    "Positivos previstos": tp + fp,
                    "Precision global": tp / (tp + fp) if tp + fp else 0.0,
                    "Recall global": tp / n_positive,
                    "Taxa FP global": fp / (n_rows - n_positive),
                    "Combinacoes": n_configs,
                    "Tempo busca (s)": elapsed,
                }
            )
    return pd.DataFrame(records)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Tipo nao serializavel: {type(value).__name__}")


def _write_csv(table, path):
    # Parametros sao JSON no CSV, mas continuam dicionarios nos DataFrames.
    exported = table.copy()
    for column in ("Params", "params"):
        if column in exported:
            exported[column] = exported[column].map(
                lambda value: (
                    json.dumps(value, default=_json_default)
                    if isinstance(value, dict)
                    else ""
                )
            )
    exported.to_csv(path, index=False, encoding="utf-8-sig")


def run_model_searches(
    model_searches,
    X_train,
    y_train,
    groups_train,
    splits,
    thresholds,
    recall_min=0.55,
    min_folds=4,
    output_dir=None,
    feature_version="atual",
    n_jobs=-1,
):
    """Executa as buscas e devolve (all_results, leaders, searches).

    leaders tem ate dois criterios por modelo: 'precisao' e 'menos_fp'.
    Ambos exigem Recall medio >= recall_min e pelo menos min_folds na meta.
    Se nenhum candidato passar, o modelo fica registrado como 'sem elegivel'.
    Nenhuma meta e reduzida automaticamente.
    """
    y_array = np.asarray(y_train)
    groups = np.asarray(groups_train)
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train deve ser um DataFrame com as colunas dos pipelines.")
    if not (len(X_train) == len(y_array) == len(groups)):
        raise ValueError("X_train, y_train e groups_train devem estar alinhados.")
    if y_array.ndim != 1 or set(np.unique(y_array)) != {0, 1}:
        raise ValueError("Use y_train unidimensional com as classes 0 e 1.")
    if pd.isna(groups).any():
        raise ValueError("groups_train nao pode conter grupos ausentes.")
    if not 0 < recall_min <= 1:
        raise ValueError("recall_min deve estar em (0, 1].")
    splits = list(splits)
    if not splits or not 1 <= min_folds <= len(splits):
        raise ValueError("min_folds deve estar entre 1 e o numero de folds.")
    coverage = np.zeros(len(y_array), dtype=int)
    fold_metadata = []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        if set(groups[train_idx]).intersection(groups[val_idx]):
            raise ValueError(
                f"Fold {fold}: terremotos compartilhados entre treino e validacao."
            )
        if (
            len(np.unique(y_array[train_idx])) != 2
            or len(np.unique(y_array[val_idx])) != 2
        ):
            raise ValueError(f"Fold {fold}: treino ou validacao sem as duas classes.")
        np.add.at(coverage, val_idx, 1)
        fold_metadata.append(
            {
                "fold": fold,
                "validation_positions": val_idx.tolist(),
                "validation_groups": sorted({str(g) for g in groups[val_idx]}),
            }
        )
    if not np.all(coverage == 1):
        raise ValueError(
            "Cada linha deve aparecer em exatamente uma validacao para somar TP/FP."
        )

    thresholds = tuple(float(t) for t in thresholds)
    scorer = make_threshold_scorer(thresholds)
    plan = search_plan(model_searches, len(splits))
    if plan.empty:
        raise ValueError("Forneca pelo menos um modelo para a busca.")
    run_dir = None
    if output_dir is not None:
        parent = Path(output_dir)
        parent.mkdir(parents=True, exist_ok=True)
        run_dir = Path(mkdtemp(prefix="run_", dir=parent))
        metadata = {
            "sklearn_version": sklearn.__version__,
            "feature_version": feature_version,
            "input_columns": X_train.columns.tolist(),
            "n_rows": len(y_array),
            "n_groups": len(np.unique(groups)),
            "n_positive": int(y_array.sum()),
            "recall_min": recall_min,
            "min_folds": min_folds,
            "thresholds": thresholds,
            "plan": plan.to_dict(orient="records"),
            "folds": fold_metadata,
            "purpose": "selecao por CV; teste reservado nao utilizado",
        }
        (run_dir / "protocol.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        print(f"Resultados desta execucao: {run_dir}", flush=True)

    frames, leader_rows, searches = [], [], {}
    for name, spec in model_searches.items():
        count = len(ParameterGrid(spec["param_grid"]))
        print(
            f"\n{name}: {count} configuracoes, {count * len(splits)} ajustes CV.",
            flush=True,
        )
        search = GridSearchCV(
            estimator=clone(spec["pipeline"]),
            param_grid=spec["param_grid"],
            scoring=scorer,
            cv=splits,
            refit=False,
            return_train_score=True,
            error_score="raise",
            n_jobs=n_jobs,
            pre_dispatch="n_jobs",
            verbose=1,
        )
        started = perf_counter()
        # Um nivel de paralelismo: as buscas distribuem fits; cada fit usa uma thread nativa.
        with (
            parallel_config(backend="loky", inner_max_num_threads=1),
            threadpool_limits(limits=1),
        ):
            search.fit(X_train, y_train)
        elapsed = perf_counter() - started
        searches[name] = search
        table = _search_table(
            search,
            name,
            thresholds,
            len(splits),
            recall_min,
            int(y_array.sum()),
            len(y_array),
            elapsed,
            feature_version,
        )
        frames.append(table)
        eligible = table[
            (table["Recall"] >= recall_min) & (table["Folds na meta"] >= min_folds)
        ]
        for criterion, columns, ascending in (
            ("precisao", ["Precision", "FP", "validation AP"], [False, True, False]),
            ("menos_fp", ["FP", "Precision", "validation AP"], [True, False, False]),
        ):
            if eligible.empty:
                empty_row = {column: np.nan for column in table.columns}
                empty_row.update(
                    {
                        "Modelo": name,
                        "Criterio": criterion,
                        "Status": "sem elegivel",
                        "Atributos": feature_version,
                        "Combinacoes": count,
                        "Tempo busca (s)": elapsed,
                        "Params": None,
                    }
                )
                leader_rows.append(empty_row)
            else:
                best = (
                    eligible.sort_values(columns, ascending=ascending).iloc[0].to_dict()
                )
                best.update({"Criterio": criterion, "Status": "elegivel"})
                leader_rows.append(best)
        all_results = pd.concat(frames, ignore_index=True)
        leaders = pd.DataFrame(leader_rows)
        if run_dir is not None:
            file_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            _write_csv(
                pd.DataFrame(search.cv_results_), run_dir / f"{file_name}_cv.csv"
            )
            _write_csv(table, run_dir / f"{file_name}_thresholds.csv")
            _write_csv(all_results, run_dir / "all_results.csv")
            _write_csv(leaders, run_dir / "leaders.csv")
        print(f"{name}: concluido em {elapsed:.1f} s.", flush=True)

    all_results.attrs["output_dir"] = str(run_dir) if run_dir is not None else None
    leaders.attrs["output_dir"] = all_results.attrs["output_dir"]
    return all_results, leaders, searches
