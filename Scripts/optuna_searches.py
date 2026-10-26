"""Busca real com Optuna para as grades ja existentes do projeto Sirius.

DEPENDENCIAS: optuna, scikit-learn, pandas, numpy, joblib, threadpoolctl e
o seu model_searches.py atual, na mesma pasta. Este modulo NAO redefine as
grades: preserva inclusive XGBoost, KNN e Naive_Bayes que voce adicionou.
Nao executa ajustes ao importar. Validado com sklearn 1.8 e Optuna 5.0.

Troca em seismo_detection.py:

    from model_searches import make_model_searches, search_plan
    from optuna_searches import run_optuna_searches

    # Depois de criar models e os mesmos splits por event_id:
    model_searches = make_model_searches(models)
    all_results, leaders, studies = run_optuna_searches(
        model_searches=model_searches,
        X_train=X_train, y_train=y_train, groups_train=groups_train,
        splits=splits, thresholds=np.round(np.arange(0.20, 0.91, 0.05), 2),
        recall_min=0.55, min_folds=4,
        output_dir="Data/optuna_searches", feature_version="atual", n_jobs=-1,
    )
    print(leaders.round(3).to_string(index=False))

Cada trial sugere UMA configuracao da sua grade e ajusta pipelines clonados
nos folds agrupados. Todos os limiares sao avaliados com essas probabilidades,
sem ajustes extras. O objetivo do Optuna e a maior precisao media entre os
cortes elegiveis (recall medio >= recall_min e >= min_folds na meta).
Configuracao sem nenhum corte elegivel recebe valor objetivo -1.

O GridSampler percorre a grade inteira, sem pruning, em trials sequenciais.
Os folds podem rodar em paralelo; assim nao ha sugestoes simultaneas repetidas.
Nao e busca bayesiana. AP continua registrada como metrica e desempate.
As tabelas leaders preservam os criterios 'precisao' e 'menos_fp' anteriores;
use essas tabelas para selecao, inclusive para os desempates, em vez de supor
que study.best_params resolve os dois criterios.

Cada modelo nao-baseline precisa de >=100 configuracoes DISTINTAS na grade.
O DummyClassifier e isento dessa exigencia. O modulo executa a grade inteira
(100 trials quando ela tem 100 pontos), nao 100 fits contando folds/cortes.

Sai uma nova pasta por execucao, com optuna.db (SQLite, registro persistente
dos trials), *_trials.csv, *_cv.csv, *_thresholds.csv, all_results.csv,
leaders.csv, trial_counts.csv e protocol.json. Os trials ficam gravados na
base conforme terminam; os CSVs sao consolidados ao terminar cada modelo.
Nao ha retomada automatica nem importacao de resultados antigos como trials.

O conjunto de teste reservado nao e recebido. Parametros, atributos e cortes
devem continuar sendo escolhidos somente no desenvolvimento. Se voce repetir
as grades para migrar a ferramenta, preserve dados, folds e configuracoes.

O retorno studies contem objetos optuna.Study, nao GridSearchCV; nao tem
best_estimator_ ou cv_results_. O ajuste final continua sendo feito com a
funcao fit_final_models e candidatos criados a partir do CSV de resultados.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import sklearn
from joblib import effective_n_jobs, parallel_config

# Reutilizar a formatacao e o scorer do seu modulo, sem chamar GridSearchCV.
from model_searches import (
    _json_default,
    _search_table,
    _write_csv,
    make_threshold_scorer,
    search_plan,
)
from optuna.trial import TrialState
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


def _fingerprint(params):
    return json.dumps(params, sort_keys=True, default=_json_default)


def _normalize_grid(raw_grid):
    if not isinstance(raw_grid, dict):
        raise TypeError("Use uma grade plana (dict) por modelo, como no script atual.")
    grid = {}
    for key, choices in raw_grid.items():
        if not isinstance(key, str) or isinstance(choices, (str, bytes)):
            raise ValueError(
                "Cada hiperparametro precisa de um nome e lista de valores."
            )
        values = []
        for choice in choices:
            value = choice.item() if isinstance(choice, np.generic) else choice
            if (
                value is not None
                and not isinstance(value, (str, int, float, bool))
                or isinstance(value, float)
                and not np.isfinite(value)
            ):
                raise ValueError(f"{key}: use valores escalares finitos ou None.")
            if any(value == previous for previous in values):
                raise ValueError(
                    f"{key}: valores duplicados nao contam como combinacoes distintas."
                )
            values.append(value)
        if not values:
            raise ValueError(f"Grade vazia para {key}.")
        grid[key] = values
    return grid


def _validated_splits(X, y, groups, splits, min_folds):
    if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
        raise ValueError("X_train deve ser um DataFrame com colunas distintas.")
    for values, name in [(y, "y_train"), (groups, "groups_train")]:
        if isinstance(values, pd.Series) and not values.index.equals(X.index):
            raise ValueError(f"Indices de X_train e {name} desalinhados.")
    y, groups = np.asarray(y), np.asarray(groups)
    if (
        y.ndim != 1
        or groups.ndim != 1
        or len(X) != len(y)
        or len(X) != len(groups)
        or pd.isna(y).any()
        or pd.isna(groups).any()
    ):
        raise ValueError(
            "Alinhe X_train, y_train e groups_train, sem rotulos/grupos ausentes."
        )
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Use y_train binario, com as classes 0 e 1.")
    splits = list(splits)
    if len(splits) < 2 or not 1 <= min_folds <= len(splits):
        raise ValueError("Numero de folds ou min_folds invalido.")
    checked, metadata = [], []
    coverage = np.zeros(len(y), dtype=int)
    for fold, pair in enumerate(splits, 1):
        train_idx, val_idx = (np.asarray(indexes) for indexes in pair)
        for indexes in (train_idx, val_idx):
            if (
                indexes.ndim != 1
                or not len(indexes)
                or not np.issubdtype(indexes.dtype, np.integer)
                or len(np.unique(indexes)) != len(indexes)
                or indexes.min() < 0
                or indexes.max() >= len(y)
            ):
                raise ValueError(f"Fold {fold}: indices invalidos.")
        if set(groups[train_idx]).intersection(groups[val_idx]):
            raise ValueError(f"Fold {fold}: ha terremotos nos dois conjuntos.")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[val_idx])) != 2:
            raise ValueError(f"Fold {fold}: treino ou validacao sem as duas classes.")
        coverage[val_idx] += 1
        checked.append((train_idx, val_idx))
        metadata.append(
            {
                "fold": fold,
                "train_positions": train_idx.tolist(),
                "validation_positions": val_idx.tolist(),
                "validation_groups": sorted({str(g) for g in groups[val_idx]}),
            }
        )
    if not np.all(coverage == 1):
        raise ValueError("Cada linha precisa aparecer em exatamente uma validacao.")
    return y, groups, checked, metadata


def _cv_row(params, results):
    """Adaptar os scores reais de cross_validate ao formato de tabela ja usado."""
    row = {"params": params}
    for key, values in results.items():
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Avaliacao retornou valores nao finitos em {key}.")
        row[f"mean_{key}"] = float(values.mean())
        row[f"std_{key}"] = float(values.std(ddof=0))
        if key.startswith(("train_", "test_")):
            for fold, value in enumerate(values):
                row[f"split{fold}_{key}"] = float(value)
    return row


def _leaders(table, recall_min, min_folds):
    rows = []
    eligible = table.loc[
        (table["Recall"] >= recall_min) & (table["Folds na meta"] >= min_folds)
    ]
    for criterion, columns, ascending in [
        ("precisao", ["Precision", "FP", "validation AP"], [False, True, False]),
        ("menos_fp", ["FP", "Precision", "validation AP"], [True, False, False]),
    ]:
        if eligible.empty:
            row = {column: np.nan for column in table.columns}
            for column in ["Modelo", "Atributos", "Combinacoes", "Tempo busca (s)"]:
                row[column] = table.iloc[0][column]
            row.update({"Params": None, "Status": "sem elegivel"})
        else:
            row = eligible.sort_values(columns, ascending=ascending).iloc[0].to_dict()
            row["Status"] = "elegivel"
        row["Criterio"] = criterion
        rows.append(row)
    return rows


def run_optuna_searches(
    model_searches,
    X_train,
    y_train,
    groups_train,
    splits,
    thresholds,
    recall_min=0.55,
    min_folds=4,
    output_dir="Data/optuna_searches",
    feature_version="atual",
    n_jobs=-1,
    *,
    random_state=42,
    min_configurations=100,
):
    """Executa as grades com Optuna e retorna (all_results, leaders, studies).

    min_configurations=100 e o minimo exigido por classificador nao-baseline.
    Valores menores existem apenas para testes pequenos da integracao.
    n_jobs distribui folds, enquanto os trials permanecem sequenciais.
    """
    if not 0 < recall_min <= 1:
        raise ValueError("recall_min deve estar em (0, 1].")
    if not isinstance(min_configurations, int) or min_configurations < 1:
        raise ValueError("min_configurations precisa ser inteiro positivo.")
    y, groups, splits, fold_metadata = _validated_splits(
        X_train,
        y_train,
        groups_train,
        splits,
        min_folds,
    )
    thresholds = tuple(float(t) for t in thresholds)
    scorer = make_threshold_scorer(thresholds)
    if not model_searches:
        raise ValueError("Informe os modelos e as grades existentes.")

    prepared = {}
    for name, spec in model_searches.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Cada modelo precisa de um nome nao vazio.")
        pipeline = spec["pipeline"]
        if not isinstance(pipeline, Pipeline) or pipeline.steps[-1][0] != "model":
            raise TypeError(f"{name}: use Pipeline com classificador final 'model'.")
        grid = _normalize_grid(spec["param_grid"])
        configurations = list(ParameterGrid(grid))
        is_baseline = isinstance(pipeline.named_steps["model"], DummyClassifier)
        if not is_baseline and len(configurations) < min_configurations:
            raise ValueError(
                f"{name}: a grade tem {len(configurations)} configuracoes; "
                f"sao exigidas pelo menos {min_configurations}."
            )
        clone(pipeline).set_params(**configurations[0])
        prepared[name] = {
            "pipeline": pipeline,
            "param_grid": grid,
            "configurations": configurations,
            "baseline": is_baseline,
        }

    parent = Path(output_dir)
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="optuna_", dir=parent)).resolve()
    storage = f"sqlite:///{(run_dir / 'optuna.db').as_posix()}"
    fold_jobs = min(len(splits), effective_n_jobs(n_jobs))
    plan = search_plan(prepared, len(splits))
    protocol = {
        "optuna_version": optuna.__version__,
        "sklearn_version": sklearn.__version__,
        "engine": "Optuna GridSampler",
        "objective": "maior precisao em corte elegivel; -1 se nenhum",
        "feature_version": feature_version,
        "input_columns": X_train.columns.tolist(),
        "n_rows": len(y),
        "n_groups": len(np.unique(groups)),
        "n_positive": int(y.sum()),
        "recall_min": recall_min,
        "min_folds": min_folds,
        "thresholds": thresholds,
        "min_configurations": min_configurations,
        "random_state": random_state,
        "trials_parallel": 1,
        "folds_parallel": fold_jobs,
        "plan": plan.to_dict(orient="records"),
        "folds": fold_metadata,
        "models": {
            name: {"pipeline": repr(spec["pipeline"]), "grid": spec["param_grid"]}
            for name, spec in prepared.items()
        },
        "purpose": "selecao por CV no treino; teste reservado nao recebido",
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(f"Resultados Optuna: {run_dir}", flush=True)
    print(plan.to_string(index=False), flush=True)

    frames, leader_rows, studies, counts = [], [], {}, []
    old_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        for model_number, (name, spec) in enumerate(prepared.items(), 1):
            grid = spec["param_grid"]
            count = len(spec["configurations"])
            configuration_ids = {
                _fingerprint(params): index
                for index, params in enumerate(spec["configurations"])
            }
            slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "modelo"
            study_name = f"{model_number:02d}_{slug}"
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                sampler=optuna.samplers.GridSampler(grid, seed=random_state),
                pruner=optuna.pruners.NopPruner(),
            )
            study.set_user_attr("model_name", name)
            study.set_user_attr("planned_configurations", count)
            study.set_user_attr("feature_version", feature_version)
            cv_rows, threshold_tables = [], []
            started = perf_counter()

            def objective(trial):
                params = {
                    key: trial.suggest_categorical(key, values)
                    for key, values in grid.items()
                }
                pipeline = clone(spec["pipeline"]).set_params(**params)
                results = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=splits,
                    scoring=scorer,
                    return_train_score=True,
                    error_score="raise",
                    n_jobs=fold_jobs,
                    pre_dispatch=fold_jobs,
                )
                cv_row = _cv_row(params, results)
                # _search_table so formata scores; nao realiza ajustes.
                adapter = SimpleNamespace(
                    cv_results_={key: [value] for key, value in cv_row.items()}
                )
                table = _search_table(
                    adapter,
                    name,
                    thresholds,
                    len(splits),
                    recall_min,
                    int(y.sum()),
                    len(y),
                    perf_counter() - started,
                    feature_version,
                )
                table["id"] = configuration_ids[_fingerprint(params)]
                table["Optuna trial"] = trial.number
                table["Combinacoes"] = count
                eligible = table.loc[
                    (table["Recall"] >= recall_min)
                    & (table["Folds na meta"] >= min_folds)
                ]
                selected_threshold, value = None, -1.0
                if not eligible.empty:
                    best = eligible.sort_values(
                        ["Precision", "FP", "validation AP"],
                        ascending=[False, True, False],
                    ).iloc[0]
                    value = float(best["Precision"])
                    selected_threshold = float(best["Limiar"])
                trial.set_user_attr("cv_results", cv_row)
                trial.set_user_attr("selected_threshold_precision", selected_threshold)
                cv_row["optuna_trial"] = trial.number
                cv_row["configuration_id"] = int(table["id"].iloc[0])
                cv_rows.append(cv_row)
                threshold_tables.append(table)
                return value

            def progress(current_study, trial):
                finished = trial.number + 1
                if finished % 10 == 0 or finished == count:
                    print(f"{name}: {finished}/{count} trials concluidos.", flush=True)

            # Um nivel de paralelismo: folds externos; threads nativas limitadas.
            with (
                parallel_config(backend="loky", inner_max_num_threads=1),
                threadpool_limits(limits=1),
            ):
                study.optimize(
                    objective, n_trials=count, n_jobs=1, callbacks=[progress]
                )

            elapsed = perf_counter() - started
            completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            distinct = len({_fingerprint(trial.params) for trial in completed})
            if len(completed) != count or distinct != count:
                raise RuntimeError(
                    f"{name}: a grade nao foi concluida com configuracoes distintas."
                )
            table = (
                pd.concat(threshold_tables, ignore_index=True)
                .sort_values(
                    ["id", "Limiar"],
                    kind="stable",
                )
                .reset_index(drop=True)
            )
            table["Tempo busca (s)"] = elapsed
            frames.append(table)
            leader_rows.extend(_leaders(table, recall_min, min_folds))
            studies[name] = study
            counts.append(
                {
                    "Modelo": name,
                    "Baseline": spec["baseline"],
                    "Combinacoes planejadas": count,
                    "Trials concluidos": len(completed),
                    "Combinacoes distintas": distinct,
                    "Folds": len(splits),
                    "Ajustes CV": len(completed) * len(splits),
                    "Atende minimo": spec["baseline"] or distinct >= min_configurations,
                }
            )
            all_results, leaders = (
                pd.concat(frames, ignore_index=True),
                pd.DataFrame(leader_rows),
            )
            _write_csv(
                pd.DataFrame(cv_rows).sort_values("configuration_id"),
                run_dir / f"{study_name}_cv.csv",
            )
            _write_csv(table, run_dir / f"{study_name}_thresholds.csv")
            study.trials_dataframe(
                attrs=(
                    "number",
                    "value",
                    "datetime_start",
                    "datetime_complete",
                    "duration",
                    "params",
                    "state",
                )
            ).to_csv(run_dir / f"{study_name}_trials.csv", index=False)
            _write_csv(all_results, run_dir / "all_results.csv")
            _write_csv(leaders, run_dir / "leaders.csv")
            pd.DataFrame(counts).to_csv(run_dir / "trial_counts.csv", index=False)
            print(
                f"{name}: {distinct} configuracoes distintas; {elapsed:.1f} s.",
                flush=True,
            )
    finally:
        optuna.logging.set_verbosity(old_verbosity)

    all_results.attrs["output_dir"] = str(run_dir)
    leaders.attrs["output_dir"] = str(run_dir)
    print(
        f"Buscas concluidas. Evidencia dos trials: {run_dir / 'trial_counts.csv'}",
        flush=True,
    )
    return all_results, leaders, studies
