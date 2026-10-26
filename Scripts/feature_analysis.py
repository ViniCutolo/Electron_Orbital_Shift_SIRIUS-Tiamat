"""Permutation importance e ablacao para os pipelines do projeto Sirius.

Uso: importe este modulo em seismo_detection.py. As funcoes recebem os
mesmos X_train, y_train, groups_train, splits e make_preprocessor do script.
Nenhuma funcao recebe o conjunto de teste reservado.

Os parametros sao recuperados do CSV da busca. Cada candidato e treinado
novamente nos folds; GridSearchCV(refit=False) nao guardou esses ajustes.
Importancias sao calculadas na validacao, sobre as colunas originais, antes
do OneHotEncoder. A ablacao reconstroi o preprocessador e mantem parametros
e limiar fixos. Os resultados servem para exploracao/selecao de atributos,
nao para uma estimativa independente do desempenho depois dessa selecao.

A permutacao padrao embaralha linhas. Metadados repetidos por terremoto e
atributos correlacionados exigem cautela: a perturbacao pode formar exemplos
fisicamente inconsistentes. As importancias medem dependencia preditiva,
nao efeitos fisicos ou causais. Use tambem a ablacao por blocos.

DP entre folds descreve a variacao das medias de importancia dos folds.
Nao e intervalo de confianca; as repeticoes de permutacao nao sao novos eventos.

Somente numpy, pandas, sklearn, joblib, threadpoolctl e, para o grafico,
matplotlib. Nao le arquivos nem treina ao importar. Cada execucao salva uma
subpasta nova. Ha checkpoints por candidato, sem retomada automatica.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_config
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_score, recall_score
from threadpoolctl import threadpool_limits


def select_feature_candidates(
    results_csv, model_searches, names=("SVC", "logistic_regression", "Naive_Bayes"),
    *, criterion="precisao", recall_min=0.55, min_folds=4,
):
    """Recupera um candidato elegivel por modelo, sem rodar uma nova busca.

    Retorna (candidates, selected). candidates contem pipelines clonados,
    parametros e limiares. selected mostra as linhas escolhidas do CSV.
    model_searches e o registro criado por make_model_searches(models),
    preservando as configuracoes fixas usadas na busca, como max_iter=5000.
    """
    if criterion not in {"precisao", "menos_fp"}:
        raise ValueError("criterion deve ser 'precisao' ou 'menos_fp'.")
    if not 0 <= recall_min <= 1 or min_folds < 1:
        raise ValueError("Meta de recall ou quantidade de folds invalida.")
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Informe nomes de modelos distintos.")
    table = (results_csv.copy() if isinstance(results_csv, pd.DataFrame)
             else pd.read_csv(results_csv))
    required = {"Modelo", "Params", "Limiar", "Recall", "Folds na meta",
                "Precision", "FP", "validation AP"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")
    # O arquivo precisa representar uma unica versao de atributos.
    if "Atributos" in table and table["Atributos"].nunique(dropna=False) != 1:
        raise ValueError("Filtre uma unica versao de atributos antes da selecao.")
    eligible = table.loc[(table["Recall"] >= recall_min)
                         & (table["Folds na meta"] >= min_folds)]
    by, ascending = (["Precision", "FP", "validation AP"], [False, True, False])
    if criterion == "menos_fp":
        by, ascending = ["FP", "Precision", "validation AP"], [True, False, False]
    candidates, rows = {}, []
    for name in names:
        if name not in model_searches:
            raise ValueError(f"{name!r} nao esta no seu model_searches.")
        options = eligible.loc[eligible["Modelo"] == name]
        if options.empty:
            raise ValueError(f"Nenhuma configuracao elegivel para {name!r}.")
        row = options.sort_values(by, ascending=ascending, kind="stable").iloc[0]
        params = row["Params"]
        if isinstance(params, str):
            params = json.loads(params)  # Nunca usar eval para ler parametros.
        if not isinstance(params, dict):
            raise ValueError(f"Params invalido para {name!r}.")
        threshold = float(row["Limiar"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Limiar invalido para {name!r}.")
        candidates[name] = {
            "pipeline": clone(model_searches[name]["pipeline"]).set_params(**params),
            "threshold": threshold,
            "params": params,
        }
        rows.append(row.to_dict())
    return candidates, pd.DataFrame(rows).reset_index(drop=True)


def _probabilities(model, X):
    positive = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("O estimador precisa conter a classe positiva 1.")
    return model.predict_proba(X)[:, positive[0]]


def _metrics(y, probabilities, threshold):
    y = np.asarray(y)
    predicted = probabilities >= threshold
    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    return {
        "AP": float(average_precision_score(y, probabilities)),
        "Precision": float(precision_score(y, predicted, zero_division=0)),
        "Recall": float(recall_score(y, predicted, zero_division=0)),
        "FP": fp, "TP": tp,
        "FN": int(np.sum(y == 1)) - tp,
        "TN": int(np.sum(y == 0)) - fp,
    }


def _permutation_scorer(model, X, y, *, threshold):
    metrics = _metrics(y, _probabilities(model, X), threshold)
    # sklearn calcula score_original - score_permutado. Para FP usamos o
    # negativo: a importancia sera FP_permutado - FP_original (FP extras).
    return {"ap": metrics["AP"], "precision": metrics["Precision"],
            "recall": metrics["Recall"], "neg_fp": -metrics["FP"]}


def _validate_inputs(X, y, groups, splits):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or X.shape[1] < 2:
        raise ValueError("X_train deve ser um DataFrame com >=2 colunas distintas.")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("Os indices de X_train e y_train estao desalinhados.")
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(X) or len(groups) != len(X):
        raise ValueError("X_train, y_train e groups_train devem estar alinhados.")
    if set(np.unique(y)) != {0, 1} or pd.isna(groups).any():
        raise ValueError("Use y binario 0/1 e event_id sem valores ausentes.")
    splits = list(splits)
    if len(splits) < 2:
        raise ValueError("Informe pelo menos dois folds.")
    seen = np.zeros(len(X), dtype=int)
    validated = []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        for indexes in (train_idx, val_idx):
            if (indexes.ndim != 1 or not np.issubdtype(indexes.dtype, np.integer)
                    or not len(indexes) or len(np.unique(indexes)) != len(indexes)
                    or indexes.min() < 0 or indexes.max() >= len(X)):
                raise ValueError(f"Indices invalidos no fold {fold}.")
        if set(groups[train_idx]).intersection(groups[val_idx]):
            raise ValueError(f"Fold {fold}: ha terremotos nos dois conjuntos.")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[val_idx])) != 2:
            raise ValueError(f"Fold {fold}: faltam classes no treino/validacao.")
        seen[val_idx] += 1
        validated.append((train_idx, val_idx))
    if not np.all(seen == 1):
        raise ValueError("Cada linha deve ser validada exatamente uma vez.")
    return y, groups, validated


def _permutation_summary(rows):
    columns = ["Modelo", "Atributo", "Importancia AP", "DP entre folds",
               "Queda Precision", "Queda Recall", "FP extras", "Folds AP positiva"]
    if not rows:
        return pd.DataFrame(columns=columns)
    summaries = []
    for (name, feature), values in pd.DataFrame(rows).groupby(["Modelo", "Atributo"]):
        folds = values.groupby("Fold")[["Queda AP", "Queda Precision",
                                         "Queda Recall", "FP extras"]].mean()
        summaries.append({
            "Modelo": name, "Atributo": feature,
            "Importancia AP": folds["Queda AP"].mean(),
            "DP entre folds": folds["Queda AP"].std(ddof=1),
            "Queda Precision": folds["Queda Precision"].mean(),
            "Queda Recall": folds["Queda Recall"].mean(),
            "FP extras": folds["FP extras"].mean(),
            "Folds AP positiva": int((folds["Queda AP"] > 0).sum()),
        })
    return pd.DataFrame(summaries, columns=columns).sort_values(
        ["Modelo", "Importancia AP"], ascending=[True, False], ignore_index=True)


def _ablation_summary(rows, recall_min):
    table = pd.DataFrame(rows)
    summaries = []
    for (name, experiment), values in table.groupby(["Modelo", "Experimento"], sort=False):
        base = table.loc[(table["Modelo"] == name)
                         & (table["Experimento"] == "completo")].set_index("Fold")
        values = values.set_index("Fold")
        reference = base.loc[values.index]
        tp, fp = int(values["TP"].sum()), int(values["FP"].sum())
        positives = int((values["TP"] + values["FN"]).sum())
        delta_p = values["Precision"] - reference["Precision"]
        delta_r = values["Recall"] - reference["Recall"]
        summaries.append({
            "Modelo": name, "Experimento": experiment,
            "Limiar": values["Limiar"].iloc[0],
            "N atributos": int(values["N atributos"].iloc[0]),
            "validation AP": values["AP"].mean(),
            "validation DP AP": values["AP"].std(ddof=0),
            "Precision": values["Precision"].mean(),
            "Recall": values["Recall"].mean(),
            "Recall minimo": values["Recall"].min(),
            "FP": fp, "TP": tp,
            "Precision global": tp / (tp + fp) if tp + fp else 0.0,
            "Recall global": tp / positives,
            "Folds na meta": int((values["Recall"] >= recall_min).sum()),
            "Delta AP": (values["AP"] - reference["AP"]).mean(),
            "Delta Precision": delta_p.mean(),
            "Delta Recall": delta_r.mean(),
            "Delta FP": int((values["FP"] - reference["FP"]).sum()),
            "Folds P melhora e R preservado": int(((delta_p > 0) & (delta_r >= -1e-12)).sum()),
        })
    return pd.DataFrame(summaries)


def run_feature_analysis(
    candidates, X_train, y_train, groups_train, splits, make_preprocessor,
    *, feature_blocks=None, n_repeats=20, do_permutation=True, do_ablation=True,
    recall_min=0.55, random_state=42, n_jobs=1, output_dir="Data/feature_analysis",
):
    """Treina os candidatos nos mesmos folds e faz as duas analises.

    feature_blocks: dict nome -> lista de colunas retiradas juntas. Alem dos
    blocos, do_ablation=True testa cada coluna individualmente. O resultado
    'completo' e sempre calculado, mesmo com do_ablation=False.

    Retorna um dict com ablation_by_fold, ablation_summary,
    permutation_repeats, permutation_summary, oof_predictions e output_dir.
    Na ablacao, Delta = sem_atributo - completo. Na permutacao, Importancia
    AP = AP_original - AP_permutada. As convencoes sao diferentes.
    'FP extras' da permutacao e a media por fold e por embaralhamento;
    'FP' e 'Delta FP' da ablacao sao somas sobre as validacoes dos folds.
    Nenhum atributo e excluido automaticamente.
    """
    y, groups, splits = _validate_inputs(X_train, y_train, groups_train, splits)
    if not candidates or not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("Informe candidatos e n_repeats inteiro positivo.")
    if not 0 <= recall_min <= 1:
        raise ValueError("recall_min deve estar entre zero e um.")
    experiments = {"completo": []}
    if do_ablation:
        experiments.update({f"sem_{column}": [column] for column in X_train.columns})
        for name, columns in (feature_blocks or {}).items():
            if name in experiments or isinstance(columns, str):
                raise ValueError(f"Bloco invalido ou com nome repetido: {name!r}.")
            columns = list(columns)
            if not columns or len(set(columns)) != len(columns):
                raise ValueError(f"Bloco vazio ou com colunas repetidas: {name!r}.")
            experiments[name] = columns
    for experiment, removed in experiments.items():
        if set(removed).difference(X_train.columns) or len(removed) >= X_train.shape[1]:
            raise ValueError(f"Colunas invalidas na ablacao {experiment!r}: {removed}")
    for name, candidate in candidates.items():
        threshold = candidate.get("threshold")
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Limiar invalido para {name!r}.")
        if "preprocess" not in candidate["pipeline"].named_steps:
            raise ValueError("O pipeline precisa ter a etapa 'preprocess'.")

    run_dir = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(mkdtemp(prefix="run_", dir=str(output_dir))).resolve()
        protocol = {
            "sklearn": sklearn.__version__, "n_rows": len(X_train),
            "n_events": int(pd.Series(groups).nunique()),
            "columns": X_train.columns.tolist(),
            "dtypes": X_train.dtypes.astype(str).to_dict(),
            "recall_min": recall_min, "n_repeats": n_repeats,
            "random_state": random_state, "do_permutation": do_permutation,
            "experiments": experiments,
            "candidates": {name: {"params": item.get("params", {}),
                                  "threshold": item["threshold"],
                                  "model": repr(item["pipeline"].named_steps["model"])}
                           for name, item in candidates.items()},
            "folds": [{"fold": fold, "validation_positions": val.tolist(),
                       "events": pd.Series(groups[val]).drop_duplicates().tolist()}
                      for fold, (_, val) in enumerate(splits, 1)],
            "interpretation": "Exploracao e selecao; nao e avaliacao final independente.",
        }
        (run_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Resultados em: {run_dir}", flush=True)

    ablation_rows, permutation_rows, oof_rows = [], [], []
    results = {}
    for name, candidate in candidates.items():
        threshold = candidate["threshold"]
        print(f"{name}: {len(splits)} folds, {len(experiments)} conjuntos de atributos.", flush=True)
        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            X_fit, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            with parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(limits=1):
                # Treino completo: reaproveitado para a permutacao e referencia.
                full = clone(candidate["pipeline"]).set_params(
                    preprocess=make_preprocessor(X_train.columns.tolist()))
                full.fit(X_fit, y[train_idx])
                probabilities = _probabilities(full, X_val)
                reference = _metrics(y[val_idx], probabilities, threshold)
                ablation_rows.append({"Modelo": name, "Experimento": "completo",
                    "Fold": fold, "Limiar": threshold, "N atributos": X_train.shape[1],
                    "Atributos retirados": "[]", **reference})
                for pos, probability in zip(val_idx, probabilities):
                    oof_rows.append({"Modelo": name, "Fold": fold,
                        "row_position": int(pos), "original_index": str(X_train.index[pos]),
                        "event_id": groups[pos], "y_real": int(y[pos]),
                        "score": float(probability), "Limiar": threshold,
                        "y_previsto": int(probability >= threshold),
                        **{f"feature__{column}": X_train.iloc[pos][column]
                           for column in X_train.columns}})

                if do_permutation:
                    importance = permutation_importance(
                        full, X_val, y[val_idx],
                        scoring=partial(_permutation_scorer, threshold=threshold),
                        n_repeats=n_repeats, random_state=random_state + fold,
                        n_jobs=n_jobs,
                    )
                    for j, column in enumerate(X_train.columns):
                        for repeat in range(n_repeats):
                            permutation_rows.append({"Modelo": name, "Fold": fold,
                                "Atributo": column, "Repeticao": repeat + 1,
                                "Queda AP": importance["ap"].importances[j, repeat],
                                "Queda Precision": importance["precision"].importances[j, repeat],
                                "Queda Recall": importance["recall"].importances[j, repeat],
                                "FP extras": importance["neg_fp"].importances[j, repeat]})

                for experiment, removed in experiments.items():
                    if experiment == "completo":
                        continue
                    kept = [column for column in X_train.columns if column not in removed]
                    reduced = clone(candidate["pipeline"]).set_params(
                        preprocess=make_preprocessor(kept))
                    reduced.fit(X_fit[kept], y[train_idx])
                    metrics = _metrics(y[val_idx], _probabilities(reduced, X_val[kept]), threshold)
                    ablation_rows.append({"Modelo": name, "Experimento": experiment,
                        "Fold": fold, "Limiar": threshold, "N atributos": len(kept),
                        "Atributos retirados": json.dumps(removed), **metrics})
            print(f"  fold {fold}/{len(splits)} concluido", flush=True)

        # Checkpoint ao terminar cada candidato, preservando o ja concluido.
        results = {
            "ablation_by_fold": pd.DataFrame(ablation_rows),
            "ablation_summary": _ablation_summary(ablation_rows, recall_min),
            "permutation_repeats": pd.DataFrame(permutation_rows, columns=[
                "Modelo", "Fold", "Atributo", "Repeticao", "Queda AP",
                "Queda Precision", "Queda Recall", "FP extras"]),
            "permutation_summary": _permutation_summary(permutation_rows),
            "oof_predictions": pd.DataFrame(oof_rows),
        }
        if run_dir is not None:
            for filename, table in results.items():
                table.to_csv(run_dir / f"{filename}.csv", index=False, encoding="utf-8-sig")
    results["output_dir"] = str(run_dir) if run_dir is not None else None
    return results


def plot_feature_importance(permutation_summary):
    """Barras: importancia AP media; erro: DP das medias entre folds, nao IC."""
    import matplotlib.pyplot as plt

    if permutation_summary.empty:
        raise ValueError("A tabela de permutation importance esta vazia.")
    names = permutation_summary["Modelo"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6),
                             squeeze=False, constrained_layout=True)
    for ax, name in zip(axes.ravel(), names):
        table = permutation_summary.loc[permutation_summary["Modelo"] == name].sort_values("Importancia AP")
        ax.barh(table["Atributo"], table["Importancia AP"],
                xerr=table["DP entre folds"], capsize=2)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Queda de AP ao embaralhar")
    return fig
