"""Ajuste final dos candidatos ja escolhidos no projeto Sirius.

Nao executa busca, ablacao ou treinamento ao importar. Recebe o dicionario
`candidates` usado em feature_analysis.py, com pipeline, threshold e params.
Cada pipeline e clonado, ajustado SOMENTE em X_train/y_train e avaliado no
teste reservado com o limiar ja escolhido. O teste nao escolhe nada.

Uso em model_comparator(), depois de obter candidates e antes da ablacao:

    final = fit_final_models(
        candidates, X_train, y_train, X_test, y_test,
        output_dir="Data/final_models",
    )
    print(final["metrics"].round(3).to_string(index=False))
    print(final["logistic_weights"]["logistic_regression"].to_string(index=False))
    print(final["logistic_intercepts"]["logistic_regression"])
    return final  # encerra antes dos blocos antigos de busca/ablacao

Os coeficientes correspondem aos atributos APOS o preprocessamento: uma
unidade de uma coluna numerica padronizada e um desvio padrao do treino.
Para one-hot, compare coeficientes de categorias da mesma variavel.
Nao sao efeitos causais nem importancias independentes dos outros atributos.

Os .joblib guardam pipeline completo, limiar, ordem das colunas e metadados.
`predict_final()` aplica esse limiar tambem em novos dados. A funcao nao usa
pipeline.predict(), pois isso descartaria os cortes ajustados de 0.45/0.35.
"""

from __future__ import annotations

import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def logistic_weights(fitted_pipeline):
    """Retorna (tabela_de_pesos, intercepto) do logistic binario ja ajustado.

    A ordem dos nomes e a ordem REAL da matriz transformada que o modelo viu.
    A tabela e ordenada pelo valor absoluto, mantendo o sinal do coeficiente.
    Intercepto e coeficientes usam a mesma escala transformada.
    """
    model = fitted_pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        raise TypeError("O passo 'model' precisa ser LogisticRegression.")
    check_is_fitted(model)
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Esta funcao interpreta a classe positiva 1, com y em 0/1.")

    names = fitted_pipeline[:-1].get_feature_names_out()
    coefficients = np.asarray(model.coef_).reshape(-1)
    if len(names) != len(coefficients):
        raise ValueError("Os nomes transformados nao correspondem aos coeficientes.")

    table = pd.DataFrame({
        "Atributo transformado": names,
        "Coeficiente": coefficients,
        "Valor absoluto": np.abs(coefficients),
    }).sort_values("Valor absoluto", ascending=False, kind="stable")
    return table.reset_index(drop=True), float(model.intercept_[0])


def predict_final(bundle, X):
    """Aplica um resultado retornado/salvo por fit_final_models em novos dados.

    bundle pode ser final['models'][nome] ou joblib.load(caminho_do_modelo).
    Retorna probabilidade da classe 1 e classe prevista no limiar armazenado.
    """
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise TypeError("X deve ser DataFrame com nomes de colunas distintos.")
    missing = set(bundle["features"]).difference(X.columns)
    if missing:
        raise ValueError(f"Atributos ausentes: {sorted(missing)}")
    pipeline = bundle["pipeline"]
    check_is_fitted(pipeline)
    positive = np.flatnonzero(np.asarray(pipeline.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("O modelo nao reconhece a classe positiva 1.")

    probabilities = pipeline.predict_proba(
        X.loc[:, bundle["features"]]
    )[:, positive[0]]
    predicted = (probabilities >= bundle["threshold"]).astype(int)
    return pd.DataFrame({
        "Probabilidade": probabilities,
        "Predicao": predicted,
    }, index=X.index)


def _validate_xy(X, y, name, require_both_classes=False):
    if (not isinstance(X, pd.DataFrame) or X.empty
            or not X.columns.is_unique
            or not all(isinstance(c, str) for c in X.columns)):
        raise ValueError(f"{name}: X deve ter linhas e colunas com nomes distintos.")
    if isinstance(y, (pd.Series, pd.DataFrame)) and not X.index.equals(y.index):
        raise ValueError(f"{name}: indices de X e y desalinhados.")
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != len(X) or pd.isna(values).any():
        raise ValueError(f"{name}: y deve ser um vetor alinhado e sem valores ausentes.")
    classes = set(np.unique(values))
    if not classes.issubset({0, 1}) or (require_both_classes and classes != {0, 1}):
        raise ValueError(f"{name}: use rotulos 0/1; o treino precisa das duas classes.")
    return values.astype(int)


def _metrics(y, probabilities, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        # Sem as duas classes, nao reportar AP como avaliacao de discriminacao.
        "AP": (float(average_precision_score(y, probabilities))
               if len(np.unique(y)) == 2 else np.nan),
        "Precision": float(precision_score(y, predicted, zero_division=0)),
        "Recall": float(recall_score(y, predicted, zero_division=0)),
        "FP": int(fp), "TP": int(tp), "FN": int(fn), "TN": int(tn),
    }


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def fit_final_models(
    candidates, X_train, y_train, X_test, y_test, *,
    output_dir="Data/final_models", groups_train=None, groups_test=None,
    target_definition="p_value <= 0.1",
):
    """Treina, avalia e salva os candidatos; devolve modelos e tabelas.

    candidates[nome] = {'pipeline': pipeline, 'threshold': corte,
                        'params': parametros_ja_escolhidos}
    'params' e opcional se o pipeline ja contem todos os parametros corretos.
    Pode receber somente os tres modelos escolhidos, na ordem desejada.

    Mantenha o MESMO split agrupado por event_id usado ate aqui. Se fornecer
    groups_train e groups_test, a funcao tambem verifica a separacao de eventos.
    Nao refaz o split nem altera atributos, parametros ou limiares pelo teste.

    Retorno:
      models: bundles com pipeline ajustado, threshold e features;
      metrics: metricas no teste (uma linha por modelo);
      predictions: previsoes do teste, por posicao e indice original;
      logistic_weights/logistic_intercepts: tabelas e interceptos;
      model_paths/output_dir: caminhos dos arquivos criados.

    Os CSVs de teste agregam todas as linhas, nao medias entre folds.
    Cada chamada cria uma subpasta nova, sem sobrescrever execucoes anteriores.
    """
    train_y = _validate_xy(X_train, y_train, "Treino", require_both_classes=True)
    test_y = _validate_xy(X_test, y_test, "Teste")
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError("Treino e teste precisam das mesmas colunas de entrada.")
    if not candidates:
        raise ValueError("Informe os candidatos finais ja escolhidos.")

    # Validar todos os candidatos antes de iniciar qualquer ajuste.
    prepared = {}
    for name, spec in candidates.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Os modelos precisam de nomes nao vazios.")
        threshold = float(spec["threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Limiar invalido para {name}.")
        pipeline = clone(spec["pipeline"]).set_params(**spec.get("params", {}))
        if not isinstance(pipeline, Pipeline) or "model" not in pipeline.named_steps:
            raise TypeError(f"{name}: use Pipeline com passo final 'model'.")
        if pipeline.steps[-1][0] != "model" or not hasattr(pipeline, "predict_proba"):
            raise TypeError(f"{name}: o classificador final precisa de predict_proba.")
        prepared[name] = (pipeline, threshold, dict(spec.get("params", {})))

    event_counts = None
    if (groups_train is None) != (groups_test is None):
        raise ValueError("Forneca groups_train e groups_test juntos, ou omita ambos.")
    if groups_train is not None:
        arrays = []
        for groups, X, partition in [(groups_train, X_train, "treino"),
                                      (groups_test, X_test, "teste")]:
            if isinstance(groups, pd.Series) and not X.index.equals(groups.index):
                raise ValueError(f"event_id desalinhado no {partition}.")
            values = np.asarray(groups)
            if values.ndim != 1 or len(values) != len(X) or pd.isna(values).any():
                raise ValueError(f"event_id invalido no {partition}.")
            arrays.append(values)
        if set(arrays[0]).intersection(arrays[1]):
            raise ValueError("Ha event_id presente tanto no treino como no teste.")
        event_counts = {"train": len(set(arrays[0])), "test": len(set(arrays[1]))}

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="final_", dir=root)).resolve()
    features = list(X_train.columns)
    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "joblib": joblib.__version__}
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_definition": target_definition,
        "features": features, "n_train": len(X_train), "n_test": len(X_test),
        "n_positive_train": int(train_y.sum()), "n_positive_test": int(test_y.sum()),
        "event_counts": event_counts, "versions": versions,
        "train_index": X_train.index.tolist(), "test_index": X_test.index.tolist(),
        "candidates": {name: {"threshold": threshold, "selected_params": params,
                              "pipeline": repr(pipeline)}
                       for name, (pipeline, threshold, params) in prepared.items()},
    }
    if event_counts is not None:
        protocol["groups_train"] = arrays[0].tolist()
        protocol["groups_test"] = arrays[1].tolist()
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, default=_json_value),
        encoding="utf-8",
    )

    fitted, rows, predictions, weights, intercepts, paths = {}, [], [], {}, {}, {}
    for number, (name, (pipeline, threshold, params)) in enumerate(prepared.items(), 1):
        # O scaler, o encoder e o classificador aprendem somente no treino.
        pipeline.fit(X_train, train_y)

        bundle = {"pipeline": pipeline, "threshold": threshold,
                  "features": features, "selected_params": params,
                  "target_definition": target_definition, "versions": versions}
        test_predictions = predict_final(bundle, X_test)
        probabilities = test_predictions["Probabilidade"].to_numpy()
        predicted = test_predictions["Predicao"].to_numpy()
        rows.append({"Modelo": name, "Limiar": threshold,
                     **_metrics(test_y, probabilities, predicted)})

        table = pd.DataFrame({
            "Modelo": name, "Posicao teste": np.arange(len(X_test)),
            "Indice original": X_test.index.to_list(), "Observado": test_y,
            "Probabilidade": probabilities, "Predicao": predicted,
            "Limiar": threshold,
        })
        if event_counts is not None:
            table["event_id"] = arrays[1]
        predictions.append(table)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "modelo"
        filename = f"{number:02d}_{slug}"

        if isinstance(pipeline.named_steps["model"], LogisticRegression):
            weights[name], intercepts[name] = logistic_weights(pipeline)
            weights[name].to_csv(output / f"{filename}_weights.csv", index=False)

        path = output / f"{filename}.joblib"
        joblib.dump(bundle, path)
        fitted[name], paths[name] = bundle, str(path)
        # Preservar os resultados ja concluidos caso um modelo seguinte falhe.
        pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(
            output / "test_predictions.csv", index=False,
        )
        (output / "logistic_intercepts.json").write_text(
            json.dumps(intercepts, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    return {"models": fitted, "metrics": pd.DataFrame(rows),
            "predictions": pd.concat(predictions, ignore_index=True),
            "logistic_weights": weights, "logistic_intercepts": intercepts,
            "model_paths": paths, "output_dir": str(output)}
