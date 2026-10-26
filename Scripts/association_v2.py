"""Associação temporal com controles aleatórios + utilitários de verificação.

Conteúdo
--------
calculate_associations_random : substitui fast_association_calculator (p-valor com controles aleatórios)
null_calibration              : teste nulo (quantos falsos positivos o método produz?)
build_model_table             : substitui generate_model_table (merge por evento/fase/eixo)
plot_event_check              : gráfico de controle positivo (envelope + trem previsto)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_seconds(value):
    """UTCDateTime, datetime, Timestamp, string ISO UTC ou segundos Unix -> float."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    stamp = getattr(value, "timestamp", None)
    if stamp is not None:
        return float(stamp() if callable(stamp) else stamp)
    return pd.Timestamp(value).timestamp()


def _union_mean(times, activity, centers, half):
    """Média da atividade na UNIÃO das janelas [c-half, c+half] (sem contar amostra 2x)."""
    lo = np.searchsorted(times, centers - half, side="left")
    hi = np.searchsorted(times, centers + half, side="right")
    mask = np.zeros(len(times), bool)
    for left, right in zip(lo, hi):
        mask[left:right] = True
    n = int(mask.sum())
    return (float(activity[mask].mean()), n) if n else (np.nan, 0)


def _random_controls(
    times, activity, centers, half, blocks, n=500, rng=None, max_tries=20000
):
    """Mesmo padrão de janelas, deslocado aleatoriamente para fora de todas as zonas bloqueadas."""
    rng = rng or np.random.default_rng(42)
    lo_off = times[0] + half - centers[0]
    hi_off = times[-1] - half - centers[-1]
    ctrl, tries = [], 0
    while len(ctrl) < n and tries < max_tries and hi_off > lo_off:
        tries += 1
        c = centers + rng.uniform(lo_off, hi_off)
        if any(not (c[-1] + half < b0 or c[0] - half > b1) for b0, b1 in blocks):
            continue
        m, k = _union_mean(times, activity, c, half)
        if k:
            ctrl.append(m)
    return np.asarray(ctrl)


def event_blocks(time_data, pad_before=600, pad_after=2700):
    """Zonas proibidas para controles: [1ª chegada - pad_before, última chegada + pad_after]
    de CADA evento do catálogo. time_data deve ter TODAS as frequências do GDM52."""
    s = time_data["arrival_time"].map(_to_seconds)
    g = s.groupby(time_data["event_id"])
    return [(lo - pad_before, hi + pad_after) for lo, hi in zip(g.min(), g.max())]


def calculate_associations_random(
    diff_panel,
    time_data,
    half=36,
    n_controls=500,
    pad_before=600,
    pad_after=2700,
    min_controls=50,
    seed=0,
    labels=("Z", "X", "Y"),
):
    """diff_panel: lista de DataFrames com 'time' e 'treated_change' (envelope normalizado, SEM clipping).
    time_data: event_id, phase ('love'/'rayleigh'), arrival_time — TODAS as frequências, sem filtro de Nyquist.
    Retorna uma linha por (evento, fase, eixo) com p_value, effect_z, score e diagnósticos."""
    rng = np.random.default_rng(seed)
    arr = time_data[["event_id", "phase", "arrival_time"]].copy()
    arr["s"] = arr["arrival_time"].map(_to_seconds)
    blocks = event_blocks(time_data, pad_before, pad_after)
    rows = []
    for axis, df in zip(labels, diff_panel):
        t = np.fromiter((_to_seconds(x) for x in df["time"]), float)
        a = np.abs(df["treated_change"].to_numpy(float))
        ok = np.isfinite(t) & np.isfinite(a)
        t, a = t[ok], a[ok]
        order = np.argsort(t, kind="stable")
        t, a = t[order], a[order]
        for (ev, ph), g in arr.groupby(["event_id", "phase"], sort=False):
            c = np.sort(g["s"].to_numpy(float))
            obs, n_obs = _union_mean(t, a, c, half)
            ctrl = (
                _random_controls(t, a, c, half, blocks, n_controls, rng)
                if n_obs
                else np.array([])
            )
            base = dict(
                event_id=ev,
                phase=ph,
                axis=axis,
                activity_mean=obs,
                n_samples=n_obs,
                n_controls=len(ctrl),
            )
            if len(ctrl) < min_controls:
                rows.append(
                    {
                        **base,
                        "temporal_association_score": np.nan,
                        "p_value": np.nan,
                        "effect_z": np.nan,
                        "status": "insufficient_controls" if n_obs else "no_samples",
                    }
                )
                continue
            sd = ctrl.std(ddof=1)
            rows.append(
                {
                    **base,
                    "temporal_association_score": float(np.mean(ctrl < obs)),
                    "p_value": (1 + np.sum(ctrl >= obs)) / (1 + len(ctrl)),
                    "effect_z": (obs - ctrl.mean()) / sd if sd > 0 else np.nan,
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def null_calibration(
    times, activity, patterns, blocks, half=36, n_fake=200, n_controls=300, seed=0
):
    """Coloca padrões reais em lugares SEM evento e mede quantos dão p pequeno.
    Método honesto => ~5% com p<=0.05 e ~10% com p<=0.10.
    patterns: lista de arrays ordenados de chegadas (s), um por (evento, fase)."""
    rng = np.random.default_rng(seed)
    ps = []
    for i in range(n_fake):
        c0 = patterns[i % len(patterns)]
        lo_off = times[0] + half - c0[0]
        hi_off = times[-1] - half - c0[-1]
        for _ in range(200):
            c = c0 + rng.uniform(lo_off, hi_off)
            if all(c[-1] + half < b0 or c[0] - half > b1 for b0, b1 in blocks):
                break
        else:
            continue
        obs, _ = _union_mean(times, activity, c, half)
        extra = [(c[0] - half - 60, c[-1] + half + 60)]
        ctrl = _random_controls(
            times, activity, c, half, blocks + extra, n_controls, rng
        )
        if len(ctrl) >= 50:
            ps.append((1 + np.sum(ctrl >= obs)) / (1 + len(ctrl)))
    ps = np.asarray(ps)
    return {
        "n": len(ps),
        "frac_p_le_005": float(
            np.mean(ps <= 0.05) if (ps <= 0.05).size != 0 else np.nan
        ),
        "frac_p_le_010": float(
            np.mean(ps <= 0.10) if (ps <= 0.10).size != 0 else np.nan
        ),
    }


def build_model_table(
    association_table, intensity_table, event_data, file_name, dt, band_hz
):
    """Substitui generate_model_table: só Love/Rayleigh, merge por (evento, fase, eixo),
    com rastreabilidade (arquivo, dt, banda) e descarte de linhas sem controles válidos."""
    surface = association_table["phase"].isin(["love", "rayleigh"])
    scores = association_table[(association_table["event_id"] != "all") & surface]
    intensity = intensity_table[
        ["event_id", "phase", "axis", "activity_order"]
    ].drop_duplicates(["event_id", "phase", "axis"])
    events = event_data[
        ["event_id", "magnitude", "magnitude_type", "epicenter_distance", "depth"]
    ]
    table = (
        scores[
            [
                "event_id",
                "phase",
                "axis",
                "temporal_association_score",
                "p_value",
                "effect_z",
                "n_samples",
                "n_controls",
                "status",
            ]
        ]
        .rename(columns={"temporal_association_score": "score"})
        .merge(
            intensity,
            on=["event_id", "phase", "axis"],
            how="left",
            validate="one_to_one",
        )
        .merge(events, on="event_id", how="left", validate="many_to_one")
    )
    table["file"] = file_name
    table["dt"] = dt
    table["band_hz"] = f"{band_hz[0]:.4f}-{band_hz[1]:.4f}"
    return table[table["status"] == "ok"].reset_index(drop=True)


def plot_event_check(
    times, activity, time_data, event_id, origin_s=None, ax=None, title=""
):
    """Controle positivo: envelope normalizado + trem previsto (Love e Rayleigh, todas as frequências)."""
    ax = ax or plt.subplots(figsize=(11, 3.5))[1]
    t0 = times[0]
    ax.plot((times - t0) / 60, activity, lw=0.8, color="k")
    ax.axhline(1, color="gray", ls=":", lw=0.8)
    d = time_data[time_data["event_id"] == event_id]
    for ph, color in [("love", "tab:blue"), ("rayleigh", "tab:red")]:
        s = d[d["phase"] == ph]["arrival_time"].map(_to_seconds)
        if len(s):
            ax.axvspan(
                (s.min() - t0) / 60,
                (s.max() - t0) / 60,
                color=color,
                alpha=0.25,
                label=f"{ph}: trem previsto",
            )
    if origin_s is not None:
        ax.axvline((origin_s - t0) / 60, color="green", ls="--", label="origem")
    ax.set_xlabel("minutos desde o início do registro")
    ax.set_ylabel("envelope / mediana local")
    ax.set_title(title or event_id[:40])
    ax.legend(loc="upper left")
    return ax
