import os
import warnings
from datetime import UTC, datetime
from zipfile import BadZipFile

import arrival_time as arrt
import matplotlib.pyplot as plt
import numpy as np
import obspy.core.utcdatetime as utc
import pandas as pd
import plotly.graph_objects as go
from association_v2 import (
    _to_seconds,
    calculate_associations_random,
    event_blocks,
    null_calibration,
)
from fast_score import fast_association_calculator
from obspy import taup
from obspy.clients.fdsn.header import FDSNNoDataException
from scipy.signal import butter, hilbert, sosfiltfilt

warnings.filterwarnings("error", category=RuntimeWarning)


def normalized_envelope(t, val, fmin, fmax, noise_window_s=4 * 3600):
    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    x = np.asarray(val, float)
    x = x - np.nanmedian(x)

    sos = butter(4, [fmin, fmax], btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))

    n = max(int(noise_window_s / dt), 3)
    noise = (
        pd.Series(env).rolling(n, center=True, min_periods=n // 4).median().to_numpy()
    )

    return env / noise, noise


def change_amplitude(data_seism: pd.DataFrame, frequencies: tuple, THRESHOLD=3):
    """
    Function that computes the amplitude variation of a seismic signal and identifies significant changes

    Input:

    - data_seism: dataframe containing the seismic signal values and their respective timestamps

    - THRESHOLD: the minimum amplitude variation required to consider a change as significant

    Output:

    - A dataframe containing the amplitude variation between consecutive samples, the noise-reduced variation, the time interval between samples, and boolean flags indicating whether each variation was detected as a significant change
    """

    seism_list = np.array(data_seism["val"].to_list())
    temporal_list = data_seism["secs"].to_list()

    treated_change, noise = normalized_envelope(
        temporal_list, seism_list, frequencies[0], frequencies[1]
    )

    change_detected = [float("NaN")]
    treated_change_detected = [float("NaN")]

    for change in abs(seism_list[1:]):
        if change > THRESHOLD:
            change_detected.append(True)
        else:
            change_detected.append(False)

    for change in abs(treated_change[1:]):
        if change > THRESHOLD:
            treated_change_detected.append(True)
        else:
            treated_change_detected.append(False)

    df = pd.DataFrame(
        {
            "time": temporal_list,
            "change": seism_list,
            "treated_change": treated_change,
            "delta_time": np.insert(np.diff(temporal_list), 0, float("NaN")),
            "detected": change_detected,
            "treated_detected": treated_change_detected,
        }
    )

    return df, noise


def detecting_earthquake(
    diff_panel: np.asarray,
    intensity_table: pd.DataFrame,
    time_data: pd.DataFrame,
    event_data: pd.DataFrame,
    sensibility=10.2,
):
    """
    Function that evaluates the statistical association between predicted
    earthquake arrival times and seismic activity changes detected in a
    set of seismic channels

    Input:

    - diff_panel: list of dataframes containing the processed seismic
    changes for each channel. Each dataframe must contain at least
    `time` and `treated_change` columns

    - intensity_table: dataframe containing the activity metrics
    associated with each earthquake-phase combination. It must contain
    the columns `event_id`, `phase`, and `activity_order`

    - time_data: dataframe containing the predicted earthquake arrival
    information. It must contain the columns `event_id` and
    `arrival_time`

    - event_data: dataframe containing the earthquake event metadata.
    It must contain the columns `event_id`, `magnitude`,
    `epicenter_distance`, and `depth`

    - sensibility: half-width of the temporal window used to evaluate
    the seismic activity around each predicted arrival

    Output:

    - detection_table: dataframe containing summary activity metrics
    for each earthquake event, including magnitude, depth, epicentral
    distance, activity statistics, and the fraction of associated
    detections

    - temporal_association_score: association score calculated for the
    last analyzed channel, representing how frequently the observed
    activity around predicted arrivals exceeds the activity obtained
    from temporally shifted arrivals

    - association_table: dataframe containing the temporal association
    score for each event and channel, as well as the score obtained
    considering all arrivals simultaneously

    The function compares seismic activity observed near predicted
    arrival times with activity measured after applying multiple time
    shifts to the arrivals. The resulting association score provides a
    statistical indicator of whether the observed activity is
    temporally consistent with the earthquake predictions.
    """

    detection_table = pd.DataFrame(
        {
            "event_id": [],
            "magnitude": [],
            "epicenter_distance": [],
            "depth": [],
            "fraction_above_reference": [],
            "greatest_order": [],
            "greatest_phase": [],
            "mean_probable_activity_order": [],
            "mean_event_activity_order": [],
        }
    )

    for idx, row in event_data.iterrows():
        name = row["event_id"]
        mag = row["magnitude"]
        ep_distance = row["epicenter_distance"]
        depth = row["depth"]

        for axis in ["Z", "X", "Y"]:
            available_earthquakes = intensity_table[
                (intensity_table["event_id"] == name)
                & (intensity_table["axis"] == axis)
            ]
            probable_earthquakes = available_earthquakes[
                available_earthquakes["activity_order"] > 1
            ]

            n_total = len(available_earthquakes)

            correlation_percentil = (
                len(probable_earthquakes) / n_total if n_total > 0 else 0
            )

            greatest_order = (
                available_earthquakes["activity_order"].to_list()[0]
                if available_earthquakes["activity_order"].to_list() != []
                else 0
            )
            greatest_phase = (
                available_earthquakes["phase"].to_list()[0]
                if available_earthquakes["phase"].to_list() != []
                else "None"
            )
            mean_probable_activity_order = (
                np.mean(probable_earthquakes["activity_order"].to_list())
                if not probable_earthquakes["activity_order"].empty
                else np.nan
            )
            mean_event_activity_order = (
                np.mean(available_earthquakes["activity_order"].to_list())
                if not available_earthquakes["activity_order"].empty
                else np.nan
            )

            detection_table.loc[len(detection_table)] = [
                name,
                mag,
                ep_distance,
                depth,
                correlation_percentil,
                greatest_order,
                greatest_phase,
                mean_probable_activity_order,
                mean_event_activity_order,
            ]

    association_table = pd.DataFrame(
        {
            "event_id": [],
            "phase": [],
            "axis": [],
            "temporal_association_score": [],
            "shift_upper_limit": [],
            "shift_lower_limit": [],
            "step": [],
        }
    )

    association_table = calculate_associations_random(
        diff_panel,
        time_data,
    )

    return detection_table, association_table


def correlations(
    diff_content_panel: np.asarray,
    time_data: pd.DataFrame,
    event_data: pd.DataFrame,
    sensibility=10.2,
):
    """
    Function that evaluates the activity intensity associated with
    predicted earthquake arrivals by comparing seismic changes inside
    and outside temporal windows around each arrival

    Input:

    - diff_content_panel: list of processed seismic change dataframes.
    The first dataframe must contain the columns `time` and
    `treated_change`

    - time_data: dataframe containing the predicted earthquake arrival
    information. It must contain the columns `event_id`, `phase`,
    and `arrival_time`

    - event_data: dataframe containing the earthquake event metadata.
    It must contain at least the columns `event_id` and `magnitude`

    - sensibility: half-width of the temporal window centered on each
    predicted arrival time used to evaluate seismic activity

    Output:

    - intensity_table: dataframe containing the activity statistics
    associated with each event-phase combination. The output includes:

        - event_id: earthquake identifier

        - phase: seismic phase associated with the arrival

        - magnitude: earthquake magnitude

        - activity_order: ratio between the mean activity inside the
        arrival window and the mean activity outside the window

        - activity_mean: mean activity measured inside the arrival
        window

        - sensibility: temporal tolerance used for the analysis

    The function creates a temporal window around each predicted
    arrival and compares the seismic activity observed inside and
    outside this window. The resulting activity ratio is used as an
    indicator of the correlation between the predicted arrival and
    the observed seismic signal.
    """

    names = event_data["event_id"].to_list()

    diff_df_Z = diff_content_panel[0].dropna()
    diff_df_X = diff_content_panel[1].dropna()
    diff_df_Y = diff_content_panel[2].dropna()

    diff_table_list = [diff_df_Z, diff_df_X, diff_df_Y]
    label = ["Z", "X", "Y"]

    table = []

    for j, diff_table in enumerate(diff_table_list):
        for name in names:
            for i, phase in enumerate(
                np.unique(time_data[time_data["event_id"] == name]["phase"].to_list())
            ):
                arrivals = time_data[
                    (time_data["event_id"] == name) & (time_data["phase"] == phase)
                ]["arrival_time"].to_list()
                arrivals = np.sort(arrivals)

                intervals = []

                for arrival in arrivals:
                    intervals.append(
                        [
                            arrival - sensibility,
                            arrival + sensibility,
                        ]
                    )
                previsions = []

                for time in diff_table["time"].to_list():
                    previsions.append(
                        any(start <= time <= finish for start, finish in intervals)
                    )

                table.append(
                    {
                        "name": name,
                        "axis": label[j],
                        "magnitude": event_data.loc[
                            event_data["event_id"] == name, "magnitude"
                        ].iloc[0],
                        "phase": phase,
                        "time": diff_table["time"].to_list(),
                        "change": diff_table["treated_change"].to_list(),
                        "prevision": previsions,
                        "activity": [
                            abs(x) for x in diff_table["treated_change"].to_list()
                        ],
                        "first_arrival": arrivals[0],
                        "last_arrival": arrivals[-1],
                    }
                )

    intensity_table = pd.DataFrame(
        {
            "event_id": [],
            "axis": [],
            "phase": [],
            "magnitude": [],
            "activity_order": [],
            "activity_mean": [],
            "sensibility": [],
            "first_arrival": [],
            "last_arrival": [],
        }
    )

    for row in table:
        name = row["name"]
        axis = row["axis"]
        magnitude = row["magnitude"]
        phase = row["phase"]
        previsions = row["prevision"]
        activity = row["activity"]
        first_arrival = row["first_arrival"]
        last_arrival = row["last_arrival"]

        mask = np.array(previsions, dtype=bool)
        values = np.array(activity)

        mean_1 = (values[mask].mean()) if len(values[mask]) != 0 else 0
        mean_2 = (values[~mask].mean()) if len(values[~mask]) != 0 else 0

        intensity_table.loc[len(intensity_table)] = [
            name,
            axis,
            phase,
            magnitude,
            (mean_1 / mean_2) if mean_2 != 0 else float("Nan"),
            mean_1,
            sensibility,
            first_arrival,
            last_arrival,
        ]

    intensity_table = intensity_table.sort_values("activity_order", ascending=False)

    return intensity_table


def table_treatment(table: pd.DataFrame, time_zone=-3):
    """
    Function that converts the time information of a seismic dataset
    to UTCDateTime objects for temporal analysis

    Input:

    - table: dataframe containing the seismic information. It must
    contain a `secs` column with timestamps expressed as Unix
    timestamps

    - time_zone: timezone offset used in previous versions of the
    conversion process (discontinued)

    Output:

    - A copy of the input dataframe with the `secs` column converted
    to ObsPy UTCDateTime objects

    The function transforms the timestamps stored in the `secs`
    column into UTC ISO-formatted strings and subsequently converts
    them to UTCDateTime objects, allowing direct use in ObsPy-based
    time calculations and comparisons.
    """

    df = table.copy()

    # df["secs"] = (
    #     df["secs"]
    #     .str.replace(", ", "T", regex=False)
    #     .str.replace(r"$", "00000Z", regex=True)
    #     .str.replace("/", "-", regex=False)
    # )

    # df["secs"] = df["secs"].apply(
    #     lambda s: "-".join(s.split("T")[0].split("-")[::-1]) + "T" + s.split("T")[1]
    # )

    # df["secs"] = df["secs"].apply(utc.UTCDateTime)
    # df["secs"] = df["secs"].apply(lambda t: t - time_zone * 3600)

    df["secs"] = df["secs"].apply(
        lambda s: datetime.fromtimestamp(s, tz=UTC).isoformat()
    )
    df["secs"] = df["secs"].apply(utc.UTCDateTime)

    return df


def plot_seismo(
    content_panel: np.asarray,
    diff_content_panel: np.asarray,
    region_df: np.asarray,
    time_data: pd.DataFrame,
    superficial_waves: pd.DataFrame = None,
):
    """
    Function that generates an interactive visualization of seismic waveform
    data, amplitude variation measurements, detected dynamic regions, and
    predicted earthquake arrival times.

    Input:

    - content_panel: list or array containing the seismogram dataframes for
    the three seismic components (Z, X, and Y). Each dataframe must
    contain the columns `x` and `y`

    - diff_content_panel: list or array containing the amplitude variation
    dataframes associated with each seismic component. Each dataframe must
    contain the columns `time` and `change`

    - region_df: list or array containing the detected dynamic region
    dataframes for the three components. Each dataframe must contain the
    columns `start` and `end`

    - time_data: dataframe containing earthquake arrival information. It
    must contain the columns `event_id` and `arrival_time`

    Output:

    - An interactive Plotly figure displaying the seismic waveforms of the
    Z, X, and Y components

    - The amplitude variation curves associated with each component

    - Vertical dashed lines representing predicted earthquake arrival times,
    grouped by event identifier and individually toggleable through the
    legend

    - Colored shaded regions highlighting intervals of detected dynamic
    activity for each component:

        - Red: Z component regions

        - Blue: X component regions

        - Yellow: Y component regions

    The function combines waveform, amplitude variation, event arrival, and
    region detection information into a single interactive visualization,
    allowing qualitative comparison between seismic activity, detected
    signal changes, and predicted earthquake arrivals.
    """

    df_Z = content_panel[0]
    df_X = content_panel[1]
    df_Y = content_panel[2]

    diff_df_Z = diff_content_panel[0]
    diff_df_X = diff_content_panel[1]
    diff_df_Y = diff_content_panel[2]

    region_df_Z = region_df[0]
    region_df_X = region_df[1]
    region_df_Y = region_df[2]

    fig = go.Figure()

    fig.add_scatter(x=df_Z["secs"].to_list(), y=df_Z["val"].to_list(), name="Curve Z")
    fig.add_scatter(x=df_X["secs"].to_list(), y=df_X["val"].to_list(), name="Curve X")
    fig.add_scatter(x=df_Y["secs"].to_list(), y=df_Y["val"].to_list(), name="Curve Y")

    fig.add_scatter(
        x=diff_df_Z["time"].to_list(),
        y=diff_df_Z["change"].to_list(),
        name="Diff Curve Z",
    )
    fig.add_scatter(
        x=diff_df_X["time"].to_list(),
        y=diff_df_X["change"].to_list(),
        name="Diff Curve X",
    )
    fig.add_scatter(
        x=diff_df_Y["time"].to_list(),
        y=diff_df_Y["change"].to_list(),
        name="Diff Curve Y",
    )
    fig.add_scatter(
        x=diff_df_Z["time"].to_list(),
        y=diff_df_Z["treated_change"].to_list(),
        name="Treated Diff Curve Z",
    )
    fig.add_scatter(
        x=diff_df_X["time"].to_list(),
        y=diff_df_X["treated_change"].to_list(),
        name="Treated Diff Curve X",
    )
    fig.add_scatter(
        x=diff_df_Y["time"].to_list(),
        y=diff_df_Y["treated_change"].to_list(),
        name="Treated Diff Curve Y",
    )

    u_max = np.max(
        np.concatenate(
            [
                df_Z["val"].to_list(),
                df_X["val"].to_list(),
                df_Y["val"].to_list(),
                diff_df_Z["treated_change"].to_list(),
                diff_df_X["treated_change"].to_list(),
                diff_df_Y["treated_change"].to_list(),
            ]
        )
    )
    u_min = np.min(
        np.concatenate(
            [
                df_Z["val"].to_list(),
                df_X["val"].to_list(),
                df_Y["val"].to_list(),
                diff_df_Z["treated_change"].to_list(),
                diff_df_X["treated_change"].to_list(),
                diff_df_Y["treated_change"].to_list(),
            ]
        )
    )

    for name in np.unique(time_data["event_id"].to_list()):
        time_list = time_data[time_data["event_id"] == name]["arrival_time"].to_list()

        for i, j in enumerate(time_list):
            fig.add_trace(
                go.Scatter(
                    x=[j, j],
                    y=[u_min, u_max],
                    mode="lines",
                    line=dict(color="purple", width=2, dash="dash"),
                    name=name,
                    legendgroup=name,
                    showlegend=(i == 0),
                )
            )

    if superficial_waves is not None:
        for typing in np.unique(superficial_waves["type"].to_list()):
            time_list = superficial_waves[superficial_waves["type"] == typing][
                "arrival_time"
            ].to_list()

            for i, j in enumerate(time_list):
                fig.add_trace(
                    go.Scatter(
                        x=[j, j],
                        y=[u_min, u_max],
                        mode="lines",
                        line=dict(color="purple", width=2, dash="dash"),
                        name=typing,
                        legendgroup=typing,
                        showlegend=(i == 0),
                    )
                )

    for idx, row in region_df_Z.iterrows():
        fig.add_vrect(
            x0=row["start"],
            x1=row["end"],
            fillcolor="red",
            opacity=0.2,
            layer="below",
            name="Z regions",
            legendgroup="Z regions",
            line_width=0,
            showlegend=(idx == 0),
        )
    for idx, row in region_df_X.iterrows():
        fig.add_vrect(
            x0=row["start"],
            x1=row["end"],
            fillcolor="blue",
            opacity=0.2,
            layer="below",
            name="X regions",
            legendgroup="X regions",
            line_width=0,
            showlegend=(idx == 0),
        )
    for idx, row in region_df_Y.iterrows():
        fig.add_vrect(
            x0=row["start"],
            x1=row["end"],
            fillcolor="yellow",
            opacity=0.2,
            layer="below",
            name="Y regions",
            legendgroup="Y regions",
            line_width=0,
            showlegend=(idx == 0),
        )

    fig.show()


def load_catalog(df: pd.DataFrame):
    """
    Function that loads earthquake events from the ISC catalog for the time
    interval covered by a seismic trace and computes the corresponding
    arrival times.

    Input:

    - df: dataframe containing the seismic trace. It must contain an `x`
    column with time values

    Output:

    - event_data: dataframe containing the information of the retrieved
    earthquake events

    - time_data: dataframe containing the predicted arrival times and
    associated event identifiers. The `arrival_time` column is converted
    to `UTCDateTime` objects for further processing and plotting
    """
    start = df["secs"].to_list()[0] - 2 * 60 * 60
    end = df["secs"].to_list()[-1] + 40 * 60

    try:
        catalog = "USP"
        client = arrt.Client(catalog)
        events = client.get_events(
            starttime=start,
            endtime=end,
            minmagnitude=4.0,
        )
    except FDSNNoDataException:
        return None, None

    event_data, time_data = arrt.analise_catalog(events, catalog)
    time_data["arrival_time"] = time_data["arrival_time"].apply(utc.UTCDateTime)

    return event_data, time_data


def group_dynamic(diff: pd.DataFrame):
    """
    Function that identifies continuous groups of detected changes in a
    dynamic signal and computes descriptive statistics for each group.

    Input:

    - diff: dataframe containing the dynamic change information. It must
    contain the columns `time`, `detected`, and `change`

    Output:

    - groups_data: dataframe containing the information of each detected
    group. The output includes:

        - start: start time of the detected group

        - end: end time of the detected group

        - n_changes: number of consecutive detected samples within the group

        - mean_amplitudes: mean value of the `change` amplitudes inside the
        detected group

    The function searches for transitions in the `detected` column,
    identifying when a detection starts and ends. For each detected
    interval, it calculates the duration in samples and the average change
    amplitude, storing the results in a summary dataframe
    """

    groups_data = pd.DataFrame(
        {
            "start": [],
            "end": [],
            "n_changes": [],
            "mean_activity": [],
            "mean_amplitudes": [],
        }
    )

    status = False
    groups = []

    for i, value in enumerate(diff["treated_detected"].to_list()):
        if abs(status - value) == 1:
            status = value
            groups.append((i, diff["time"].to_list()[i]))

        if len(groups) == 2:
            start = groups[0][1]
            end = groups[1][1]
            n_changes = groups[1][0] - groups[0][0]
            mean_activity = np.mean(
                [
                    abs(x)
                    for x in diff["treated_change"].to_list()[
                        groups[0][0] : groups[1][0]
                    ]
                ]
            )
            mean_amplitude = np.mean(
                diff["treated_change"].to_list()[groups[0][0] : groups[1][0] + 1]
            )

            groups_data.loc[len(groups_data)] = [
                start,
                end,
                n_changes,
                mean_activity,
                mean_amplitude,
            ]

            groups = []

    if status:
        groups.append((i, diff["time"].to_list()[i]))
        start = groups[0][1]
        end = groups[1][1]
        n_changes = groups[1][0] - groups[0][0]
        mean_activity = np.mean(
            [
                abs(x)
                for x in diff["treated_change"].to_list()[groups[0][0] : groups[1][0]]
            ]
        )
        mean_amplitude = np.mean(
            diff["treated_change"].to_list()[groups[0][0] : groups[1][0] + 1]
        )

        groups_data.loc[len(groups_data)] = [
            start,
            end,
            n_changes,
            mean_activity,
            mean_amplitude,
        ]

    if status:
        groups.append((i, diff["time"].to_list()[i]))
        start = groups[0][1]
        end = groups[1][1]
        n_changes = groups[1][0] - groups[0][0]
        mean_activity = np.mean(
            [
                abs(x)
                for x in diff["treated_change"].to_list()[groups[0][0] : groups[1][0]]
            ]
        )
        mean_amplitude = np.mean(
            diff["treated_change"].to_list()[groups[0][0] : groups[1][0] + 1]
        )

        groups_data.loc[len(groups_data)] = [
            start,
            end,
            n_changes,
            mean_activity,
            mean_amplitude,
        ]

    return groups_data


def save_excel(
    content_panel: np.asarray,
    diff_content_panel: np.asarray,
    regions_content_panel: np.asarray,
    time_data: pd.DataFrame,
    event_data: pd.DataFrame,
    detection_table: pd.DataFrame,
    intensity_table: pd.DataFrame,
    association_table: pd.DataFrame,
    model_table: pd.DataFrame,
    path_name: str = "Data/resultado.xlsx",
):

    filtered_events = event_data[event_data["event_id"].isin(time_data["event_id"])]

    if not os.path.exists(path_name[0:-14]):
        os.mkdir(path_name[0:-14])

    with pd.ExcelWriter(path_name, engine="openpyxl") as writer:
        content_panel[0].to_excel(writer, sheet_name="seismometer Z", index=False)
        content_panel[1].to_excel(writer, sheet_name="seismometer X", index=False)
        content_panel[2].to_excel(writer, sheet_name="seismometer Y", index=False)
        diff_content_panel[0].to_excel(
            writer, sheet_name="diff seismometer Z", index=False
        )
        diff_content_panel[1].to_excel(
            writer, sheet_name="diff seismometer X", index=False
        )
        diff_content_panel[2].to_excel(
            writer, sheet_name="diff seismometer Y", index=False
        )
        regions_content_panel[0].to_excel(
            writer, sheet_name="regions seismometer Z", index=False
        )
        regions_content_panel[1].to_excel(
            writer, sheet_name="regions seismometer X", index=False
        )
        regions_content_panel[2].to_excel(
            writer, sheet_name="regions seismometer Y", index=False
        )
        time_data.to_excel(writer, sheet_name="arrival times table", index=False)
        detection_table.to_excel(writer, sheet_name="correlation table", index=False)
        intensity_table.to_excel(writer, sheet_name="intensity table", index=False)
        association_table.to_excel(writer, sheet_name="scores", index=False)
        filtered_events.to_excel(writer, sheet_name="events table", index=False)
        model_table.to_excel(writer, sheet_name="model table", index=False)


def generate_model_table(final_panel: np.asarray, noise):

    first_time = final_panel[0][0]["secs"].to_list()[0]

    intensity_table = final_panel[-3]
    association_table = final_panel[-2]
    event_data = final_panel[-1]
    time_data = final_panel[-5]

    label = ["Z", "X", "Y"]

    model_table = pd.DataFrame(
        {
            "event_id": [],
            "phase": [],
            "depth": [],
            "incident_angle": [],
            "takeoff_angle": [],
            "log_depth": [],
            "axis": [],
            "axis_noise": [],
            "activity_order": [],
            "magnitude": [],
            "expected_amp": [],
            "magnitude_type": [],
            "epicenter_distance": [],
            "log_dist": [],
            "first_arrival": [],
            "last_arrival": [],
            "ray_param": [],
            "score": [],
            "p_value": [],
            "effect_z": [],
        }
    )

    for idx, score_row in association_table[
        association_table["event_id"] != "all"
    ].iterrows():
        name = score_row["event_id"]
        phase = score_row["phase"]
        axis = score_row["axis"]
        score = score_row["temporal_association_score"]
        p_value = score_row["p_value"]
        effect_z = score_row["effect_z"]

        origin_time = event_data[event_data["event_id"] == name]["origin_time"].iloc[0]

        n_intensity_table = intensity_table[
            (intensity_table["event_id"] == name)
            & (intensity_table["phase"] == phase)
            & (intensity_table["axis"] == axis)
        ]

        activity_order = n_intensity_table["activity_order"].iloc[0]
        first_arrival = n_intensity_table["first_arrival"].iloc[0]
        last_arrival = n_intensity_table["last_arrival"].iloc[0]

        n_time_data = time_data[
            (time_data["phase"] == phase) & (time_data["event_id"] == name)
        ]

        ray_param = n_time_data["ray_parameter"]
        takeoff = n_time_data["takeoff_angle"]
        incident = n_time_data["incidente_angle"]

        try:
            ray_param = ray_param.iloc[0]
            takeoff = takeoff.iloc[0]
            incident = incident.iloc[0]
        except ValueError:
            pass

        magnitude = event_data[event_data["event_id"] == name]["magnitude"].iloc[0]
        magnitude_type = event_data[event_data["event_id"] == name][
            "magnitude_type"
        ].iloc[0]
        epicenter_distance = event_data[event_data["event_id"] == name][
            "epicenter_distance"
        ].iloc[0]
        depth = event_data[event_data["event_id"] == name]["depth"].iloc[0]

        log_dist = np.log10(6371 * (2 * np.pi / 180) * epicenter_distance)
        log_depth = np.log(1 + depth * 1000)

        axis_noise = noise[label.index(axis)][int(origin_time - first_time)]

        expected_amp = magnitude - 1.66 * np.log10(epicenter_distance)

        model_table.loc[len(model_table)] = [
            name,
            phase,
            depth,
            incident,
            takeoff,
            log_depth,
            axis,
            axis_noise,
            activity_order,
            magnitude,
            expected_amp,
            magnitude_type,
            epicenter_distance,
            log_dist,
            first_arrival,
            last_arrival,
            ray_param,
            score,
            p_value,
            effect_z,
        ]

    return model_table


def minmax_frequency(content_panel, time_data):
    t0 = np.asarray([s.timestamp for s in content_panel[0]["secs"]], float)
    f_max_hz = 0.9 * 0.5 / np.median(np.diff(t0))
    freq_hz = pd.to_numeric(time_data["frequency"], errors="coerce") / 1000.0
    surface = time_data["phase"].isin(["love", "rayleigh"])

    f_list = freq_hz[surface & (freq_hz <= f_max_hz)].to_list()
    if not f_list:
        print(f"Nenhuma onda superficial observável em {path}")
        return []
    return min(f_list), max(f_list)


def create_analysis_panel(
    path: str, saving_path="Data/resultado.xlsx", saving=False, plotting=False
):
    """
    Function that creates the complete analysis panel of a seismic dataset.

    The function loads the three seismometer components (Z, X, and Y) from an
    Excel file, applies signal preprocessing, computes amplitude changes,
    identifies dynamic regions, retrieves earthquake catalog information, and
    associates detected signals with cataloged events. Optional plotting and
    saving of the generated results can also be performed.

    Input:
    - path: path to the Excel file containing the seismic data
    - saving_path: output path used when saving the generated analysis tables
    - saving: boolean that enables saving the analysis results to an Excel file
    - plotting: boolean that enables visualization of the processed data and
      detection statistics

    Output:
    - data_sheet: list containing all generated analysis products:
        * content_panel: treated seismic data for each component
        * diff_content_panel: amplitude variation data
        * regions_content_panel: grouped dynamic regions
        * time_data: catalog arrival information
        * detection_table: earthquake detection results
        * intensity_table: signal-event correlation metrics
        * association_table: temporal association statistics
        * event_data: earthquake catalog information
        * model_table: summary table used for model evaluation

      Returns an empty list if no earthquake-related catalog information is
      found within the analyzed time window.
    """

    df_Z = pd.read_excel(path, sheet_name=0)
    df_X = pd.read_excel(path, sheet_name=1)
    df_Y = pd.read_excel(path, sheet_name=2)

    content_panel = [df_Z, df_X, df_Y]
    content_panel = [table_treatment(df, -3) for df in content_panel]

    event_data, time_data = load_catalog(content_panel[0])

    if event_data is not None and time_data is not None:
        fmin, fmax = minmax_frequency(content_panel, time_data)

        diff_content_panel_surface, noise_surface = zip(
            *[change_amplitude(df, (fmin, fmax)) for df in content_panel]
        )
        diff_content_panel_surface = [df.dropna() for df in diff_content_panel_surface]

        regions_content_panel_surface = [
            group_dynamic(df) for df in diff_content_panel_surface
        ]

        start = content_panel[0]["secs"].to_list()[0]
        end = content_panel[0]["secs"].to_list()[-1]
        time_data = time_data[
            (time_data["arrival_time"] >= start) & (time_data["arrival_time"] <= end)
        ]

        event_data = event_data[event_data["event_id"].isin(time_data["event_id"])]
        surface_data = time_data[time_data["phase"].isin(["love", "rayleigh"])]
        intensity_table_surface = correlations(
            diff_content_panel_surface, surface_data, event_data
        )
        detection_table_surface, association_table_surface = detecting_earthquake(
            diff_content_panel_surface,
            intensity_table_surface,
            surface_data,
            event_data,
        )

        fmin, fmax = 0.05, 0.20

        diff_content_panel_body, noise_body = zip(
            *[change_amplitude(df, (fmin, fmax)) for df in content_panel]
        )
        regions_content_panel_body = [
            group_dynamic(df) for df in diff_content_panel_body
        ]

        body_data = time_data[~time_data["phase"].isin(["love", "rayleigh"])]
        intensity_table_body = correlations(
            diff_content_panel_body, body_data, event_data
        )

        detection_table_body, association_table_body = detecting_earthquake(
            diff_content_panel_body,
            intensity_table_body,
            body_data,
            event_data,
        )

        data_sheet_surface = [
            content_panel,
            diff_content_panel_surface,
            regions_content_panel_surface,
            surface_data,
            detection_table_surface,
            intensity_table_surface,
            association_table_surface,
            event_data,
        ]
        data_sheet_body = [
            content_panel,
            diff_content_panel_body,
            regions_content_panel_body,
            body_data,
            detection_table_body,
            intensity_table_body,
            association_table_body,
            event_data,
        ]

        try:
            model_table_surface = generate_model_table(
                data_sheet_surface, noise_surface
            )
            model_table_body = generate_model_table(data_sheet_body, noise_body)
        except KeyError:
            return [], []

        data_sheet_surface = [
            content_panel,
            diff_content_panel_surface,
            regions_content_panel_surface,
            surface_data,
            detection_table_surface,
            intensity_table_surface,
            association_table_surface,
            event_data,
            model_table_surface,
        ]
        data_sheet_body = [
            content_panel,
            diff_content_panel_body,
            regions_content_panel_body,
            body_data,
            detection_table_body,
            intensity_table_body,
            association_table_body,
            event_data,
            model_table_body,
        ]

        if plotting:
            plot_seismo(
                content_panel,
                diff_content_panel_surface,
                regions_content_panel_surface,
                time_data,
            )
            plot_seismo(
                content_panel,
                diff_content_panel_body,
                regions_content_panel_body,
                time_data,
            )

            detection_table_surface.plot.bar(
                "event_id", "fraction_above_reference", rot=25
            )
            plt.tight_layout()
            plt.show()

        # if saving:
        #     save_excel(
        #         content_panel,
        #         diff_content_panel_surface,
        #         regions_content_panel_surface,
        #         time_data,
        #         event_data,
        #         detection_table_surface,
        #         intensity_table_surface,
        #         association_table,
        #         model_table,
        #         saving_path,
        #     )
    else:
        print(f"No earthquake detected in the data sheet {path[-10:-1]}")
        return [], []

    return data_sheet_surface, data_sheet_body


if __name__ == "__main__":
    name_list = os.listdir("Data/sismic_data")

    path = "Data/sismic_data/"

    for i, name in enumerate(name_list):
        data_sheet_surface, data_sheet_body = create_analysis_panel(path + name)
        try:
            save_excel(
                data_sheet_surface[0],
                data_sheet_surface[1],
                data_sheet_surface[2],
                data_sheet_surface[3],
                data_sheet_surface[7],
                data_sheet_surface[4],
                data_sheet_surface[5],
                data_sheet_surface[6],
                data_sheet_surface[8],
                "Data/" + name[0:-5] + "/Surface_resultado.xlsx",
            )
            save_excel(
                data_sheet_body[0],
                data_sheet_body[1],
                data_sheet_body[2],
                data_sheet_body[3],
                data_sheet_body[7],
                data_sheet_body[4],
                data_sheet_body[5],
                data_sheet_body[6],
                data_sheet_body[8],
                "Data/" + name[0:-5] + "/Body_resultado.xlsx",
            )
            while True:
                path_surface = "Data/" + name[0:-5] + "/Surface_resultado.xlsx"
                path_body = "Data/" + name[0:-5] + "/Body_resultado.xlsx"
                try:
                    df_surface = pd.read_excel(path_surface, sheet_name=-1)
                    df_body = pd.read_excel(path_body, sheet_name=-1)

                    break
                except BadZipFile:
                    save_excel(
                        data_sheet_surface[0],
                        data_sheet_surface[1],
                        data_sheet_surface[2],
                        data_sheet_surface[3],
                        data_sheet_surface[7],
                        data_sheet_surface[4],
                        data_sheet_surface[5],
                        data_sheet_surface[6],
                        data_sheet_surface[8],
                        "Data/" + name[0:-5] + "/Surface_resultado.xlsx",
                    )
                    save_excel(
                        data_sheet_body[0],
                        data_sheet_body[1],
                        data_sheet_body[2],
                        data_sheet_body[3],
                        data_sheet_body[7],
                        data_sheet_body[4],
                        data_sheet_body[5],
                        data_sheet_body[6],
                        data_sheet_body[8],
                        "Data/" + name[0:-5] + "/Body_resultado.xlsx",
                    )

            print(f"{i + 1} / {len(name_list)}")
        except IndexError:
            try:
                save_excel(
                    data_sheet_body[0],
                    data_sheet_body[1],
                    data_sheet_body[2],
                    data_sheet_body[3],
                    data_sheet_body[7],
                    data_sheet_body[4],
                    data_sheet_body[5],
                    data_sheet_body[6],
                    data_sheet_body[8],
                    "Data/" + name[0:-5] + "/Body_resultado.xlsx",
                )
                print(f"{i + 1} / {len(name_list)}")
            except IndexError:
                print(f"{i + 1} / {len(name_list)} - error in: {name}")
                continue
