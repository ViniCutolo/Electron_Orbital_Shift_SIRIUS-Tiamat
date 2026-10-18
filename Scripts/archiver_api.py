import datetime
import getpass
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import requests
from constants import API_KEY, BASE_URL, MODEL_NAME, SYSTEM_PROMPT
from openai import OpenAI


def get_archiver_data(
    pv,
    from_time,
    to_time,
    optimize=-1,
    diff=True,
    base_url="https://ais-eng-srv-ca.cnpem.br/retrieval/data/getData.json",
):
    """
    Retrieve time-series data from the EPICS Archiver Appliance and return it
    as a pandas DataFrame.

    Parameters
    ----------
    pv : str
        Process Variable (PV) name to query.
    from_time : str
        Start timestamp of the query interval. Must be in a format accepted
        by the Archiver Appliance API.
    to_time : str
        End timestamp of the query interval. Must be in a format accepted
        by the Archiver Appliance API.
    optimize : int, optional
        Optimization level for server-side aggregation. If set to -1, raw
        data are retrieved. Otherwise, the PV is requested using the
        ``optimized_<optimize>(PV)`` syntax provided by the Archiver API.
        Default is -1.
    diff : bool, optional
        If ``True`` and an optimized query is used, the returned ``val``
        column is shifted so that its first value becomes zero. This is
        useful for analyzing cumulative quantities as relative variations
        over the selected time interval. Ignored when ``optimize = -1``.
        Default is ``True``.
    base_url : str, optional
        URL of the Archiver Appliance data retrieval endpoint.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the retrieved data.

        For non-optimized queries, the DataFrame contains the fields
        returned directly by the API.

        For optimized queries, the aggregated statistics are expanded into
        separate columns:

        - ``val``: aggregated value (or relative value if ``diff=True``)
        - ``std``: standard deviation
        - ``min``: minimum value
        - ``max``: maximum value
        - ``count``: number of samples used in the aggregation

    Raises
    ------
    requests.HTTPError
        If the HTTP request to the Archiver Appliance fails.

    Notes
    -----
    The function expects the response format of the Archiver Appliance
    ``getData.json`` endpoint, where the payload is available at
    ``response.json()[0]["data"]``.

    When optimization is enabled, the API returns each value as a list
    containing the aggregated value and additional statistics. These are
    automatically unpacked into separate DataFrame columns.
    """

    parameters = {
        "pv": f"optimized_{optimize}({pv})" if optimize != -1 else pv,
        "from": from_time,
        "to": to_time,
    }

    response = requests.get(base_url, params=parameters)

    if not response.ok:
        response.raise_for_status()

    json_data = response.json()[0]["data"]

    df = pd.json_normalize(json_data)

    if optimize != -1:
        df["std"] = df["val"].apply(lambda x: x[1])
        df["min"] = df["val"].apply(lambda x: x[2])
        df["max"] = df["val"].apply(lambda x: x[3])
        df["count"] = df["val"].apply(lambda x: int(x[4]))

        df["val"] = df["val"].apply(lambda x: x[0])
        if diff:
            df["val"] -= df["val"][0]

    return df


def get_times_from_description(log_text):
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": log_text},
        ],
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content
