import os

import matplotlib.pyplot as plt
import obspy.core.utcdatetime as utc
import obspy.geodetics as geo
import pandas as pd
from obspy import taup
from obspy.clients.fdsn import Client
from obspy.core.event.catalog import Catalog

# Constants
POS_LAT_SIRIUS = -22.804617
POS_LON_SIRIUS = -47.05297


def arrival_list(latitude, longitude, depth):

    # geodesic_distance is in meters, azimuth_degree, back_azimuth_degree and epicenter_distance is in degree
    geodesic_distance, azimuth_degree, back_azimuth_degree = geo.gps2dist_azimuth(
        latitude, longitude, POS_LAT_SIRIUS, POS_LON_SIRIUS
    )
    epicenter_distance_degree = geo.locations2degrees(
        latitude, longitude, POS_LAT_SIRIUS, POS_LON_SIRIUS
    )

    model_name = "iasp91"
    model = taup.TauPyModel(model=model_name)

    travel_times = model.get_travel_times(
        source_depth_in_km=depth, distance_in_degree=epicenter_distance_degree
    )

    return travel_times, epicenter_distance_degree, model, model_name


def earth_plot(model: taup.TauPyModel, depth, epicenter_distance_degree):

    paths = model.get_ray_paths(depth, epicenter_distance_degree)

    fig = plt.figure(figsize=(12, 5))

    ax_rays = fig.add_subplot(1, 2, 1, projection="polar")
    ax_times = fig.add_subplot(1, 2, 2)

    paths.plot_rays(legend=True, fig=fig, ax=ax_rays, show=False)
    paths.plot_times(legend=False, fig=fig, ax=ax_times, show=False)

    plt.tight_layout()
    plt.show()


def table_plot(id: str, event_data: pd.DataFrame, time_data: pd.DataFrame):
    """
    This function will get the specific event wanted and will plot the earthquake

    Input:

    - id: the name of the event

    - event_data: the dataframe of the events being used

    - time_data: the dataframe of the arrival times of the events being used
    """

    new_event_data = event_data[event_data["event_id"] == id]
    new_time_data = time_data[time_data["event_id"] == id]

    model_name = new_time_data["model_used"].tolist()[0]

    model = taup.TauPyModel(model=model_name)

    earth_plot(
        model,
        new_event_data["depth"].tolist()[0],
        new_event_data["epicenter_distance"].tolist()[0],
    )


def table_create(
    latitude: float,
    longitude: float,
    depth: float,
    magnitude: float,
    magnitude_type: str,
    event_id: str,
    arrivals: taup.tau.Arrivals,
    UTC_origin: utc.UTCDateTime,
    model_name: str,
    catalog: str,
    epicenter_distance: float,
    event_data: pd.DataFrame = None,
    time_data: pd.DataFrame = None,
):
    """
    Function that add or create a Data Frame in pandas format for the event and it's waves.

    Entry:

    - latitude: the latitude of the event in degree

    - longitude: the longitude of the event in degree

    - depth: the estimated depth of the event in kilometers

    - magnitude: how strong was the earthquake in magnitude type

    - magnitude_type: the type of earthquake used

    - event_id: the name of the event

    - arrivals: the arrival list calculated by ``get_travel_time``

    - UTC_origin: the time date of the event in the UTC scale

    - model_name: the name of the model used to calculate the ray pattern

    - catalog: the source of the earthquake that is being analysed

    - epicenter_distance: the epicenter distance between the source and the target in degree

    - event_data: the data frame of the event, `None` is the default value if none was created until now

    - time_data: the data frame of every single wave that reaches the target, `None` is the default value if none was created until now

    Output:

    - An updated version of event_data or event_data if none was given

    - An updated version of time_data or time_data if none was given
    """

    if event_data is None or time_data is None:
        # Creating data frame if it doesn't exist
        event_data = pd.DataFrame(
            {
                "event_id": [event_id + str(UTC_origin)],
                "latitude": [latitude],
                "longitude": [longitude],
                "epicenter_distance": [epicenter_distance],
                "depth": [depth],
                "magnitude": [magnitude],
                "magnitude_type": [magnitude_type],
                "Source": [catalog],
                "origin_time": [UTC_origin],
            }
        )
        time_data = pd.DataFrame(
            {
                "event_id": [],
                "phase": [],
                "takeoff_angle": [],
                "incidente_angle": [],
                "ray_parameter": [],
                "travel_time": [],
                "arrival_time": [],
                "model_used": [],
            }
        )

        for arrival in arrivals:
            time_data.loc[len(time_data)] = [
                event_id,
                arrival.name,
                arrival.takeoff_angle,
                arrival.incident_angle,
                arrival.ray_param,
                arrival.time,
                UTC_origin + arrival.time,
                model_name,
            ]

    else:
        # Adding new data
        event_data.loc[len(event_data)] = [
            event_id + str(UTC_origin),
            latitude,
            longitude,
            epicenter_distance,
            depth,
            magnitude,
            magnitude_type,
            catalog,
            UTC_origin,
        ]

        for arrival in arrivals:
            time_data.loc[len(time_data)] = [
                event_id + str(UTC_origin),
                arrival.name,
                arrival.takeoff_angle,
                arrival.incident_angle,
                arrival.ray_param,
                arrival.time,
                UTC_origin + arrival.time,
                model_name,
            ]

    return event_data, time_data


def analise_catalog(events: Catalog, catalog: str, plotting: bool = False):
    """
    With a given list of events of a catalog, this function will create a dataframe with all the events and each phase wave

    Input:

    - events: the catalog list of all the events tha will be put in the dataframe

    - catalog: the name of the catalog used to get the events

    - plotting: if `True`, all the events will be plotted. The default argument is `False`.

    Output:

    - An updated version of the event dataframe or event_data if none was given

    - An updated version of the time dataframe or time_data if none was given

    """

    event_data, time_data = load_dataframe()

    for event in events:
        pref_origin = event.preferred_origin()
        pref_mag = event.preferred_magnitude()

        if pref_origin is None:
            pref_origin = event.origins[0]

        if pref_mag is None:
            pref_mag = event.magnitudes[0]

        latitude = pref_origin.latitude
        longitude = pref_origin.longitude
        depth = pref_origin.depth / 1000
        magnitude = pref_mag.mag
        magnitude_type = pref_mag.magnitude_type
        event_id = event.event_descriptions[0].text
        origin_time = pref_origin.time

        travel_time, epicenter_distance_degree, model, model_name = arrival_list(
            latitude, longitude, depth
        )

        if plotting:
            earth_plot(model, depth, epicenter_distance_degree)

        event_data, time_data = table_create(
            latitude,
            longitude,
            depth,
            magnitude,
            magnitude_type,
            event_id,
            travel_time,
            origin_time,
            model_name,
            catalog,
            epicenter_distance_degree,
            event_data,
            time_data,
        )

    return event_data, time_data


def save_dataframe(timeframe: pd.DataFrame, eventframe: pd.DataFrame):
    """
    Saves two dataframes in .csv format in a data folder.

    Input:

    - eventframe: the data frame of the events

    - timeframe: the data frame of every single wave that reaches the target
    """

    timeframe.to_csv("Data/time_data.csv", sep=";", index=False)
    eventframe.to_csv("Data/event_data.csv", sep=";", index=False)


def load_dataframe():
    """
    Load the saved data in .csv format and output a dataframe with the earthquakes

    Output:

    - an existing dataframe of the events or an empty dataframe with the correct number of columns


    - an existing dataframe of the phase arrival or an empty dataframe with the correct number of columns
    """

    if os.path.exists("Data/time_data.csv") and os.path.exists("Data/event_data.csv"):
        timeframe = pd.read_csv("Data/time_data.csv", sep=";")
        eventframe = pd.read_csv("Data/event_data.csv", sep=";")
    else:
        eventframe, timeframe = table_create(
            0,
            0,
            0,
            0,
            "None",
            "None",
            [],
            utc.UTCDateTime("2000-01-01"),
            "None",
            "None",
            0,
        )

        eventframe.drop(0)

    return eventframe, timeframe


if __name__ == "__main__":
    # event_data, time_data = load_dataframe()
    # latitude = float(input("Latitude in degree: "))
    # longitude = float(input("Longitude in degree: "))
    # depth = float(input("Depth in kilometers: "))

    catalog = "ISC"
    client = Client(catalog)
    events = client.get_events(
        starttime=utc.UTCDateTime("2025-01-01"),
        endtime=utc.UTCDateTime("2025-01-05"),
        minmagnitude=5.0,
    )

    print(type(events))

    event_data, time_data = analise_catalog(events, catalog)

    # event_1 = events[0]

    # latitude = event_1.origins[0].latitude
    # longitude = event_1.origins[0].longitude
    # depth = event_1.origins[0].depth / 1000
    # magnitude = event_1.magnitudes[0].mag
    # magnitude_type = event_1.magnitudes[0].magnitude_type
    # event_id = event_1.event_descriptions[0].text
    # origin_time = event_1.origins[0].time

    # print(len(event_1.origins))
    # print(len(event_1.magnitudes))
    # print(len(event_1.event_descriptions))

    # travel_time, epicenter_distance_degree, model, model_name = arrival_list(
    #     latitude, longitude, depth
    # )

    # earth_plot(model, depth, epicenter_distance_degree)

    # date1 = utc.UTCDateTime("2026-09-30 05:00:25", precision=True)

    # event_data, time_data = table_create(
    #     latitude,
    #     longitude,
    #     depth,
    #     "None",
    #     "None",
    #     "test",
    #     travel_time,
    #     date1,
    #     model_name,
    #     # event_data,
    #     # time_data,
    # )

    print(time_data)
    print("-------------------------------------------------------------------")
    print(event_data)

    table_plot("Ethiopia2025-01-04T23:03:59.970000Z", event_data, time_data)

    # save_dataframe(time_data, event_data)
