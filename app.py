#!/usr/bin/env python
# coding: utf-8

# In[5]:

import ssl
from geopy.geocoders import Nominatim
import time
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.enums import Resampling
from pyproj import Transformer
import rasterio
import netCDF4 as nc
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from collections import defaultdict
from flask import Flask, request, jsonify
import io
import base64
from flask_cors import CORS
import statistics
import mysql.connector
from datetime import datetime

# Database connection details
db_config = {
    'host': '127.0.0.1',
    'user': 'migracoast',
    'password': 'St5SfZXpxRPS4XZy',
    'database': 'migracoast'
}

# In[6]:


file_path_ar6_ssp585 = 'ar6data/total_ssp585_medium_confidence_values.nc'

file_path_ar6_ssp245 = 'ar6data/total_ssp245_medium_confidence_values.nc'


# In[25]:


storm_and_flood_csv = "cmip6data/StormEvents_details-ftp_v1.0_d2000_c20220425.csv"


inundation_maps_MA = {
    0: "inundationmaps/MA_connectRaster_0.tif",
    0.5: "inundationmaps/MA_connectRaster_0_5.tif",
    1: "inundationmaps/MA_connectRaster_1.tif",
    1.5: "inundationmaps/MA_connectRaster_1_5.tif",
    2: "inundationmaps/MA_connectRaster_2.tif",
    2.5: "inundationmaps/MA_connectRaster_2_5.tif",
    3: "inundationmaps/MA_connectRaster_3.tif",
    3.5: "inundationmaps/MA_connectRaster_3_5.tif",
    4: "inundationmaps/MA_connectRaster_4.tif"
}

inundation_maps_PA = {
    0: "inundationmaps/PA_connectRaster_0.tif",
    0.5: "inundationmaps/PA_connectRaster_0_5.tif",
    1: "inundationmaps/PA_connectRaster_1.tif",
    1.5: "inundationmaps/PA_connectRaster_1_5.tif",
    2: "inundationmaps/PA_connectRaster_2.tif",
    2.5: "inundationmaps/PA_connectRaster_2_5.tif",
    3: "inundationmaps/PA_connectRaster_3.tif",
    3.5: "inundationmaps/PA_connectRaster_3_5.tif",
    4: "inundationmaps/PA_connectRaster_4.tif"
}

inundation_maps_RI = {
    0: "inundationmaps/RI_connectRaster_0.tif",
    0.5: "inundationmaps/RI_connectRaster_0_5.tif",
    1: "inundationmaps/RI_connectRaster_1.tif",
    1.5: "inundationmaps/RI_connectRaster_1_5.tif",
    2: "inundationmaps/RI_connectRaster_2.tif",
    2.5: "inundationmaps/RI_connectRaster_2_5.tif",
    3: "inundationmaps/RI_connectRaster_3.tif",
    3.5: "inundationmaps/RI_connectRaster_3_5.tif",
    4: "inundationmaps/RI_connectRaster_4.tif"
}

inundation_maps_AL = {
    0: "inundationmaps/AL_connectRaster_0.tif",
    0.5: "inundationmaps/AL_connectRaster_0_5.tif",
    1: "inundationmaps/AL_connectRaster_1.tif",
    1.5: "inundationmaps/AL_connectRaster_1_5.tif",
    2: "inundationmaps/AL_connectRaster_2.tif",
    2.5: "inundationmaps/AL_connectRaster_2_5.tif",
    3: "inundationmaps/AL_connectRaster_3.tif",
    3.5: "inundationmaps/AL_connectRaster_3_5.tif",
    4: "inundationmaps/AL_connectRaster_4.tif"
}

inundation_maps_CT = {
    0: "inundationmaps/CT_connectRaster_0.tif",
    0.5: "inundationmaps/CT_connectRaster_0_5.tif",
    1: "inundationmaps/CT_connectRaster_1.tif",
    1.5: "inundationmaps/CT_connectRaster_1_5.tif",
    2: "inundationmaps/CT_connectRaster_2.tif",
    2.5: "inundationmaps/CT_connectRaster_2_5.tif",
    3: "inundationmaps/CT_connectRaster_3.tif",
    3.5: "inundationmaps/CT_connectRaster_3_5.tif",
    4: "inundationmaps/CT_connectRaster_4.tif"
}

inundation_maps_DC = {
    0: "inundationmaps/DC_connectRaster_0.tif",
    0.5: "inundationmaps/DC_connectRaster_0_5.tif",
    1: "inundationmaps/DC_connectRaster_1.tif",
    1.5: "inundationmaps/DC_connectRaster_1_5.tif",
    2: "inundationmaps/DC_connectRaster_2.tif",
    2.5: "inundationmaps/DC_connectRaster_2_5.tif",
    3: "inundationmaps/DC_connectRaster_3.tif",
    3.5: "inundationmaps/DC_connectRaster_3_5.tif",
    4: "inundationmaps/DC_connectRaster_4.tif"
}

inundation_maps_DE = {
    0: "inundationmaps/DE_connectRaster_0.tif",
    0.5: "inundationmaps/DE_connectRaster_0_5.tif",
    1: "inundationmaps/DE_connectRaster_1.tif",
    1.5: "inundationmaps/DE_connectRaster_1_5.tif",
    2: "inundationmaps/DE_connectRaster_2.tif",
    2.5: "inundationmaps/DE_connectRaster_2_5.tif",
    3: "inundationmaps/DE_connectRaster_3.tif",
    3.5: "inundationmaps/DE_connectRaster_3_5.tif",
    4: "inundationmaps/DE_connectRaster_4.tif"
}

inundation_maps_MS = {
    0: "inundationmaps/MS_connectRaster_0.tif",
    0.5: "inundationmaps/MS_connectRaster_0_5.tif",
    1: "inundationmaps/MS_connectRaster_1.tif",
    1.5: "inundationmaps/MS_connectRaster_1_5.tif",
    2: "inundationmaps/MS_connectRaster_2.tif",
    2.5: "inundationmaps/MS_connectRaster_2_5.tif",
    3: "inundationmaps/MS_connectRaster_3.tif",
    3.5: "inundationmaps/MS_connectRaster_3_5.tif",
    4: "inundationmaps/MS_connectRaster_4.tif"
}

inundation_maps_NH = {
    0: "inundationmaps/NH_connectRaster_0.tif",
    0.5: "inundationmaps/NH_connectRaster_0_5.tif",
    1: "inundationmaps/NH_connectRaster_1.tif",
    1.5: "inundationmaps/NH_connectRaster_1_5.tif",
    2: "inundationmaps/NH_connectRaster_2.tif",
    2.5: "inundationmaps/NH_connectRaster_2_5.tif",
    3: "inundationmaps/NH_connectRaster_3.tif",
    3.5: "inundationmaps/NH_connectRaster_3_5.tif",
    4: "inundationmaps/NH_connectRaster_4.tif"
}

inundation_maps_FL = {
    0: "inundationmaps/FL_SE_connectRaster_0_0.tif",
    0.5: "inundationmaps/FL_SE_connectRaster_0_5.tif",
    1: "inundationmaps/FL_SE_connectRaster_1_0.tif",
    1.5: "inundationmaps/FL_SE_connectRaster_1_5.tif",
    2: "inundationmaps/FL_SE_connectRaster_2_0.tif",
    2.5: "inundationmaps/FL_SE_connectRaster_2_5.tif",
    3: "inundationmaps/FL_SE_connectRaster_3_0.tif",
    3.5: "inundationmaps/FL_SE_connectRaster_3_5.tif",
    4: "inundationmaps/FL_SE_connectRaster_4_0.tif"
}

inundation_maps_CA = {
    0: "inundationmaps/CA_LOX_connectRaster_0.tif",
    1: "inundationmaps/CA_LOX_connectRaster_1.tif",
    2: "inundationmaps/CA_LOX_connectRaster_2.tif",
    3: "inundationmaps/CA_LOX_connectRaster_3.tif",
    4: "inundationmaps/CA_LOX_connectRaster_4.tif"
}

inundation_maps_NY = {
    0: "inundationmaps/NY_SK_connectRaster_0.tif",
    0.5: "inundationmaps/NY_SK_connectRaster_0_5.tif",
    1: "inundationmaps/NY_SK_connectRaster_1.tif",
    1.5: "inundationmaps/NY_SK_connectRaster_1_5.tif",
    2: "inundationmaps/NY_SK_connectRaster_2.tif",
    2.5: "inundationmaps/NY_SK_connectRaster_2_5.tif",
    3: "inundationmaps/NY_SK_connectRaster_3.tif",
    3.5: "inundationmaps/NY_SK_connectRaster_3_5.tif",
    4: "inundationmaps/NY_SK_connectRaster_4.tif"
}
# In[8]:


state_inundation_maps = {
    "Massachusetts": inundation_maps_MA,
    "MA": inundation_maps_MA,
    "Pennsylvania": inundation_maps_PA,
    "PA": inundation_maps_PA,
    "Rhode Island": inundation_maps_RI,
    "RI": inundation_maps_RI,
   "Alabama": inundation_maps_AL,
   "AL": inundation_maps_AL,
    "Connecticut": inundation_maps_CT,
    "CT": inundation_maps_CT,
    "Washington, DC": inundation_maps_DC,
    "DC": inundation_maps_DC,
    "Delaware": inundation_maps_DE,
    "DE": inundation_maps_DE,
    "Mississippi": inundation_maps_MS,
    "MS": inundation_maps_MS,
    "New Hampshire": inundation_maps_NH,
    "NH": inundation_maps_NH,
    "FL": inundation_maps_FL,
    "Florida": inundation_maps_FL,
    "California": inundation_maps_CA,
    "CA": inundation_maps_CA,
    "New York": inundation_maps_NY,
    "NY": inundation_maps_NY
}

# SQL query to insert the data
insert_query = """
INSERT INTO m_query_data (
    latitude, longitude, state, zoom_factor, find_radius, find_step, 
    number_locations, coordinates, flood_score, future_inundation_occur, 
    inundation_occur, inundation_score, inundation_year, 
    major_city_ras, maximum_slr, maximum_slr_year, 
    place_name, proximity_score, risk_assessment_score, 
    safe_locations, slr_score, ssp585_result, storm_score, 
    create_date
) 
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""

# Function to insert data into the database
def insert_data(data):
    try:
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        data["create_date"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        data_values = (
            data["latitude"], data["longitude"], data["state"], data["zoom_factor"],
            data["find_radius"], data["find_step"], data["number_locations"],
            str(data["coordinates"]), data["flood_score"], data["future_inundation_occur"],
            data["inundation_occur"], data["inundation_score"],
            data["inundation_year"], str(data["major_city_ras"]),
            data["maximum_slr"], data["maximum_slr_year"],
            data["place_name"], data["proximity_score"],
            data["risk_assessment_score"], data["safe_locations"], data["slr_score"],
            data["ssp585_result"], data["storm_score"],
            data["create_date"]
        )

        # Insert data
        cursor.execute(insert_query, data_values)

        # Commit the transaction
        conn.commit()
        print("Data inserted successfully.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            
# In[9]:

def get_sea_level_projection(input_lat, input_lon, file_path):
    # Load the dataset
    ds = nc.Dataset(file_path, 'r')
    
    # Extract latitude, longitude, sea level change, and years
    latitude = ds.variables['lat'][:]
    longitude = ds.variables['lon'][:]
    sea_level_change = ds.variables['sea_level_change'][:]  # (quantiles, years, locations)
    years = ds.variables['years'][:]
    
    # Adjust longitude to [-180, 180] if necessary
    if np.max(longitude) > 180:
        longitude = np.where(longitude > 180, longitude - 360, longitude)
    
    # Function to find nearby points within a given threshold
    def find_nearby_points(lat_threshold, lon_threshold, max_points=5):
        nearby_lat = np.argwhere(np.abs(latitude - input_lat) < lat_threshold).flatten()
        nearby_lon = np.argwhere(np.abs(longitude[nearby_lat] - input_lon) < lon_threshold).flatten()
        qualified_idx = nearby_lat[nearby_lon]
        return qualified_idx[:max_points]  # Return at most max_points
    
    # Try finding points within 0.5 degrees
    qualified_idx = find_nearby_points(0.5, 0.5)
    
    # If no points found, expand the search to 1 degree
    if len(qualified_idx) == 0:
        qualified_idx = find_nearby_points(1.0, 1.0)
    
    # If still no points found, return an empty DataFrame
    if len(qualified_idx) == 0:
        return None
    
    # Step 2: Extract sea level change data for the nearby coordinates
    # Assuming quantile 50 (median) for sea level rise projections
    quantile_idx = 50  # Adjust if necessary

    sea_level_rise_data = []
    for idx in qualified_idx:
        slr_at_location = sea_level_change[quantile_idx, :, idx]  # Sea level rise at this location for all years
        sea_level_rise_data.append(slr_at_location)

    # Step 3: Average sea level rise over all nearby locations
    sea_level_rise_avg = np.mean(sea_level_rise_data, axis=0)

    # Create a DataFrame for the time series
    df = pd.DataFrame({
        'Year': years,
        'Sea Level Rise': sea_level_rise_avg
    })
    df = df[df['Year'] <= 2100]
    # Close the dataset
    ds.close()
    if not np.ma.is_masked(df):
        return df
    return None

# In[11]:


def get_map_boundary(inundation_map):
    with rasterio.open(inundation_map) as src:
            # Check the bounds of the raster file (lon_min, lon_max should be in [-180, 180])
            lat_min, lat_max = src.bounds.bottom, src.bounds.top
            lon_min, lon_max = src.bounds.left, src.bounds.right
    return lat_min, lat_max, lon_min, lon_max


# In[12]:


# Function to check if a coordinate is inundated using a smaller window
def check_inundation(file_path, lat, lon, window_size=5):
    def check_point(lat, lon):
        """Helper function to check inundation for a single point."""
        with rasterio.open(file_path) as src:
            # Check the bounds of the raster file (lon_min, lon_max should be in [-180, 180])
            lat_min, lat_max = src.bounds.bottom, src.bounds.top
            lon_min, lon_max = src.bounds.left, src.bounds.right
            
            # Transform the input coordinates to the raster's CRS
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            
            # Check if the input coordinate is within the bounds of the raster
            if not (lon_min <= x <= lon_max and lat_min <= y <= lat_max):
                return "OUTSIDE BOUNDARY"
            
            # Get the row and column indices for the input coordinate
            row, col = src.index(x, y)
            
            # Define a window around the input coordinate
            row_start = max(row - window_size, 0)
            row_end = min(row + window_size, src.height)
            col_start = max(col - window_size, 0)
            col_end = min(col + window_size, src.width)
            
            window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
            
            # Read only the window of interest and ensure it's valid
            try:
                data = src.read(1, window=window, masked=True)
            except Exception as e:
                print(f"Error reading data in the window: {e}")
                return "OUTSIDE BOUNDARY"

            # Calculate local row and column within the window
            local_row = row - row_start
            local_col = col - col_start

            # Ensure the indices are within bounds of the data window
            if local_row >= data.shape[0] or local_col >= data.shape[1]:
                return "OUTSIDE BOUNDARY"
            
            # Check the value at the location and handle NoData explicitly
            value_at_location = data[local_row, local_col]

            # Handle NoData and inundation values explicitly
            if value_at_location == 0 or value_at_location == 0.0:
                return False  # Uninundated
            elif value_at_location == 1 or value_at_location == 1.0:
                return True  # Inundated
            else:
                return None

    # First check the original input point
    result = check_point(lat, lon)
    if result in [True, False]:  # If we get a valid result (inundated or not), return it
        #print("The value is taken directly from the input coordinate (not from surrounding buffer)")
        return result
    elif result == "OUTSIDE BOUNDARY":
        return "OUTSIDE BOUNDARY"

    # If the original point has no value, check surrounding 8 points
    lat_offset = 0.0001
    lon_offset = 0.0001
    nearby_points = [
        (lat + lat_offset, lon),
        (lat - lat_offset, lon),
        (lat, lon + lon_offset),
        (lat, lon - lon_offset),
        (lat + lat_offset, lon + lon_offset),
        (lat - lat_offset, lon - lon_offset),
        (lat + lat_offset, lon - lon_offset),
        (lat - lat_offset, lon + lon_offset),
    ]
    nearby_results = []
    for nearby_lat, nearby_lon in nearby_points:
        nearby_result = check_point(nearby_lat, nearby_lon)
        if nearby_result == True:  # Prioritize inundation
            #print("The value is taken from surrounding buffer")
            return True
        elif nearby_result == False:  # Uninundated
            #print("The value is taken from surrounding buffer")
            nearby_results.append(nearby_result)
            continue  # Keep checking for inundation
        # Skip if no value (None or "UNKNOWN_VALUE or NO_DATA")
    if nearby_results:
        return False
    # If none of the points had a value, return no value
    return None



# Initialize the geolocator with a more specific user agent
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
geolocator = Nominatim(user_agent="my_geocoding_application", ssl_context=ssl_context, timeout=600)

def get_place_name(lat, lon):
    try:
        # Introduce a delay to avoid rate limiting
        time.sleep(1)  # Wait for 1 second between requests
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location and location.address:
            address = location.raw['address']
            # Return the first available place name: city, town, or village
            if 'city' in address:
                return address['city']
            elif 'town' in address:
                return address['town']
            elif 'village' in address:
                return address['village']
            else:
                return "Unknown"
        else:
            return "Unknown"
    except Exception as e:
        return "Unknown"


# In[17]:


def find_closest_uninundated_locations(input_lat, input_lon, inundation_map, search_radius=0.08, step_size=0.02, max_locations=3, safe_margin=0.005, safe_step=0.0025):
    # Get the boundary of the map (latitude and longitude limits)
    
    lat_min, lat_max, lon_min, lon_max = get_map_boundary(inundation_map)
    if not (lon_min <= input_lon <= lon_max and lat_min <= input_lat <= lat_max):
        return None
    # List to store safe locations
    safe_locations = []
    unique_place_names = set()  # Set to avoid duplicate place names

    layer = 1  # Start at layer 1 (surrounding the input point)
    
    # Continue searching until the required number of safe locations is found or search_radius is exceeded
    while len(safe_locations) < max_locations and layer * step_size <= search_radius:
        # Generate the nearby points for the current layer
        nearby_points = []
        
        # Top row (moving right)
        for i in range(-layer, layer + 1):
            nearby_points.append((input_lat + i * step_size, input_lon + layer * step_size))
        
        # Right column (moving down)
        for i in range(layer - 1, -layer - 1, -1):
            nearby_points.append((input_lat + layer * step_size, input_lon + i * step_size))
        
        # Bottom row (moving left)
        for i in range(layer - 1, -layer - 1, -1):
            nearby_points.append((input_lat + i * step_size, input_lon - layer * step_size))
        
        # Left column (moving up)
        for i in range(-layer + 1, layer):
            nearby_points.append((input_lat - layer * step_size, input_lon + i * step_size))
        
        # Filter out points outside the map boundaries
        nearby_points = [(lat, lon) for lat, lon in nearby_points if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max]

        # Check the nearby points for inundation status and margin safety
        for nearby_lat, nearby_lon in nearby_points:
            # Skip if the point is inundated
            if check_inundation(inundation_map, nearby_lat, nearby_lon):
                continue
            
            # Perform reverse geocoding to get the nearest place name
            place_name = get_place_name(nearby_lat, nearby_lon)
            
            if place_name not in unique_place_names and place_name != "Unknown":
                # Add the safe location to the list
                safe_locations.append((nearby_lat, nearby_lon, place_name))
                unique_place_names.add(place_name)

            # If we've found the required number of locations, return them
            if len(safe_locations) >= max_locations:
                return safe_locations
        
        # Move to the next outer layer
        layer += 1

    # Return the found locations or None if no safe locations are found
    return safe_locations if safe_locations else None


# In[18]:

def process_scenario(scenario_name, file_path, inundation_map, input_lat, input_lon):
    coordinates = (input_lat, input_lon)
    bound_check = check_inundation(inundation_map[0], input_lat, input_lon, window_size=10)
    output = ""
    if bound_check == "OUTSIDE BOUNDARY":
        output, ssh_change_df, highest_threshold_exceeded, next_output, inundation_occurred, inundation_year, coordinates, place_name, next_inundation_status = None, None, None, None, None, None, None, None, None
        return output, ssh_change_df, highest_threshold_exceeded, next_output, inundation_occurred, inundation_year, coordinates, place_name, next_inundation_status
        
    slr_df = get_sea_level_projection(input_lat, input_lon, file_path)
    
    # Convert SSH_Change from meters to feet
    mm_to_meters = 0.001
    meters_to_feet = 3.28084
    slr_df['Sea Level Rise'] = slr_df['Sea Level Rise'] * mm_to_meters * meters_to_feet

    # Track thresholds exceeded, their corresponding years, and whether inundation occurs
    threshold_exceedance = []

    available_thresholds = sorted(inundation_map.keys())
    highest_threshold_exceeded = 0
    inundation_occurred = False
    inundation_year = None
    for threshold in available_thresholds:
        exceedance_year = slr_df[slr_df['Sea Level Rise'] > threshold]['Year'].min()
        if not pd.isna(exceedance_year):
            file_path = inundation_map[threshold]
            inundation_status = check_inundation(file_path, input_lat, input_lon, window_size=10)  # Pass window_size
            threshold_exceedance.append((threshold, exceedance_year, inundation_status))
            if inundation_status:
                inundation_occurred = True
                highest_threshold_exceeded = threshold
                inundation_year = exceedance_year
                break
            highest_threshold_exceeded = threshold
            
    # Check the next threshold if applicable
    if highest_threshold_exceeded < max(available_thresholds):
        next_threshold_idx = available_thresholds.index(highest_threshold_exceeded) + 1
        if next_threshold_idx < len(available_thresholds):
            next_threshold = available_thresholds[next_threshold_idx]
            next_file_path = inundation_map[next_threshold]
            next_threshold_inundation = check_inundation(next_file_path, input_lat, input_lon, window_size=10)
            next_inundation_status = "Yes" if next_threshold_inundation else "No"
            next_output = f"Next Unmet Threshold: {next_threshold} feet, Year Exceeded: Unknown, Inundation: {next_inundation_status}"
        else:
            next_output = "No higher threshold available to check."

    place_name = get_place_name(input_lat, input_lon)
    if threshold_exceedance:
        for threshold, year, inundated in threshold_exceedance:
            inundation_status = "Yes" if inundated else "No"
            output += f"Threshold: {threshold} feet, Year Exceeded: {year}, Inundation: {inundation_status}"

    return output, slr_df, highest_threshold_exceeded, next_output, inundation_occurred, inundation_year, coordinates, place_name, next_inundation_status
    
# In[19]:

def find_safe_locations(input_lat, input_lon, inundation_map, inundation_occurance, highest_threshold_exceeded, find_radius=0.05, find_step=0.01, max_location=3, max_coord=5, whether_find_loc=True, whether_find_coord=False):
    if inundation_occurance is None:
        return None, None, None, None
    nearby_safe_locations = None
    nearby_safe_coordinates = None
    min_safe_distance = None
    safe_distance = None
    safe_location = ""
    safe_distances = []
    if whether_find_loc:
        nearby_safe_locations = find_closest_uninundated_locations(input_lat, input_lon, inundation_map[highest_threshold_exceeded+1], search_radius=find_radius, step_size=find_step, max_locations=max_location)
        if nearby_safe_locations is not None:
            for lat, lon, place in nearby_safe_locations:
                if lat and lon:
                    safe_distance = geodesic((input_lat, input_lon), (lat, lon)).miles
                    safe_distances.append(safe_distance)
                    safe_location += f"{place} (coordinate: {lat}, {lon}). Distance: {safe_distance:.2f} miles."
                else:
                    safe_location += "No safe location found within the search radius."
            if safe_distances:
                min_safe_distance = np.min(safe_distances)
    return safe_location, nearby_safe_locations, nearby_safe_coordinates, min_safe_distance

# In[20]:


#Risk Assessment Score
weights = {
    'slr': 0.1, #Maximum SSH of the ocean near the location
    'inundation': 0.5, #1, 0.5, 0, whether the inundation will be inundated
    'proximity_safe_location': 0.1, #distance between the closest safe neighborhood and your house. Change it to only when knowing that the house is inundated, only when inundation score yields 0.5 or 0.
    'storm': 0.15, #storm frequency
    'flood': 0.15 #historical flood count
}

def calculate_risk_score(slr_score, inundation_score, proximity_score, storm_score, flood_score, weights):
    # Combine the normalized risks using weights
    if not slr_score:
        total_score = "Not enough data"
        return total_score
    total_score = (
        weights['slr'] * slr_score +
        weights['inundation'] * inundation_score +
        weights['proximity_safe_location'] * proximity_score +
        weights['storm'] * storm_score +
        weights['flood'] * flood_score
    )
    # Scale to 0-100
    total_score *= 100
    return total_score

def calculate_slr_score(max_ssh_change, max_expected_slr=3.5):
    if not max_ssh_change:
        slr_score = None
        return slr_score
    # Normalize SLR (ensure it doesn't exceed max_expected_slr)
    normalized_slr = min(max_ssh_change / max_expected_slr, 1)
    # Invert to make higher values safer
    slr_score = 1 - normalized_slr
    return slr_score #Change it

def calculate_inundation_score(inundation_year):
    if inundation_year is None:
        # Not inundated by 2100
        return 1
    elif inundation_year <= 2050:
        # Inundated before or in 2050
        return 0
    elif inundation_year > 2050 and inundation_year <= 2100:
        # Inundated after 2050 but before or in 2100
        return 0.5
    else:
        # Inundation year beyond 2100 (unlikely in your dataset)
        return 1

def calculate_proximity_score(min_safe_distance, inundation_year, max_distance=2):
    if not min_safe_distance:
        min_safe_distance = 2
    if inundation_year:
        # Normalize distance (ensure it doesn't exceed max_distance)
        normalized_distance = min(min_safe_distance / max_distance, 1)
        # Invert to make closer distances safer
        proximity_score = 1 - normalized_distance
        return proximity_score
    else:
        return 1
        
def get_county_and_state(lat, lon):
    try:
        time.sleep(1)
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location and location.address:
            address = location.raw['address']
            county_name = address.get('county', 'Unknown')
            state_name = address.get('state', 'Unknown')
            return county_name, state_name
        else:
            return "Unknown", "Unknown"
    except Exception as e:
        return "Unknown", "Unknown"

def normalize_county_name(county_name):
    # Remove common suffixes
    suffixes = ['County', 'Parish', 'Municipality', 'Borough', 'Census Area', 'City and Borough']
    for suffix in suffixes:
        if suffix in county_name:
            county_name = county_name.replace(suffix, '')
    # Strip whitespace and convert to uppercase
    county_name = county_name.strip().upper()
    return county_name

usecols = [
    'BEGIN_LAT', 'BEGIN_LON', 'EVENT_TYPE', 'BEGIN_DATE_TIME',
    'EVENT_ID', 'CZ_NAME', 'CZ_TYPE', 'STATE'
]
storm_and_flood_df = pd.read_csv(storm_and_flood_csv, usecols=usecols)
def get_storm_and_flood_frequency(lat, lon, df, radius_km=20):

    # Filter for storm-related events
    storm_types = ['Thunderstorm Wind', 'Tornado', 'Hail', 'Hurricane']
    storm_events = df[df['EVENT_TYPE'].isin(storm_types)].copy()

    # Filter for flood-related events
    flood_types = ['Flood', 'Flash Flood', 'Coastal Flood']
    flood_events = df[df['EVENT_TYPE'].isin(flood_types)].copy()


    # Handle events with coordinates
    storm_events_geo = storm_events.dropna(subset=['BEGIN_LAT', 'BEGIN_LON']).copy()
    flood_events_geo = flood_events.dropna(subset=['BEGIN_LAT', 'BEGIN_LON']).copy()

    # Create geometries
    storm_geometry = [Point(xy) for xy in zip(storm_events_geo['BEGIN_LON'], storm_events_geo['BEGIN_LAT'])]
    flood_geometry = [Point(xy) for xy in zip(flood_events_geo['BEGIN_LON'], flood_events_geo['BEGIN_LAT'])]

    storm_gdf = gpd.GeoDataFrame(storm_events_geo, geometry=storm_geometry, crs='EPSG:4326')
    flood_gdf = gpd.GeoDataFrame(flood_events_geo, geometry=flood_geometry, crs='EPSG:4326')

    # Create a buffer around the input point
    center_point = Point(lon, lat)
    center_gdf = gpd.GeoDataFrame(index=[0], geometry=[center_point], crs='EPSG:4326')
    buffer = center_gdf.to_crs(epsg=3857).buffer(radius_km * 1000)
    buffer = buffer.to_crs(epsg=4326).unary_union
   
    # Find events within the buffer
    storms_nearby = storm_gdf[storm_gdf.geometry.within(buffer)]
    floods_nearby = flood_gdf[flood_gdf.geometry.within(buffer)]


    county_storm_events = storm_events.iloc[0:0].copy()
    county_flood_events = flood_events.iloc[0:0].copy()                     
    county_name, state_name = get_county_and_state(lat, lon)
    # Normalize state name
    #state_abbr = get_state_abbreviation(state_name)
    if county_name != "Unknown":
        
        county_name_norm = normalize_county_name(county_name)
        # Normalize 'CZ_NAME' in the dataset
        storm_events['CZ_NAME_NORM'] = storm_events['CZ_NAME'].str.upper().str.strip()
        flood_events['CZ_NAME_NORM'] = flood_events['CZ_NAME'].str.upper().str.strip()

        # Remove 'County' etc. from 'CZ_NAME_NORM'
        storm_events['CZ_NAME_NORM'] = storm_events['CZ_NAME_NORM'].apply(normalize_county_name)
        flood_events['CZ_NAME_NORM'] = flood_events['CZ_NAME_NORM'].apply(normalize_county_name)

        # Get events affecting the county
        county_storm_events = storm_events[
            (storm_events['CZ_NAME_NORM'] == county_name_norm) &
            (storm_events['CZ_TYPE'] == 'C') &
            (storm_events['STATE'] == state_name.upper().strip())
        ]

        county_flood_events = flood_events[
            (flood_events['CZ_NAME_NORM'] == county_name_norm) &
            (flood_events['CZ_TYPE'] == 'C') &
            (flood_events['STATE'] == state_name.upper().strip())
        ]
    
    # Combine events and remove duplicates
    storm_events_combined = pd.concat([storms_nearby, county_storm_events], ignore_index=True)
    flood_events_combined = pd.concat([floods_nearby, county_flood_events], ignore_index=True)

    # Remove duplicates based on 'EVENT_ID'
    storm_events_combined = storm_events_combined.drop_duplicates(subset='EVENT_ID')
    flood_events_combined = flood_events_combined.drop_duplicates(subset='EVENT_ID')
    
    # Return the counts
    return len(storm_events_combined), len(flood_events_combined)

def calculate_storm_score(storm_event_count, max_storm_event_count=45):
    if not storm_event_count:
        storm_event_count = 0
    normalized_storm_event_count = min(storm_event_count / max_storm_event_count, 1)
    storm_score = 1 - normalized_storm_event_count
    return storm_score

def calculate_flood_score(flood_event_count, max_flood_events=15):
    if not flood_event_count:
        flood_event_count = 0
    normalized_flood_events = min(flood_event_count / max_flood_events, 1)
    flood_score = 1 - normalized_flood_events
    return flood_score

app = Flask(__name__)

CORS(app)

@app.route('/')
def home():
    return "Welcome to the SLR/Inundation Analysis API. Use the /analyze endpoint to run analysis."

# Function to get user inputs and run the SLR/inundation analysis
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
        
    if request.method == 'OPTIONS':
        return ''
        
    # Expecting the following input JSON structure from the client
    user_input = request.get_json()
    
        
    # Get user input parameters
    input_lat = user_input.get('latitude')
    input_lon = user_input.get('longitude')
    user_input_state = user_input.get('state')
    desired_zoom_factor = user_input.get('zoom_factor', 0.0005)
    find_radius = user_input.get('find_radius', 0.02)
    find_step = user_input.get('find_step', 0.005)
    number_locations = user_input.get('number_locations', 3)
    
    # print(input_lat, input_lon, user_input_state, desired_zoom_factor, find_radius, find_step, number_locations)
        
    # Run the SLR analysis functions (example with SSP585 scenario)
    inundation_maps = state_inundation_maps.get(user_input_state)
    
    if not inundation_maps:
        return jsonify({'error': f"This region has no data."}), 400

    # Example of running one scenario (SSP585)
    result_ssp585, ssh_change_df_ssp585, highest_threshold_ssp585, next_output_ssp585, inundation_occurred_ssp585, inundation_year_ssp585, coordinates_ssp585, place_name_ssp585, next_inundation_status_ssp585 = process_scenario(
        "SSP585", file_path_ar6_ssp585, inundation_maps, input_lat, input_lon)
    if ssh_change_df_ssp585:
        max_ssh_change_df_ssp585 = ssh_change_df_ssp585.loc[ssh_change_df_ssp585['Sea Level Rise'].idxmax()]
        max_ssh_change_ssp585 = max_ssh_change_df_ssp585['Sea Level Rise']
        max_ssh_change_ssp585 = round(max_ssh_change_ssp585, 3)
        max_ssh_change_year_ssp585 = max_ssh_change_df_ssp585['Year']
    else:
        max_ssh_change_ssp585 = None
        max_ssh_change_year_ssp585 = None
    # Find safe locations (optional based on user input)
    safe_location_ssp585, nearby_safe_locations_ssp585, nearby_safe_coordinates_ssp585, min_safe_distance_ssp585 = find_safe_locations(
        input_lat, input_lon, inundation_maps, inundation_occurred_ssp585, highest_threshold_ssp585,
        find_radius=find_radius, find_step=find_step, max_location=number_locations, max_coord=0,
        whether_find_loc=True, whether_find_coord=False)
    
    slr_score = calculate_slr_score(max_ssh_change_ssp585)
    if slr_score:
        slr_score_100scale = round(slr_score*100)
    else:
        slr_score_100scale = "NA"
    inundation_score = calculate_inundation_score(inundation_year_ssp585)
    inundation_score_100scale = round(inundation_score*100)
    proximity_score = calculate_proximity_score(min_safe_distance_ssp585, inundation_year_ssp585)
    proximity_score_100scale = round(proximity_score*100)
    storm_count, flood_count = get_storm_and_flood_frequency(input_lat, input_lon, storm_and_flood_df)
    storm_score = calculate_storm_score(storm_count)
    storm_score_100scale = round(storm_score*100)
    flood_score = calculate_flood_score(flood_count)
    flood_score_100scale = round(flood_score*100)
    risk_assessment_score = calculate_risk_score(slr_score, inundation_score, proximity_score, storm_score, flood_score, weights)
    if isinstance(risk_assessment_score, (int, float)):
        risk_assessment_score = round(risk_assessment_score)
    else:
        risk_assessment_score = risk_assessment_score
    inundation_occur_ssp585 = "Yes" if inundation_occurred_ssp585 else "No"

    Newyork_RAS = 83
    Boston_RAS = 85
    Miami_RAS = 69
    California_RAS = 80
    major_city_RAS = {
        "New York": Newyork_RAS,
        "Boston": Boston_RAS,
        "Miami": Miami_RAS,
        "California": California_RAS
    }

    
    response = {
        "latitude": input_lat,
        "longitude": input_lon,
        "state": user_input_state,
        "zoom_factor": desired_zoom_factor,
        "find_radius": find_radius,
        "find_step": find_step,
        "number_locations": number_locations,
        "coordinates": coordinates_ssp585,
        "place_name": place_name_ssp585,
        "inundation_occur": inundation_occur_ssp585,
        "inundation_year": str(inundation_year_ssp585),
        "future_inundation_occur": next_inundation_status_ssp585,
        "ssp585_result": result_ssp585,
        "maximum_slr": max_ssh_change_ssp585,
        "maximum_slr_year": max_ssh_change_year_ssp585,
        "risk_assessment_score": risk_assessment_score,
        "slr_score": slr_score_100scale,
        "inundation_score": inundation_score_100scale,
        "proximity_score": proximity_score_100scale,
        "storm_score": storm_score_100scale,
        "flood_score": flood_score_100scale,
        "major_city_ras": major_city_RAS,
        "safe_locations": safe_location_ssp585,
        # "plot_original": plot_original,
        # "plot_safe_location": plot_safe_location
    }
    
    insert_data(response)
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)

