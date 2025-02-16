# Databricks notebook source
# MAGIC %pip install folium

# COMMAND ----------

import map_tools_v6 as map_tools
import routing_tools as routing_tools
import general_tools_v3 as general_tools
import arc_tools_v2 as arc_tools
import pandas as pd
import numpy as np
from pyspark.sql.functions import col


map_tools.version()
routing_tools.version()
general_tools.version()
arc_tools.version()

# COMMAND ----------

# Setup SnowFlake connection
options = {
"sfUrl": "ab43836.west-europe.azure.snowflakecomputing.com",
"sfUser": "r****", # Email address of your personal
"sfPassword": "", # Password you choose when first login to Snowflake
"sfDatabase": "GATECH",
"sfSchema": "GROUP_7", # Replace * by your group number
"sfWarehouse": "GATECH_WH"
}

# COMMAND ----------

# Read the data from EVENTS table.
master_segments_sdf = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "G7_SEGMENTS_V8") \
.load()

master_segments_sdf = master_segments_sdf.select(
    "ARC_INDEX_HASH", 
    "SEGMENT_ID_DS",
    "DISTANCE_M", 
    "SEQ_ON_SEGMENT", 
    "EVENT_COUNT", 
    "CRASH_COUNT"
)

# COMMAND ----------

# Read the data from EVENTS table.
routes_sdf = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "NXTRAQ_ROUTES") \
.load()

# COMMAND ----------

routes_df = arc_tools.add_arc_columns(routes_sdf).toPandas()
display(routes_df)

unique_routes_df = routes_df["ROUTE"].unique()
print(len(unique_routes_df))

# COMMAND ----------

# MAGIC %md
# MAGIC # Routes Visualization

# COMMAND ----------

colors = ['green', 'orange', 'red', 'maroon']

for route in ['141_34']:#unique_routes_df[:3]:
    print(route)
    route_itineraries = np.sort(routes_df[routes_df["ROUTE"] == route]["ITINERARY_ID"].unique())
    # colors = general_tools.generate_color_palette(len(route_itineraries))
    # print(colors)
    
    m = None
    for idx,itinerary_id in route_itineraries:
        print(itinerary_id)
        m = map_tools.generate_map(routes_df[routes_df["TRIP_ID"] == f'{route}_{itinerary_id}'], m, colors[idx], False)

    m.save(f'routes_visualizer_maps/{route}.html')
        

# COMMAND ----------

file_path = '/Workspace/Repos/gatech-group7/gatech-group7/Omar/routes_visualizer_maps/165_161.html'

with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Display the HTML content in the notebook
displayHTML(m._repr_html_())

# COMMAND ----------

# MAGIC %md
# MAGIC # Fill holes in Arc Routes

# COMMAND ----------

models = ['G7_RISK_MODEL1_RF', 'G7_RISK_MODEL2_XGB1', 'G7_RISK_MODEL3_XGB2', 'G7_RISK_MODEL4_ZIP']

for model in models:
    print(f'adding predictions of {model}')
    risk_model_sdf = spark.read \
    .format("snowflake") \
    .options(**options) \
    .option("dbtable", model) \
    .load()

    master_segments_sdf = master_segments_sdf.join(risk_model_sdf, master_segments_sdf['ARC_INDEX_HASH'] == risk_model_sdf['ARC_ID_HASH'], 'left').drop('ARC_ID_HASH').withColumnRenamed('PREDICTION', model)

# COMMAND ----------

display(master_segments_sdf)

# COMMAND ----------

def expand_arc_route(arc_route, master_segments_sdf):
    """
    Makes sure that all arc indexes that the
    route passes through are included in the arc_route,
    this is to compensate for the routing engine
    returning points that jump over arc indexes. This
    uses the segments_table

    Expands the given arc route to include all arc indexes that the route passes through.

    Args:
    arc_route (list): List of arc indexes.
    segments_df (pd.DataFrame): DataFrame containing segments data with columns 'SEGMENT_ID_DS' and 'SEQ_ON_SEGMENT'.

    Returns:
    pd.DataFrame: DataFrame containing all arcs visited.
    """
    # Dictionary to hold the min and max SEQ_ON_SEGMENT for each SEGMENT_ID_DS
    segment_ranges = {}

    original_arcs_visited_df = master_segments_sdf.filter(col('ARC_INDEX_HASH').isin(arc_route)).toPandas()
    # print(f'{len(arc_route)} arcs provided, {original_arcs_visited_df.shape[0]} arcs found')

    # Update the segment_ranges dictionary
    for _, row in original_arcs_visited_df.iterrows():
        segment_id = row['SEGMENT_ID_DS']
        seq_num = row['SEQ_ON_SEGMENT']

        if segment_id in segment_ranges:
            segment_ranges[segment_id] = (
                min(segment_ranges[segment_id][0], seq_num),
                max(segment_ranges[segment_id][1], seq_num)
            )
        else:
            segment_ranges[segment_id] = (seq_num, seq_num)

    # print(segment_ranges)

    segment_ids_visited = list(segment_ranges.keys())
    
    # print(f'Filtering master segments df for segments={segment_ids_visited}')
    segments_visited_df = master_segments_sdf.filter(col('SEGMENT_ID_DS').isin(segment_ids_visited)).toPandas()
    # print(f'--> Done: {len(segments_visited_df)} found')

    # DataFrame to hold the result of the filter
    arcs_visited = pd.DataFrame()

    # Iterate over the segment_ranges dictionary and filter segments_visited_df
    for segment_id, (min_seq, max_seq) in segment_ranges.items():
        filtered_segments = segments_visited_df[
            (segments_visited_df['SEGMENT_ID_DS'] == segment_id) & 
            (segments_visited_df['SEQ_ON_SEGMENT'] >= min_seq) & 
            (segments_visited_df['SEQ_ON_SEGMENT'] <= max_seq)
        ]
        # print(f'segment_id:{segment_id} ({min_seq}-{max_seq}) = {len(filtered_segments)}')
        # Append the result to arcs_visited DataFrame
        arcs_visited = pd.concat([arcs_visited, filtered_segments])

    # New step: Check for missing arc indexes and add them with null values
    missing_arc_indexes = [arc for arc in arc_route if arc not in arcs_visited['ARC_INDEX_HASH'].tolist()]
    for missing_arc in missing_arc_indexes:
        missing_row = { 'ARC_INDEX_HASH': missing_arc }
        arcs_visited = pd.concat([arcs_visited, pd.DataFrame([missing_row])])

    return arcs_visited, segment_ranges



# COMMAND ----------

# given filtered_segments_df, arc_route
print(f'unique routes: {len(unique_routes_df)}')

columns = ['Route', 'Itinerary_ID', 'Num_Itinerary_Arcs', 'Num_Expanded_Arcs', 'Risk']
route_scores_df = pd.DataFrame(columns=columns)

for route in unique_routes_df:
    print(route)
    route_itineraries = np.sort(routes_df[routes_df["ROUTE"] == route]["ITINERARY_ID"].unique())
    
    for idx,itinerary_id in enumerate(route_itineraries):
        itinerary_arcs = routes_df[routes_df["TRIP_ID"] == f'{route}_{itinerary_id}']
        arcs_visited = itinerary_arcs['ARC_INDEX_HASH'].values.tolist()
        expanded_itinerary_arcs,ranges = expand_arc_route(arcs_visited, master_segments_sdf)

        itinerary_data = {
            'Route': route,
            'Itinerary_ID': itinerary_id,
            'Num_Itinerary_Arcs': len(itinerary_arcs),
            'Num_Expanded_Arcs': len(expanded_itinerary_arcs),
            'Risk_BasicCrashCount': np.sum(expanded_itinerary_arcs['CRASH_COUNT']),
        }
        for model in models:
            itinerary_data[model] = np.sum(expanded_itinerary_arcs[model])
        route_scores_df = pd.concat([route_scores_df, pd.DataFrame([itinerary_data])])


# COMMAND ----------

# route_scores.keys
ttl = [ranges[k][1]-ranges[k][0]+1 for k in ranges.keys()]
print(np.sum(ttl))

# COMMAND ----------

display(route_scores_df[route_scores_df['Route']=='141_34'])
# display(route_scores_df)
