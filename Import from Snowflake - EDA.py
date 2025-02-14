# Databricks notebook source
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import sum, col, desc

# COMMAND ----------

# Use dbutils secrets to get Snowflake credentials.
options = {
  "sfUrl": "ab43836.west-europe.azure.snowflakecomputing.com",
  "sfUser": "rsoliman3@gatech.edu", # Email address of your personal
  "sfPassword": "subhanAllah2193&", # Password you choose when first login to Snowflake
  "sfDatabase": "GATECH",
  "sfSchema": "GROUP_7", # Replace * by your group number
  "sfWarehouse": "GATECH_WH"
}

# COMMAND ----------

# Read the data written by the previous cell back.
events_df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "EVENTS") \
  .load()

#display(events_df.limit(30))

# COMMAND ----------


#print(type(df))

#print(df.count())
#df_sample = df.sample(True, 0.00001, seed=1234) #df.filter(df.TRIP_DATE == "2020-12-01")
#print(df_sample.count())
#events_df = df_sample.toPandas() #limit(1000000)
#print(type(events_df))

#xtick_loc = plot.get_xticks()
#xtick_labels = plot.get_xticklabels()
#plot.set_xticklabels(labels = xtick_labels, rotation=45)


# COMMAND ----------

counted_by_county = events_df.groupBy("ROAD_DEPARTEMENT_DS").count()
#counted.display()
counted_by_county_pd = counted_by_county.toPandas()

plt.figure(figsize=(20,5))

counted_by_county_pd = counted_by_county_pd.sort_values("ROAD_DEPARTEMENT_DS")
sns.barplot(data=counted_by_county_pd,x="ROAD_DEPARTEMENT_DS", y="count")
plt.xticks(rotation=45)
plt.show()
#plt.figure(figsize=(25,5))
#plot = sns.countplot(x = 'ROAD_DEPARTEMENT_DS', data = events_df, order=events_df['ROAD_DEPARTEMENT_DS'].value_counts().index)

# COMMAND ----------

'''plt.figure(figsize=(25,5))
plot = sns.countplot(x = 'ROAD_DEPARTEMENT_DS',  data = events_df, order=events_df['ROAD_DEPARTEMENT_DS'].value_counts().index, hue='EVENT_TYPE_DESC')
'''

# COMMAND ----------


#plot = sns.countplot(x = 'EVENT_TYPE_DESC',  data = events_df, order=events_df['EVENT_TYPE_DESC'].value_counts().index)

counted_by_event_type = events_df.groupBy("EVENT_TYPE_DESC").count()
#counted.display()
counted_by_event_type_pd = counted_by_event_type.toPandas()

plt.figure(figsize=(10,5))

counted_by_event_type_pd = counted_by_event_type_pd.sort_values("EVENT_TYPE_DESC")
sns.barplot(data=counted_by_event_type_pd,x="EVENT_TYPE_DESC", y="count")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------


#plt.figure(figsize=(25,5))
#plot = sns.countplot(x = 'CONTEXT_ROAD_CYCLE',  data = events_df, order=events_df['CONTEXT_ROAD_CYCLE'].value_counts().index)
counted_by_road_cycle = events_df.groupBy("CONTEXT_ROAD_CYCLE").count()
#counted.display()
counted_by_road_cycle_pd = counted_by_road_cycle.toPandas()

plt.figure(figsize=(10,5))

counted_by_road_cycle_pd = counted_by_road_cycle_pd.sort_values("CONTEXT_ROAD_CYCLE")
sns.barplot(data=counted_by_road_cycle_pd,x="CONTEXT_ROAD_CYCLE", y="count")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

#plt.figure(figsize=(25,5))
#plot = sns.countplot(x = 'CONTEXT_WEATHER',  data = events_df, order=events_df['CONTEXT_WEATHER'].value_counts().index)

counted_by_weather = events_df.groupBy("CONTEXT_WEATHER").count()
#counted.display()
counted_by_weather_pd = counted_by_weather.toPandas()

plt.figure(figsize=(10,5))

counted_by_weather_pd = counted_by_weather_pd.sort_values("CONTEXT_WEATHER")
sns.barplot(data=counted_by_weather_pd,x="CONTEXT_WEATHER", y="count")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

#plt.figure(figsize=(25,5))
#plot = sns.countplot(x = 'CIRCULATION_POI_DS',  data = events_df, order=events_df['CIRCULATION_POI_DS'].value_counts().index)
counted_by_circulation_poi = events_df.groupBy("CIRCULATION_POI_DS").count()
counted_by_circulation_poi_pd = counted_by_circulation_poi.toPandas()

plt.figure(figsize=(10,5))

counted_by_circulation_poi_pd = counted_by_circulation_poi_pd.sort_values("CIRCULATION_POI_DS")
sns.barplot(data=counted_by_circulation_poi_pd,x="CIRCULATION_POI_DS", y="count")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

#plt.figure(figsize=(25,5))
#plot = sns.countplot(x = 'CONTEXT_SLOPE',  data = events_df, order=events_df['CONTEXT_SLOPE'].value_counts().index)
counted_by_context_slope = events_df.groupBy("CONTEXT_SLOPE").count()
counted_by_context_slope_pd = counted_by_context_slope.toPandas()

plt.figure(figsize=(10,5))

counted_by_context_slope_pd = counted_by_context_slope_pd.sort_values("CONTEXT_SLOPE")
sns.barplot(data=counted_by_context_slope_pd,x="CONTEXT_SLOPE", y="count")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------


counted_by_date = events_df.groupBy("TRIP_DATE").count()
counted_by_date_pd = counted_by_date.toPandas()
plt.figure(figsize=(25,5))
counted_by_date_pd = counted_by_date_pd.sort_values("TRIP_DATE")
sns.barplot(data=counted_by_date_pd,x="TRIP_DATE", y="count")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

#the most dangerous start and end arcs
counted_by_arcs_id = events_df.groupBy("ARC_INDEX_HASH_START", "ARC_INDEX_HASH_END").count()#("SEGMENT_ID_DS").count()
#print(type(counted_by_seg_id))
#counted_by_seg_id_pd = counted_by_seg_id.toPandas()
#counted_by_seg_id_pd = counted_by_seg_id_pd.sort_values("count",ascending=False)

counted_by_arcs_id = counted_by_arcs_id.orderBy(col("count").desc())
#print(type(counted_by_seg_id))
counted_by_arcs_id = counted_by_arcs_id.limit(15)
#print(type(counted_by_seg_id))
display(counted_by_arcs_id)
#counted_by_seg_id.select(sum(counted_by_seg_id["count"])).show()
#type(counted_by_seg_id)


# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit
counted_by_seg_id = counted_by_arcs_id.withColumn('start',  F.split(counted_by_arcs_id['ARC_INDEX_HASH_START'], '_').getItem(0)) \
         .withColumn('end',    F.split(counted_by_arcs_id['ARC_INDEX_HASH_START'], '_').getItem(1))

counted_by_seg_id = counted_by_seg_id.withColumn('start_lon', F.substring(counted_by_seg_id['start'],  1,  9).cast("Integer"))\
	.withColumn('start_lat', F.substring(counted_by_seg_id['start'], 10,  9).cast("Integer"))\
	.withColumn('end_lon',   F.substring(counted_by_seg_id['end'],  1, 9).cast("Integer"))\
	.withColumn('end_lat',   F.substring(counted_by_seg_id['end'], 10, 9).cast("Integer")).drop('start', 'end')

counted_by_seg_id = counted_by_seg_id.withColumn('end_lon',   F.col("end_lon")  /1000000- 500)\
	.withColumn('start_lon', F.col("start_lon")/1000000- 500)\
	.withColumn('start_lat', F.col("start_lat")/1000000- 500)\
	.withColumn('end_lat',   F.col("end_lat")  /1000000- 500)

counted_by_seg_id = counted_by_seg_id.withColumn('start_lon', F.col('start_lon').cast("Double"))\
	.withColumn('start_lat', F.col('start_lat').cast("Double"))\
	.withColumn('end_lon', F.col('end_lon').cast("Double"))\
	.withColumn('end_lat', F.col('end_lat').cast("Double")) 

counted_by_seg_id = counted_by_seg_id.withColumn('geom_wkt', F.concat(lit('LINESTRING('),col('start_lon'), lit(' '), col('start_lat'), lit(','), col('end_lon'), lit(' '),  col('end_lat'),lit(')')))
display(counted_by_seg_id)

# COMMAND ----------

# MAGIC %pip install folium
# MAGIC import folium
# MAGIC arc_pd = counted_by_seg_id.toPandas()
# MAGIC mapx = folium.Map(location=[arc_pd.iloc[0,4], arc_pd.iloc[0,3]], zoom_start=14, control_scale=True)
# MAGIC print("starlng", arc_pd.iloc[0,4])
# MAGIC print("starlat", arc_pd.iloc[0,3])
# MAGIC print(arc_pd[0,2])
# MAGIC #for index, location_info in pandasDF.iterrows():
# MAGIC folium.Marker([arc_pd.iloc[0,4], arc_pd.iloc[0,3]], popup=arc_pd[0,2]).add_to(mapx)
# MAGIC folium.Marker([arc_pd.iloc[0,6], arc_pd.iloc[0,5]], popup=arc_pd[0,2]).add_to(mapx)
# MAGIC display(mapx)

# COMMAND ----------

arcs_dang_segments= counted_by_arcs_id.join(events_df , counted_by_arcs_id.ARC_INDEX_HASH_START == events_df.ARC_INDEX_HASH_START, "inner")
#counted_by_seg_id.SEGMENT_ID_DS == events_df.SEGMENT_ID_DS, "inner")

type(arcs_dang_segments)
arcs_dang_segments = arcs_dang_segments.select(counted_by_arcs_id["ARC_INDEX_HASH_START"],"count", "START_LAT","START_LNG", "END_LAT", "END_LNG") #"SEGMENT_ID_DS"
#display(arcs_dang_segments)
df_dist = arcs_dang_segments.dropDuplicates()
print(df_dist.count())

#display(df_dist)

# COMMAND ----------

#join with poi table to find out if there is a relation between arcs with high events and their distance from poi
df_poi = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "POI") \
  .load()

dangerous_arc_start= counted_by_seg_id.join(df_poi , counted_by_seg_id.ARC_INDEX_HASH_START == df_poi.ARC_ID_HASH, "inner")
#counted_by_seg_id.SEGMENT_ID_DS == events_df.SEGMENT_ID_DS, "inner")

display(dangerous_arc_start)
'''arcs_dang_segments = arcs_dang_segments.select(counted_by_seg_id["ARC_INDEX_HASH_START"],"count", "START_LAT","START_LNG", "END_LAT", "END_LNG") #"SEGMENT_ID_DS"
#display(arcs_dang_segments)
df_dist = arcs_dang_segments.dropDuplicates()
print(df_dist.count())
'''

# COMMAND ----------

arcs_dang_segments= counted_by_seg_id.join(events_df , counted_by_seg_id.ARC_INDEX_HASH_START == events_df.ARC_INDEX_HASH_START, "inner")
#counted_by_seg_id.SEGMENT_ID_DS == events_df.SEGMENT_ID_DS, "inner")

type(arcs_dang_segments)
arcs_dang_segments = arcs_dang_segments.select(counted_by_seg_id["ARC_INDEX_HASH_START"],"count", "START_LAT","START_LNG", "END_LAT", "END_LNG") #"SEGMENT_ID_DS"
#display(arcs_dang_segments)
df_dist = arcs_dang_segments.dropDuplicates()
print(df_dist.count())

# COMMAND ----------

#safest segments
counted_by_seg_id = events_df.groupBy("SEGMENT_ID_DS").count()
counted_by_seg_id_pd = counted_by_seg_id.toPandas()
counted_by_seg_id_pd = counted_by_seg_id_pd.sort_values("count",ascending=True)
display(counted_by_seg_id_pd[:15])

# COMMAND ----------

##Read Trips Table
df_trips = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "TRIPS") \
  .load()

# COMMAND ----------

'''counted_by_date_trips = df_trips.groupBy("TRIP_DATE").count()
#counted.display()
counted_by_date_trips_pd = counted_by_date_trips.toPandas()

plt.figure(figsize=(25,5))
counted_by_date_trips_pd = counted_by_date_trips_pd.sort_values("TRIP_DATE")
sns.barplot(data=counted_by_date_trips_pd,x="TRIP_DATE", y="count")
plt.xticks(rotation=45)
plt.show()
'''
#count the distinct count of trip per day as each trip consists of multiple segmants 
counted_by_date_trips = df_trips.groupBy("TRIP_DATE").agg(countDistinct('TRIP_ID'))#.show(truncate=False)
counted_by_date_trips_pd = counted_by_date_trips.toPandas()

plt.figure(figsize=(25,5))
counted_by_date_trips_pd = counted_by_date_trips_pd.sort_values("TRIP_DATE")
sns.barplot(data=counted_by_date_trips_pd,x="TRIP_DATE", y="count(TRIP_ID)")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

counted_by_county_trips = df_trips.groupBy("ROAD_DEPARTEMENT_DS").agg(countDistinct('TRIP_ID'))
#counted.display()
counted_by_county_trips_pd = counted_by_county_trips.toPandas()

plt.figure(figsize=(25,5))
counted_by_county_trips_pd = counted_by_county_trips_pd.sort_values("ROAD_DEPARTEMENT_DS")
sns.barplot(data=counted_by_county_trips_pd,x="ROAD_DEPARTEMENT_DS", y="count(TRIP_ID)")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

counted_by_county_trips = df_trips.groupBy("ROAD_DEPARTEMENT_DS").count()
#counted.display()
counted_by_county_trips_pd = counted_by_county_trips.toPandas()

plt.figure(figsize=(25,5))
counted_by_county_trips_pd = counted_by_county_trips_pd.sort_values("ROAD_DEPARTEMENT_DS")
sns.barplot(data=counted_by_county_trips_pd,x="ROAD_DEPARTEMENT_DS", y="count")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Write the data to a Delta table

#df.write.format("delta").saveAsTable("sf_ingest_table")