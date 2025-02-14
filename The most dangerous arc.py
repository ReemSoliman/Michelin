# Databricks notebook source
# MAGIC %pip install folium

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import sum, col, desc, mean
import folium
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, mean

# COMMAND ----------

options = {
  "sfUrl": "ab43836.west-europe.azure.snowflakecomputing.com",
  "sfUser": "rsoliman3@gatech.edu", # Email address of your personal
  "sfPassword": "subhanAllah2193&", # Password you choose when first login to Snowflake
  "sfDatabase": "GATECH",
  "sfSchema": "GROUP_7", # Replace * by your group number
  "sfWarehouse": "GATECH_WH"
}

# COMMAND ----------

events_df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "EVENTS") \
  .load()


# COMMAND ----------

counted_by_arcs_id = events_df.groupBy("ARC_INDEX_HASH_START", "ARC_INDEX_HASH_END").count()#("SEGMENT_ID_DS").count()
#print(type(counted_by_seg_id))
#counted_by_seg_id_pd = counted_by_seg_id.toPandas()
#counted_by_seg_id_pd = counted_by_seg_id_pd.sort_values("count",ascending=False)

counted_by_arcs_id = counted_by_arcs_id.orderBy(col("count").desc())
#print(type(counted_by_seg_id))
counted_by_arcs_id = counted_by_arcs_id.limit(15)
#print(type(counted_by_seg_id))
display(counted_by_arcs_id)


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


arc_pd = counted_by_seg_id.toPandas()
mapx = folium.Map(location=[arc_pd.iloc[0,4], arc_pd.iloc[0,3]], zoom_start=14, control_scale=True)
#print("starlng", arc_pd.iloc[0,4])
#print("starlat", arc_pd.iloc[0,3])
#print(arc_pd[0,2])
#for index, location_info in pandasDF.iterrows():
folium.Marker([arc_pd.iloc[0,4], arc_pd.iloc[0,3]], popup=arc_pd.iloc[0,2]).add_to(mapx)
folium.Marker([arc_pd.iloc[0,6], arc_pd.iloc[0,5]], popup=arc_pd.iloc[0,2]).add_to(mapx)
display(mapx)

# COMMAND ----------

most_dan_arc_events = events_df.filter(col("ARC_INDEX_HASH_START")==arc_pd.iloc[0,0])

counted_by_event_type = most_dan_arc_events.groupBy("EVENT_TYPE_DESC").count()
#counted.display()
counted_by_event_type_pd = counted_by_event_type.toPandas()

plt.figure(figsize=(10,5))

counted_by_event_type_pd = counted_by_event_type_pd.sort_values("EVENT_TYPE_DESC")
sns.barplot(data=counted_by_event_type_pd,x="EVENT_TYPE_DESC", y="count")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

counted_by_weather = most_dan_arc_events.groupBy("CONTEXT_WEATHER").count()
#counted.display()
counted_by_weather_pd = counted_by_weather.toPandas()

plt.figure(figsize=(10,5))

counted_by_weather_pd = counted_by_weather_pd.sort_values("CONTEXT_WEATHER")
sns.barplot(data=counted_by_weather_pd,x="CONTEXT_WEATHER", y="count")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

'''counted_by_circulation_poi = most_dan_arc_events.groupBy("INTERACTION_POI_DS").count()
counted_by_circulation_poi_pd = counted_by_circulation_poi.toPandas()

plt.figure(figsize=(10,5))

counted_by_circulation_poi_pd = counted_by_circulation_poi_pd.sort_values("INTERACTION_POI_DS")
print(counted_by_circulation_poi_pd.count())
sns.barplot(data=counted_by_circulation_poi_pd,x="INTERACTION_POI_DS", y="count")
plt.xticks(rotation=45)
plt.show()
'''

# COMMAND ----------

display(most_dan_arc_events.select(mean('SAMPLE_SPEED')) )
row = most_dan_arc_events.agg({'SAMPLE_SPEED':'max'})
display(row)

# COMMAND ----------

display(most_dan_arc_events.select(mean('SPEED_LIMIT_DS')) )