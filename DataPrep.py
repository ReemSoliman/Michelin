# Databricks notebook source
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import sum, col, desc, mean
#import folium
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

counted_by_arcs_id = events_df.groupBy("ARC_INDEX_HASH_START").count()#("SEGMENT_ID_DS").count()
print(counted_by_arcs_id.count())
#print(type(counted_by_seg_id))
#counted_by_seg_id_pd = counted_by_seg_id.toPandas()
#counted_by_seg_id_pd = counted_by_seg_id_pd.sort_values("count",ascending=False)

counted_by_arcs_id = counted_by_arcs_id.orderBy(col("count").desc())
print(counted_by_arcs_id.count())
#print(type(counted_by_seg_id))
#counted_by_arcs_id = counted_by_arcs_id.limit(15)
#print(type(counted_by_seg_id))
display(counted_by_arcs_id)

# COMMAND ----------


counted_by_arcs_id = counted_by_arcs_id.withColumn('start',  F.split(counted_by_arcs_id['ARC_INDEX_HASH_START'], '_').getItem(0)) \
         .withColumn('end',    F.split(counted_by_arcs_id['ARC_INDEX_HASH_START'], '_').getItem(1))

counted_by_arcs_id = counted_by_arcs_id.withColumn('start_lon', F.substring(counted_by_arcs_id['start'],  1,  9).cast("Integer"))\
	.withColumn('start_lat', F.substring(counted_by_arcs_id['start'], 10,  9).cast("Integer"))\
	.withColumn('end_lon',   F.substring(counted_by_arcs_id['end'],  1, 9).cast("Integer"))\
	.withColumn('end_lat',   F.substring(counted_by_arcs_id['end'], 10, 9).cast("Integer")).drop('start', 'end')

counted_by_arcs_id = counted_by_arcs_id.withColumn('end_lon',   F.col("end_lon")  /1000000- 500)\
	.withColumn('start_lon', F.col("start_lon")/1000000- 500)\
	.withColumn('start_lat', F.col("start_lat")/1000000- 500)\
	.withColumn('end_lat',   F.col("end_lat")  /1000000- 500)

counted_by_arcs_id = counted_by_arcs_id.withColumn('start_lon', F.col('start_lon').cast("Double"))\
	.withColumn('start_lat', F.col('start_lat').cast("Double"))\
	.withColumn('end_lon', F.col('end_lon').cast("Double"))\
	.withColumn('end_lat', F.col('end_lat').cast("Double")) 

counted_by_arcs_id = counted_by_arcs_id.withColumn('geom_wkt', F.concat(lit('LINESTRING('),col('start_lon'), lit(' '), col('start_lat'), lit(','), col('end_lon'), lit(' '),  col('end_lat'),lit(')')))
print(counted_by_arcs_id.count())

# COMMAND ----------

poi_df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "POI") \
  .load()

# COMMAND ----------

arcs = counted_by_arcs_id.join(poi_df, counted_by_arcs_id.ARC_INDEX_HASH_START == poi_df.ARC_ID_HASH ,how="left")

# COMMAND ----------

print(arcs.count())

# COMMAND ----------

#display(arcs.limit(3))

# COMMAND ----------

crashes_df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "CRASHES_2019_2023_MAPMATCHED") \
  .load()

# COMMAND ----------

#join arcs with crashes counted by arc
crashes_count_by_arcs_id = crashes_df.groupBy("ARC_ID_HASH").count()#("SEGMENT_ID_DS").count()
print(crashes_count_by_arcs_id.count())
arcs = arcs.join(crashes_count_by_arcs_id, counted_by_arcs_id.ARC_INDEX_HASH_START == crashes_count_by_arcs_id.ARC_ID_HASH ,how="left")
print(arcs.count())

# COMMAND ----------

display(arcs)