# Databricks notebook source
# MAGIC %pip install folium

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import sum, col, desc
from pyspark.sql.functions import mean
import pyspark
import folium

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
trips_df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "TRIPS") \
  .load()

# COMMAND ----------

#picked a random trip
trip = trips_df.filter(trips_df["trip_id"] == '661424983605-2020-12-01')
print(trip.count())

# COMMAND ----------

from pyspark.sql.functions import element_at
trip = trip.select(col("LAT"), col("LNG"), col("TRIP_ID"))
display(trip)
'''
x = element_at(mean(trip["Lat"]),0)

print(x)
print(type(x))
'''


# COMMAND ----------

pandasDF = trip.toPandas()
latMean = pandasDF['LAT'].mean()
lngMean = pandasDF['LNG'].mean()
mapx = folium.Map(location=[latMean, lngMean], zoom_start=14, control_scale=True)

for index, location_info in pandasDF.iterrows():
    folium.Marker([location_info["LAT"], location_info["LNG"]], popup=location_info["TRIP_ID"]).add_to(mapx)
display(mapx)

# COMMAND ----------

