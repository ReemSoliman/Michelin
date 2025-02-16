# Databricks notebook source
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import sum, col, desc, mean, isnan
#import folium
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, mean, when, median, isnan, isnull#, isinf

# COMMAND ----------

options = {
  "sfUrl": "ab43836.west-europe.azure.snowflakecomputing.com",
  "sfUser": "r****", # Email address of your personal
  "sfPassword": "", # Password you choose when first login to Snowflake
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

crashes_df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "CRASHES_2019_2023_MAPMATCHED") \
  .load()  

  

# COMMAND ----------

events_df = events_df.withColumnRenamed("arc_index_hash_start", "ARC_ID_HASH")
counted_by_arcs_id = events_df.groupBy("ARC_ID_HASH").count()
counted_by_arcs_id = counted_by_arcs_id.selectExpr("ARC_ID_HASH as ARC_ID_HASH_1", "count as events_count")


# COMMAND ----------

Events = events_df.join(counted_by_arcs_id, counted_by_arcs_id.ARC_ID_HASH_1 == events_df.ARC_ID_HASH ,how="inner")

events_columns_to_keep = ['ARC_ID_HASH', 'ARC_LENGTH_DS', 'ROAD_PRIORITY_DS', 'ROAD_ROUNDABOUT_DS', 'ROAD_LANES_DS', 'LANES_FORWARD_DS', 'TUNNEL_DS', 'BRIDGE_DS', 'SPEED_LIMIT_DS', 'events_count']

Events = Events.select(events_columns_to_keep)
#display(Events)

# COMMAND ----------

#print(Events.count())
Events = Events.dropDuplicates()
#print(Events.count())

# COMMAND ----------

crashes_counted_by_arcs_id = crashes_df.groupBy("ARC_ID_HASH").count()
crashes_counted_by_arcs_id = crashes_counted_by_arcs_id.selectExpr("ARC_ID_HASH as ARC_ID_HASH_1","count as crashes_count")
#crashes_counted_by_arcs_id = crashes_counted_by_arcs_id.withColumnRenamed("count","crashes_count")
#display(crashes_counted_by_arcs_id)
#print(crashes_counted_by_arcs_id.count())
Events_Crashes = Events.join(crashes_counted_by_arcs_id, Events.ARC_ID_HASH == crashes_counted_by_arcs_id.ARC_ID_HASH_1, how="outer")
#print(Events_Crashes.count())

# COMMAND ----------

aadt_df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "AADT") \
  .load()  

aadt_columns_to_keep = ['ARC_ID_HASH', 'AADT_2022']
aadt_df_select = aadt_df.select(aadt_columns_to_keep)
aadt_df_select = aadt_df_select.withColumnRenamed("ARC_ID_HASH","ARC_ID_HASH_2")


# COMMAND ----------

aadt_events_crashes = Events_Crashes.join(aadt_df_select, Events_Crashes.ARC_ID_HASH == aadt_df_select.ARC_ID_HASH_2, how="left")

# COMMAND ----------

cols_to_drop = ['ARC_ID_HASH_1', 'ARC_ID_HASH_2']
aadt_events_crashes = aadt_events_crashes.drop(*cols_to_drop)


# COMMAND ----------

#Pre-process the data
aadt_events_crashes = aadt_events_crashes.na.fill({'crashes_count': 0})
aadt_events_crashes = aadt_events_crashes.na.fill({'events_count': 0})

#Add column crashes_aadt
aadt_events_crashes = aadt_events_crashes.withColumn("crashes_aadt", lit(0))
#convert boolean columns to null
aadt_events_crashes = aadt_events_crashes.withColumn("ROAD_ROUNDABOUT_DS", col("ROAD_ROUNDABOUT_DS").cast("int"))
aadt_events_crashes = aadt_events_crashes.withColumn("TUNNEL_DS", col("TUNNEL_DS").cast("int"))
aadt_events_crashes = aadt_events_crashes.withColumn("BRIDGE_DS", col("BRIDGE_DS").cast("int"))

median = aadt_events_crashes.approxQuantile("ROAD_LANES_DS", [0.5], 0.01)[0]
aadt_events_crashes = aadt_events_crashes.withColumn("ROAD_LANES_DS", when(col("ROAD_LANES_DS").isNotNull(), col("ROAD_LANES_DS")).otherwise(median))

median = aadt_events_crashes.approxQuantile("LANES_FORWARD_DS", [0.5], 0.01)[0]
aadt_events_crashes = aadt_events_crashes.withColumn("LANES_FORWARD_DS", when(col("LANES_FORWARD_DS").isNotNull(), col("LANES_FORWARD_DS")).otherwise(median))

median = aadt_events_crashes.approxQuantile("SPEED_LIMIT_DS", [0.5], 0.01)[0]
aadt_events_crashes = aadt_events_crashes.withColumn("SPEED_LIMIT_DS", when(col("SPEED_LIMIT_DS").isNotNull(), col("SPEED_LIMIT_DS")).otherwise(median))

aadt_events_crashes = aadt_events_crashes.withColumn("AADT_2022", aadt_events_crashes["AADT_2022"].cast("float"))
aadt_events_crashes = aadt_events_crashes.withColumn("ROAD_PRIORITY_DS", col("ROAD_PRIORITY_DS").cast("int"))

# COMMAND ----------

poi_df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "POI") \
  .load()  

poi_df = poi_df.select(['ARC_ID_HASH', 'AMENITY_SCHOOL','AMENITY_UNIVERSITY','AMENITY_HOSPITAL','HIGHWAY_CROSSING','LEISURE_PARK'])
poi_df = poi_df.withColumnRenamed("ARC_ID_HASH","ARC_ID_HASH_2")
#if arc is close to school, uni, hospital, highway crossing or park
distance = 300
poi_df = poi_df.withColumn("CloseToPOI", when( ((poi_df["AMENITY_SCHOOL"]< distance) &  (poi_df["AMENITY_SCHOOL"]> -1) )|
                                               ((poi_df["AMENITY_UNIVERSITY"]< distance)  &  (poi_df["AMENITY_UNIVERSITY"]> -1) ) |
                                               ((poi_df["AMENITY_HOSPITAL"]< distance ) &  (poi_df["AMENITY_HOSPITAL"]> -1))|
                                               ((poi_df["HIGHWAY_CROSSING"]< distance ) &  (poi_df["HIGHWAY_CROSSING"]> -1)) |
                                               ((poi_df["LEISURE_PARK"]< distance ) &  (poi_df["LEISURE_PARK"]> -1)), 1)
                                  .otherwise(0))
display(poi_df)                                  

# COMMAND ----------

aadt_events_crashes = aadt_events_crashes.join(poi_df, aadt_events_crashes.ARC_ID_HASH == poi_df.ARC_ID_HASH_2, how="left")

# COMMAND ----------

aadt_events_crashes = aadt_events_crashes.withColumn("CloseToPOI", when(col("CloseToPOI").isNotNull(), col("CloseToPOI")).otherwise(0))
display(aadt_events_crashes)

# COMMAND ----------

aadt_events_crashes.stat.corr('crashes_count', 'CloseToPOI')

# COMMAND ----------

aadt_events_crashes.stat.corr('events_count', 'CloseToPOI')

# COMMAND ----------


#get rows with aadt_2022 not equals null to train the model
aadt_events_crashes_train = aadt_events_crashes.filter(aadt_events_crashes["AADT_2022"].isNotNull())
aadt_events_crashes_train = aadt_events_crashes_train.withColumn("crashes_aadt", col("crashes_count")/col("AADT_2022"))
#display(aadt_events_crashes_train)

# COMMAND ----------

aadt_events_crashes_train = aadt_events_crashes_train.filter(~isnan(col("AADT_2022"))) #filter out nan b/c corr came as nan

aadt_events_crashes_train.stat.corr('crashes_count', 'AADT_2022')


# COMMAND ----------

#test data
aadt_events_crashes_test = aadt_events_crashes.subtract(aadt_events_crashes_train)

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

df_train = aadt_events_crashes_train
df_test = aadt_events_crashes_test

feature_cols = ['ARC_LENGTH_DS', 'ROAD_PRIORITY_DS','ROAD_ROUNDABOUT_DS',
                'ROAD_LANES_DS','TUNNEL_DS','BRIDGE_DS',
                'SPEED_LIMIT_DS','events_count','crashes_count' ]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
df_train = assembler.transform(df_train)
df_test = assembler.transform(df_test)

# Create a RandomForestRegressor model
rf = RandomForestRegressor(featuresCol="features", labelCol="AADT_2022", numTrees=10)

# Train the model
rf_model = rf.fit(df_train)

# Make predictions on the test data
test_predictions = rf_model.transform(df_test)


# COMMAND ----------

train_predictions = rf_model.transform(df_train)

# COMMAND ----------

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="crashes_aadt", predictionCol="prediction", metricName="rmse")
train_rmse = evaluator.evaluate(train_predictions)
#rmse = evaluator.evaluate(test_predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {train_rmse}")

# Print feature importances
print("Feature Importances:")
print(rf_model.featureImportances)

# Show the predictions
#test_predictions.select("crashes_aadt", "prediction").show()

# COMMAND ----------

#print(rf_model.featureImportances)

# COMMAND ----------

test_df = test_predictions.select( ['ARC_ID_HASH','ARC_LENGTH_DS', 'ROAD_PRIORITY_DS',
                                    'ROAD_ROUNDABOUT_DS','ROAD_LANES_DS','LANES_FORWARD_DS', 
                                    'TUNNEL_DS','BRIDGE_DS','SPEED_LIMIT_DS',
                                    'events_count','crashes_count',
                                    'prediction'])
display(test_df.show(5))



# COMMAND ----------

test_df = test_df.withColumnRenamed("prediction","AADT_2022")

# COMMAND ----------

train_df= df_train.select( ['ARC_ID_HASH','ARC_LENGTH_DS', 'ROAD_PRIORITY_DS',
                                    'ROAD_ROUNDABOUT_DS','ROAD_LANES_DS','LANES_FORWARD_DS', 
                                    'TUNNEL_DS','BRIDGE_DS','SPEED_LIMIT_DS',
                                    'events_count','crashes_count', 'AADT_2022'])

# COMMAND ----------

combined_df = train_df.union(test_df)

# COMMAND ----------

train_df.write \
  .format("snowflake") \
  .options(**options)\
  .option("dbtable", "G7_AADT_PREDICTION") \
  .mode("overwrite") \
  .save()

# COMMAND ----------

test_df.write \
  .format("snowflake") \
  .options(**options)\
  .option("dbtable", "G7_AADT_PREDICTION") \
  .mode("append") \
  .save()
