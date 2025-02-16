# Databricks notebook source
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import sum, col, desc, mean, isnan
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, mean, when, median, isnan, isnull#, isinf

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler


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

events_columns_to_keep = ['ARC_ID_HASH', 'ARC_LENGTH_DS', 'ROAD_PRIORITY_DS', 'ROAD_ROUNDABOUT_DS', 'ROAD_LANES_DS', 'LANES_FORWARD_DS', 'TUNNEL_DS', 'BRIDGE_DS', 'SPEED_LIMIT_DS', 'CURVE_MAX_ANGLE_DS','ARC_SLOPE_DS', 'events_count']

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
#display(poi_df)                                  

# COMMAND ----------

aadt_events_crashes = aadt_events_crashes.join(poi_df, aadt_events_crashes.ARC_ID_HASH == poi_df.ARC_ID_HASH_2, how="left")

# COMMAND ----------

aadt_events_crashes = aadt_events_crashes.withColumn("CloseToPOI", when(col("CloseToPOI").isNotNull(), col("CloseToPOI")).otherwise(0))
aadt_events_crashes = aadt_events_crashes.drop(*["ARC_ID_HASH_1","ARC_ID_HASH_2"])
#display(aadt_events_crashes)

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

aadt_events_crashes_train_pd = aadt_events_crashes_train.toPandas()
aadt_events_crashes_train_pd = aadt_events_crashes_train_pd[['ARC_LENGTH_DS', 'ROAD_PRIORITY_DS', 'ROAD_ROUNDABOUT_DS', 'ROAD_LANES_DS', 'LANES_FORWARD_DS', 'BRIDGE_DS', 'SPEED_LIMIT_DS', 'CURVE_MAX_ANGLE_DS', 'events_count','crashes_count', 'AADT_2022', 'CloseToPOI']] #'ARC_SLOPE_DS' arc_slope_DS has null corr with BRIDGE_DS
correlation_matrix_pd = aadt_events_crashes_train_pd.corr()

# Plot the correlation matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_pd, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()


# COMMAND ----------

#test data
aadt_events_crashes_test = aadt_events_crashes.subtract(aadt_events_crashes_train)

# COMMAND ----------

###evaluate model
# ev_df = aadt_events_crashes_train
# feature_cols = ['ARC_LENGTH_DS', 'ROAD_PRIORITY_DS','ROAD_ROUNDABOUT_DS',
#                 'ROAD_LANES_DS','TUNNEL_DS','BRIDGE_DS',
#                 'SPEED_LIMIT_DS', 'CURVE_MAX_ANGLE_DS','events_count','crashes_count', 'CloseToPOI' ]

# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
# ev_df_trans = assembler.transform(ev_df)
# train_data1, test_data1 = ev_df_trans.randomSplit([0.8, 0.2], seed=167)
# rf = RandomForestRegressor(featuresCol="features",labelCol="crashes_aadt" , numTrees=10)
# rf_ev_model = rf.fit(train_data1)

# # Make predictions on the test data
# test_ev__predictions = rf_ev_model.transform(test_data1)
# evaluator = RegressionEvaluator(labelCol="crashes_aadt", predictionCol="prediction", metricName="rmse")
# test_ev_rmse = evaluator.evaluate(test_ev__predictions)
# #rmse = evaluator.evaluate(test_predictions)
# print(f"Root Mean Squared Error (RMSE) on test data: {test_ev_rmse}")
# ev_r2 = evaluator.evaluate(test_ev__predictions, {evaluator.metricName: "r2"})
# print(f"R-squared (R2): {ev_r2}")


# COMMAND ----------


df_train = aadt_events_crashes_train
df_test = aadt_events_crashes_test

feature_cols = ['ARC_LENGTH_DS', 'ROAD_PRIORITY_DS','ROAD_ROUNDABOUT_DS',
                'ROAD_LANES_DS','TUNNEL_DS','BRIDGE_DS',
                'SPEED_LIMIT_DS', 'CURVE_MAX_ANGLE_DS','events_count','crashes_count', 'CloseToPOI' ]

# TRI_RAW_START
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
df_train = assembler.transform(df_train)
df_test = assembler.transform(df_test)

# Create a RandomForestRegressor model
rf = RandomForestRegressor(featuresCol="features",labelCol="crashes_aadt" , numTrees=10) #labelCol="crashes_aadt"

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
#eature_cols = ['ARC_LENGTH_DS', 'ROAD_PRIORITY_DS','ROAD_ROUNDABOUT_DS',
#                'ROAD_LANES_DS','TUNNEL_DS','BRIDGE_DS',
#                'SPEED_LIMIT_DS', 'CURVE_MAX_ANGLE_DS','events_count','crashes_count', 'CloseToPOI' ]

# COMMAND ----------

feature_idx = [0,1,3,5,6,7]
importance = [0.38568766408397154,0.15895701034790566,0.24825203932881731,0.07900699140089858,0.06333306660254513,0.0647632282358616]
for idx, (f, i) in enumerate(zip(feature_idx, importance)):
    print(f'{feature_cols[f]}: {i}')

# COMMAND ----------

feature_idx = [0,1,3,5,6,7,8,9]
importance = [0.060546452605763336,0.07924518047781827,0.14487252586875998,0.020786086255563056,0.07386951739139057,0.044881946041524484,0.5676833014580108,0.008114989901169305]
for idx, (f, i) in enumerate(zip(feature_idx, importance)):
    print(f'{feature_cols[f]}: {i}')

# COMMAND ----------

# Evaluate the Model
evaluator = RegressionEvaluator(labelCol="crashes_aadt", predictionCol="prediction")

# RMSE (Root Mean Squared Error)
rmse = evaluator.evaluate(train_predictions, {evaluator.metricName: "rmse"})
print(f"Root Mean Squared Error (RMSE): {rmse}")

# MSE (Mean Squared Error)
mse = evaluator.evaluate(train_predictions, {evaluator.metricName: "mse"})
print(f"Mean Squared Error (MSE): {mse}")

# MAE (Mean Absolute Error)
mae = evaluator.evaluate(train_predictions, {evaluator.metricName: "mae"})
print(f"Mean Absolute Error (MAE): {mae}")

# R2 (R-squared)
r2 = evaluator.evaluate(train_predictions, {evaluator.metricName: "r2"})
print(f"R-squared (R2): {r2}")

# COMMAND ----------

test_df = test_predictions.select( ['ARC_ID_HASH','ARC_LENGTH_DS', 'ROAD_PRIORITY_DS',
                                    'ROAD_ROUNDABOUT_DS','ROAD_LANES_DS','LANES_FORWARD_DS', 
                                    'TUNNEL_DS','BRIDGE_DS','SPEED_LIMIT_DS','CURVE_MAX_ANGLE_DS',
                                    'events_count','crashes_count','AADT_2022','CloseToPOI','prediction'   #'AADT_2022'
                                    ])
#display(test_df.show(5))

'''tst = test_predictions.select( ['ARC_ID_HASH', 'prediction'])
df_test = tst.join(df_test, tst.ARC_ID_HASH == df_test.ARC_ID_HASH)
display(df_test.toPandas())


tst = test_predictions.select(["ARC_ID_HASH", "prediction"]).limit(100)
display(tst)
'''
'''
# Join using aliases
df_test = df_test.join(tst, col("tst.ARC_ID_HASH") == col("df_test.ARC_ID_HASH"))

# Display the resulting DataFrame
display(df_test.toPandas())
'''

# COMMAND ----------

test_df = test_df.withColumnRenamed("prediction","crashes_aadt")
#test_df = test_df.withColumn("crashes_aadt", col("crashes_count")/col("AADT_2022"))

# COMMAND ----------

train_df= df_train.select( ['ARC_ID_HASH','ARC_LENGTH_DS', 'ROAD_PRIORITY_DS',
                                    'ROAD_ROUNDABOUT_DS','ROAD_LANES_DS','LANES_FORWARD_DS', 
                                    'TUNNEL_DS','BRIDGE_DS','SPEED_LIMIT_DS','CURVE_MAX_ANGLE_DS',
                                    'events_count','crashes_count', 'AADT_2022','CloseToPOI','crashes_aadt'])

# COMMAND ----------

#combined_df = train_df.union(test_df)

# COMMAND ----------

train_df.write \
  .format("snowflake") \
  .options(**options)\
  .option("dbtable", "G7_Model2_v2") \
  .mode("overwrite") \
  .save()

# COMMAND ----------

test_df.write \
  .format("snowflake") \
  .options(**options)\
  .option("dbtable", "G7_Model2_v2") \
  .mode("append") \
  .save()

# COMMAND ----------

print(crashes_counted_by_arcs_id.count())

# COMMAND ----------


feature_importances = rf_model.featureImportances
print(feature_importances)


# COMMAND ----------


indces = [0,1,2,3,4,5,6,7,8,9,10]
feature_cols
names = [feature_cols[i] for i in indces]
print(names)
importance = feature_importances.toArray()
print(importance)
s = np.array(importance)
sorted_importance = np.sort(s)
sort_index = np.argsort(s)
print(sort_index)
names = [feature_cols[i] for i in sort_index]
print(names)
plt.figure(figsize=(10, 6))
plt.barh(names, sorted_importance, align="center")
#plt.xticks(range(l), feature_names[indices], rotation=45)
plt.ylabel("Feature")
plt.xlabel("Importance")
