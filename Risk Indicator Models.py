# Databricks notebook source
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, lit, first, isnan, mean
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Summarizer, Correlation
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator





# COMMAND ----------

#Credentials#

options = {
"sfUrl": "ab43836.west-europe.azure.snowflakecomputing.com",
"sfUser": "r****", # Email address of your personal
"sfPassword": "", # Password you choose when first login to Snowflake
"sfDatabase": "GATECH",
"sfSchema": "GROUP_7", # Replace * by your group number
"sfWarehouse": "GATECH_WH"
}

# COMMAND ----------

#LOAD DATAFRAMES AND CONFIGURE#

############EVENTS###############
EVENT = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "EVENTS") \
.load()

EVENT = EVENT.withColumnRenamed("arc_index_hash_start", "ARC_ID_HASH")

# Pivot the DataFrame to create new columns for each event_type_desc
EVENTS = EVENT.groupBy("ARC_ID_HASH").pivot("event_type_desc").count().fillna(0)

#Summing events
columns_to_sum = [col(column) for column in EVENTS.columns if column != "ARC_ID_HASH"]

# Add a new column "Total Events" with the sum of all other columns
EVENT_TYPES = EVENTS.withColumn("Total Events", sum(columns_to_sum))


# Divide specified columns by "Total Events"
columns_to_divide = ["Acceleration Event", "Braking Event", "Excessive Speeding", "Phone Handling Event", "Potential Collision Event", "Speeding Event"]

for column in columns_to_divide:
    new_column_name = f"{column} Ratio"
    EVENT_TYPES = EVENT_TYPES.withColumn(new_column_name, col(column) / col("Total Events"))


###########TRIPS############
TRIP = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "TRIPS") \
.load()

TRIP = TRIP.withColumnRenamed("ARC_INDEX_HASH", "ARC_ID_HASH")





###############CRASHES##############
CRASH = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "CRASHES_2019_2023_MAPMATCHED") \
.load()

# Create a 'count' field that counts the number of crashes per road segment
CRASHES = CRASH.groupBy("ARC_ID_HASH").agg(count("*").alias("count"))

# Create a 'crash' field indicating whether a crash occurred (1) or not (0)
CRASHES = CRASHES.withColumn("crash", when(CRASHES["count"] > 0, 1).otherwise(0))

# Get the ARC_ID_HASH values from EVENTS that are not in CRASHES
new_arc_id_hashes = EVENT.select("ARC_ID_HASH").subtract(CRASHES.select("ARC_ID_HASH"))

# Add new entries with 0s in count and crash columns to CRASHES
new_entries_df = new_arc_id_hashes.withColumn("count", lit(0)).withColumn("crash", lit(0))

# Union the new entries with CRASHES
CRASHES = CRASHES.union(new_entries_df)

# Select only the necessary columns from EVENTS DataFrame
events_subset = EVENT_TYPES.select("ARC_ID_HASH", "Total Events")

# Join the CRASHES and the subset of EVENTS on the common identifier ARC_ID_HASH
joined_df = CRASHES.join(events_subset, CRASHES.ARC_ID_HASH == events_subset.ARC_ID_HASH, "inner")

# Perform the division to calculate "Crashes Per Event" and add it as a new column
result_df = joined_df.withColumn("Crashes Per Event", col("CRASH") / col("Total Events"))

# Drop the additional ARC_ID_HASH column from the EVENTS subset
CRASHES = result_df.drop(events_subset.ARC_ID_HASH, "Total Events")





# COMMAND ----------

EVENT_TYPES.printSchema()

# COMMAND ----------

#####SELECT THE FEATURES WE CARE ABOUT########

# TRIPS FEATURES
trips_filtered = TRIP[["ARC_ID_HASH", "ROAD_PRIORITY_DS", "TRI_RAW", "SPEED_LIMIT_DS"]]

trips_filtered = trips_filtered.dropDuplicates(["ARC_ID_HASH"])

trips_filtered = trips_filtered.na.drop()

# Convert ROAD_PRIORITY_DS to long
trips_filtered = trips_filtered.withColumn("ROAD_PRIORITY_DS", col("ROAD_PRIORITY_DS").cast("long"))

# Convert TRI_RAW to long
trips_filtered = trips_filtered.withColumn("TRI_RAW", col("TRI_RAW").cast("long"))

# Convert SPEED_LIMIT_DS to long
trips_filtered = trips_filtered.withColumn("SPEED_LIMIT_DS", col("SPEED_LIMIT_DS").cast("long"))



# EVENTS FEATURES NOT INCLUDING THE EVENT TYPES


events_filtered = EVENT[["ARC_ID_HASH", "TRI_RAW_START", "ROAD_PRIORITY_DS", "TUNNEL_DS", "BRIDGE_DS", "INTERACTION_POI_DS", "ARC_SLOPE_DS", "CIRCULATION_POI_DS","CURVE_MAX_ANGLE_DS", "ARC_LENGTH_DS", "ROAD_LANES_DS", "SPEED_LIMIT_DS"]]

events_filtered = events_filtered.dropDuplicates(["ARC_ID_HASH"])

#Fill in null values for CATEGORICAL values
events_filtered = events_filtered.withColumn("INTERACTION_POI_DS", when(col("INTERACTION_POI_DS").isNull(), "None").otherwise(col("INTERACTION_POI_DS")))
events_filtered = events_filtered.withColumn("CIRCULATION_POI_DS", when(col("CIRCULATION_POI_DS").isNull(), "None").otherwise(col("CIRCULATION_POI_DS")))



median = events_filtered.approxQuantile("ROAD_LANES_DS", [0.5], 0.01)[0]
events_filtered = events_filtered.withColumn("ROAD_LANES_DS", when(col("ROAD_LANES_DS").isNotNull(), col("ROAD_LANES_DS")).otherwise(median))

median = events_filtered.approxQuantile("SPEED_LIMIT_DS", [0.5], 0.01)[0]
events_filtered = events_filtered.withColumn("SPEED_LIMIT_DS", when(col("SPEED_LIMIT_DS").isNotNull(), col("SPEED_LIMIT_DS")).otherwise(median))

# Convert ROAD_LANES_DS to long
events_filtered = events_filtered.withColumn("ROAD_LANES_DS", col("ROAD_LANES_DS").cast("long"))

# Convert SPEED_LIMIT_DS to long
events_filtered = events_filtered.withColumn("SPEED_LIMIT_DS", col("SPEED_LIMIT_DS").cast("long"))


# Convert ROAD_PRIORITY_DS to long
events_filtered = events_filtered.withColumn("ROAD_PRIORITY_DS", col("ROAD_PRIORITY_DS").cast("long"))

# Convert TRI_RAW to long
events_filtered = events_filtered.withColumn("TRI_RAW_START", col("TRI_RAW_START").cast("long"))

# Convert  "TUNNEL_DS"from boolean
events_filtered = events_filtered.withColumn("TUNNEL_DS", col("TUNNEL_DS").cast("int"))

# Convert  "BRIDGE_DS"from boolean
events_filtered = events_filtered.withColumn("BRIDGE_DS", col("BRIDGE_DS").cast("int"))

# Convert ARC_SLOPE_DS to long
events_filtered = events_filtered.withColumn("ARC_SLOPE_DS", col("ARC_SLOPE_DS").cast("long"))

# Convert CURVE_MAX_ANGLE_DS to long
events_filtered = events_filtered.withColumn("CURVE_MAX_ANGLE_DS", col("CURVE_MAX_ANGLE_DS").cast("long"))

# Convert ARC_LENGTH_DS to long
events_filtered = events_filtered.withColumn("ARC_LENGTH_DS", col("ARC_LENGTH_DS").cast("long"))





#Fill categorical null's wiht 'None'

# List of columns to replace nulls in
columns_to_replace = ["INTERACTION_POI_DS", "CIRCULATION_POI_DS"]

# Replace nulls with "None" in specified columns
for column in columns_to_replace:
    events_filtered = events_filtered.fillna("None", subset=[column])





# COMMAND ----------

events_filtered.select([col(c).isNull().alias(c) for c in events_filtered.columns]).show()

# COMMAND ----------

##########SETUP CATEGORICAL VARIABLES AS INDEXED##########



# Define categorical columns
categorical_columns = ["INTERACTION_POI_DS", "CIRCULATION_POI_DS", "ROAD_PRIORITY_DS"]


# Create StringIndexer for each categorical column
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_columns]

# Create OneHotEncoder for each indexed column
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded") for col in categorical_columns]

# Fit the pipeline to your data
pipeline_model = Pipeline(stages=indexers + encoders).fit(events_filtered)

# Transform the original DataFrame and add the transformed columns
transformed_data = pipeline_model.transform(events_filtered)



# COMMAND ----------

DATA.printSchema()

# COMMAND ----------

transformed_data.select([col(c).isNull().alias(c) for c in transformed_data.columns]).show()

# COMMAND ----------

########CREATE JOINED FEATURES DATAFRAMES########

#Joining EVENTS to CRASHES (Trips has redundant fields in this test)
#joined_df = events_filtered.join(EVENT_TYPES, 'ARC_ID_HASH', 'inner').join(CRASHES, 'ARC_ID_HASH', 'inner')

#Select columns of interest
#joined_df = joined_df[["ARC_ID_HASH", "SPEED_LIMIT_DS", "TRI_RAW_START", "ROAD_PRIORITY_DS", "TUNNEL_DS", "BRIDGE_DS", "interaction_poi_encoded","INTERACTION_POI_DS", "ARC_SLOPE_DS", "circulation_poi_encoded", "CIRCULATION_POI_DS", "Acceleration Event", "Braking Event", "Excessive Speeding", "Phone Handling Event", "Potential Collision Event", "Speeding Event", "count", "crash", "Crashes Per Event"]]

# COMMAND ----------

########CREATE JOINED FEATURES DATAFRAMES########

# Perform left outer joins
joined_df = transformed_data.join(EVENT_TYPES, 'ARC_ID_HASH', 'left_outer') \
    .join(CRASHES, 'ARC_ID_HASH', 'left_outer')


# Handle null values in the resulting DataFrame
#joined_df = joined_df.fillna(0)  # Replace null values with 0


# COMMAND ----------

joined_df.select([col(c).isNull().alias(c) for c in joined_df.columns]).show()

# COMMAND ----------

##########GET A SAMPLE WITH A BETTER PROPORTION OF ARCS WITH CRASHES (1:4) #######


filtered_df_1 = joined_df.filter(col("crash") == 1)

# Count the number of rows with 1 in the "Crash" column
count_1 = joined_df.filter(col("crash") == 1).count()

# Duplicate rows with 0 in the "Crash" column
filtered_df_0 = joined_df.filter(col("crash") == 0).limit(count_1 * 4)

# Combine the DataFrames
DATA = filtered_df_1.union(filtered_df_0)

# COMMAND ----------

for col_name, col_type in DATA.dtypes:
    if col_type in ['int', 'bigint', 'smallint', 'float']:
        DATA = DATA.withColumn(col_name, col(col_name).cast(DoubleType()))

# COMMAND ----------

# List of columns to check for null values
columns_to_check = ["TRI_RAW_START", "ROAD_PRIORITY_DS", "TUNNEL_DS", "BRIDGE_DS", "ARC_SLOPE_DS", "SPEED_LIMIT_DS",  "Excessive Speeding", "Phone Handling Event"]

# Check for null values in each column
null_counts = {col_name: DATA.filter(col(col_name).isNull()).count() for col_name in columns_to_check}

# Display the null counts
for col_name, count in null_counts.items():
   print(f"Null values in column '{col_name}': {count}")

# COMMAND ----------



columns_to_check = ["ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS"]

# Display summary statistics
summary_stats = DATA.select(columns_to_check).summary()
summary_stats.show()




# COMMAND ----------

columns = ["ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS" ]

# Calculate column-wise averages
column_averages = DATA.agg(*(mean(col(c)).alias(c) for c in columns))

# Fill null values with the corresponding column average
DATA = DATA.na.fill(column_averages.first().asDict())

# COMMAND ----------

DATA.select([col(c).isNull().alias(c) for c in DATA.columns]).show()

# COMMAND ----------

DATA.printSchema()

# COMMAND ----------

DATA.groupBy("crash").count().show()

# COMMAND ----------



# COMMAND ----------

###########CORRELATION########



# Select relevant columns
selected_columns = ["Crashes Per Event", "Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio"]

# Select the relevant columns from the DataFrame
selected_df = DATA.select(selected_columns)

# Drop rows with missing values (if necessary)
selected_df = selected_df.na.drop()

# Assemble features into a vector
assembler = VectorAssembler(inputCols=selected_columns[1:], outputCol="features")
assembled_df = assembler.transform(selected_df)

# Compute the correlation matrix
correlation_matrix = Correlation.corr(assembled_df, "features").head()

# Display the correlation matrix
print("Correlation Matrix:")
corr_matrix = correlation_matrix[0].toArray()

for i in range(len(selected_columns) - 1):
    for j in range(i + 1, len(selected_columns)):
        print(f"{selected_columns[i]} - {selected_columns[j]}: {corr_matrix[i, j]}")



# COMMAND ----------


# Compute the correlation matrix for predictor features
predictor_columns = selected_columns[1:]
predictor_correlation_matrix = Correlation.corr(assembled_df, "features").head()

# Extract the correlation matrix for predictor features
predictor_corr_matrix = predictor_correlation_matrix[0].toArray()

# Display the correlation matrix for predictor features
print("\nCorrelation Matrix for Predictor Features:")
for i in range(len(predictor_columns)):
    for j in range(len(predictor_columns)):
        if i < j:  # Print only the upper triangular part to avoid duplication
            print(f"{predictor_columns[i]} - {predictor_columns[j]}: {predictor_corr_matrix[i][j]}")


# COMMAND ----------

# Select relevant columns
selected_columns = ["Crashes Per Event", "SPEED_LIMIT_DS", "TRI_RAW_START", "ARC_SLOPE_DS"]

# Select the relevant columns from the DataFrame
selected_df = data_df.select(selected_columns)

# Drop rows with missing values (if necessary)
selected_df = selected_df.na.drop()

# Assemble features into a vector
assembler = VectorAssembler(inputCols=selected_columns[1:], outputCol="features")
assembled_df = assembler.transform(selected_df)

# Compute the correlation matrix
correlation_matrix = Correlation.corr(assembled_df, "features").head()
# Display the correlation scores
correlation_scores = correlation_matrix[1:]  # Exclude the first element (correlation with itself)
print("Correlation Scores:")
for i in range(len(selected_columns) - 1):
    print(f"{selected_columns[i + 1]} - Crashes Per Event: {correlation_scores[i]}")


# COMMAND ----------



# COMMAND ----------

###########MODEL 1#####################


dependent_variable1 = "Crashes Per Event Count" # Crashes / AADT(Calculated from Reems model)

# Features to be used in the model
#feature_columns1 = ["TRI_RAW_START", "ROAD_PRIORITY_DS", "TUNNEL_DS", "BRIDGE_DS", "ARC_SLOPE_DS",  "Excessive Speeding", "Phone Handling Event"]
feature_columns1 = ["ROAD_PRIORITY_DS","TUNNEL_DS", "BRIDGE_DS",  "Excessive Speeding", "Phone Handling Event"]

#feature_columns1 = ["ROAD_PRIORITY_DS","TUNNEL_DS", "BRIDGE_DS",  "Excessive Speeding/Total Events", "Phone Handling Event/Total Events"]

# Assemble features into a vector column
assembler1 = VectorAssembler(inputCols=feature_columns1, outputCol="features")
assembled_df1 = assembler1.transform(data_df)

# Split the data into training and test sets (80% train, 20% test)
train_data1, test_data1 = assembled_df1.randomSplit([0.8, 0.2], seed=42)

# Decision Tree Regressor
dt1 = DecisionTreeRegressor(featuresCol="features", labelCol=dependent_variable1)


# COMMAND ----------

# Fit the model
model_dt1 = dt1.fit(train_data1)

# COMMAND ----------

model_dt1.featureImportances

# COMMAND ----------



# COMMAND ----------

# Make predictions
predictions_dt1 = model_dt1.transform(test_data1)


# COMMAND ----------

# Display predictions
predictions_dt1.select("ARC_ID_HASH", dependent_variable1, "prediction").show(20)

# COMMAND ----------


# Filter for no crashes
filtered_predictions_dt1 = predictions_dt1.filter(predictions_dt1[dependent_variable1] == 0)

# Display filtered predictions
filtered_predictions_dt1.select("ARC_ID_HASH", dependent_variable1, "prediction").show(20)

# COMMAND ----------

# Evaluate the model using regression metrics
evaluator1 = RegressionEvaluator(labelCol=dependent_variable1, predictionCol="prediction", metricName="mse")
mse1 = evaluator1.evaluate(predictions_dt1)

evaluator1 = RegressionEvaluator(labelCol=dependent_variable1, predictionCol="prediction", metricName="rmse")
rmse1 = evaluator1.evaluate(predictions_dt1)

evaluator1 = RegressionEvaluator(labelCol=dependent_variable1, predictionCol="prediction", metricName="mae")
mae1 = evaluator1.evaluate(predictions_dt1)

evaluator1 = RegressionEvaluator(labelCol=dependent_variable1, predictionCol="prediction", metricName="r2")
r21 = evaluator1.evaluate(predictions_dt1)

# Display the regression metrics
print(f"Mean Squared Error (MSE): {mse1}")
print(f"Root Mean Squared Error (RMSE): {rmse1}")
print(f"Mean Absolute Error (MAE): {mae1}")
print(f"R-squared (R2): {r21}")

# COMMAND ----------

# Extract prediction and label columns as RDDs
predictionAndLabels1 = predictions_dt1.select('prediction', 'Crashes Per Event').rdd.map(tuple)

# Compute confusion matrix
metrics1 = MulticlassMetrics(predictionAndLabels1)
confusion_matrix1 = metrics1.confusionMatrix().toArray()
print("Confusion Matrix:")
print(confusion_matrix1)

# COMMAND ----------

# Create a pipeline with the DecisionTreeClassifier
pipeline1 = Pipeline(stages=[dt1])

# Define a parameter grid to search
param_grid1 = ParamGridBuilder() \
    .addGrid(dt1.maxDepth, [5, 10, 15]) \
    .addGrid(dt1.maxBins, [20, 40, 60]) \
    .build()

# COMMAND ----------

# Create a CrossValidator
cross_validator1 = CrossValidator(estimator=pipeline1,
                                 estimatorParamMaps=param_grid1,
                                 evaluator=MulticlassClassificationEvaluator(labelCol="Crashes Per Event", predictionCol="prediction"),
                                 numFolds=5,  
                                 seed=42)

# Fit the CrossValidator to the training data
cv_model1 = cross_validator1.fit(train_data1)

# Make predictions on the test data using the best model
best_model1 = cv_model1.bestModel
test_predictions1 = best_model1.transform(test_data1)

# COMMAND ----------

# Evaluate the model using regression metrics
evaluator12 = RegressionEvaluator(labelCol=dependent_variable1, predictionCol="prediction", metricName="mse")
mse12 = evaluator12.evaluate(test_predictions1)

evaluator12 = RegressionEvaluator(labelCol=dependent_variable1, predictionCol="prediction", metricName="rmse")
rmse12 = evaluator12.evaluate(test_predictions1)

evaluator12 = RegressionEvaluator(labelCol=dependent_variable1, predictionCol="prediction", metricName="mae")
mae12 = evaluator12.evaluate(test_predictions1)

evaluator12 = RegressionEvaluator(labelCol=dependent_variable1, predictionCol="prediction", metricName="r2")
r212 = evaluator12.evaluate(test_predictions1)

# Display the regression metrics
print(f"Mean Squared Error (MSE): {mse12}")
print(f"Root Mean Squared Error (RMSE): {rmse12}")
print(f"Mean Absolute Error (MAE): {mae12}")
print(f"R-squared (R2): {r212}")

# COMMAND ----------

###########MODEL 2#####################




# COMMAND ----------

# Split the data into training and test sets (80% train, 20% test)
train_data2, test_data2 = DATA.randomSplit([0.8, 0.2], seed=42)

feature_columns2 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio", "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS", "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded"]



# TRI_RAW_START
assembler2 = VectorAssembler(inputCols=feature_columns2, outputCol="features", handleInvalid="keep")
df_train2 = assembler2.transform(train_data2)
df_test2 = assembler2.transform(test_data2)

# Create a RandomForestRegressor model
rf2 = RandomForestRegressor(featuresCol="features", labelCol="Crashes Per Event", numTrees=5)




# COMMAND ----------

# Train the model
rf_model2 = rf2.fit(df_train2)




# COMMAND ----------

# Make predictions on the test data
test_predictions2 = rf_model2.transform(df_test2)


# COMMAND ----------

train_predictions2 = rf_model2.transform(df_train2)

# COMMAND ----------

# Evaluate the model
evaluator2 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
train_rmse2 = evaluator2.evaluate(train_predictions2)
#rmse = evaluator.evaluate(test_predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {train_rmse2}")

# Print feature importances
print("Feature Importances:")
print(rf_model2.featureImportances)

# COMMAND ----------

feature_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,25,26,27,28,29,30,31,35,36]
importance = [0.2082537241805309,0.05894655009553097,0.06404886344525117,0.02977307858765609,0.0025127411763237045,0.0010293270283692683,0.09929449357438974,0.040536193098394256,0.08862187775441692,0.004571138891041994,0.0075494020142129635,0.003770012228382786,0.014378737008451728,0.030263292937894253,0.0008209670745246931,0.004666017804812135,0.04493734978228575,0.0034749756554631034,0.15145388305416385,0.003193491209124645,0.10116043268868972,0.0012272695539260758,0.03492709444838521,0.0003713981082719958,0.00021768859950624047]
for idx, (f, i) in enumerate(zip(feature_idx, importance)):
    print(f'{feature_columns2[f]}: {i}')

# COMMAND ----------

# Evaluate the Model
evaluator2 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction")

# RMSE (Root Mean Squared Error)
rmse2 = evaluator2.evaluate(test_predictions2, {evaluator2.metricName: "rmse"})
print(f"Root Mean Squared Error (RMSE): {rmse2}")

# MSE (Mean Squared Error)
mse2 = evaluator2.evaluate(test_predictions2, {evaluator2.metricName: "mse"})
print(f"Mean Squared Error (MSE): {mse2}")

# MAE (Mean Absolute Error)
mae2 = evaluator2.evaluate(test_predictions2, {evaluator2.metricName: "mae"})
print(f"Mean Absolute Error (MAE): {mae2}")

# R2 (R-squared)
r22 = evaluator2.evaluate(test_predictions2, {evaluator2.metricName: "r2"})
print(f"R-squared (R2): {r22}")

# COMMAND ----------

###########MODEL 3#####################



# Split the data into training and test sets (80% train, 20% test)
train_data3, test_data3 = DATA.randomSplit([0.8, 0.2], seed=42)

#feature_columns2 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio", "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS", "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded"]

feature_columns3 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio", "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS", "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded", "SPEED_LIMIT_DS", "ROAD_LANES_DS"]

# TRI_RAW_START
assembler3 = VectorAssembler(inputCols=feature_columns3, outputCol="features", handleInvalid="keep")
df_train3 = assembler3.transform(train_data3)
df_test3 = assembler3.transform(test_data3)

# Create a RandomForestRegressor model
rf3 = RandomForestRegressor(featuresCol="features", labelCol="Crashes Per Event", numTrees=5)


# COMMAND ----------

# Train the model
rf_model3 = rf3.fit(df_train3)

# COMMAND ----------

# Make predictions on the test data
test_predictions3 = rf_model3.transform(df_test3)

# COMMAND ----------

train_predictions3 = rf_model3.transform(df_train3)

# COMMAND ----------

# Evaluate the model
evaluator3 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
train_rmse3 = evaluator3.evaluate(train_predictions3)
#rmse = evaluator.evaluate(test_predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {train_rmse3}")

# Print feature importances
print("Feature Importances:")
print(rf_model3.featureImportances)

# COMMAND ----------

feature_idx = [0,1,2,3,4,5,6,7,8,9,12,14,17,20,25,27,28,29,30,31,34,40,41]
importance = [0.33496190182871427,0.09354395127655277,0.07069575643113497,0.020973730709029595,0.0009016008469098068,0.0001533064421413271,0.011156910816643554,0.03901650794924849,0.0997240354493726,0.0008123628455989292,0.011635453072865874,0.026436646388455057,0.006449476384328401,0.0010947254177835584,0.01661994251189247,0.019708287578803812,0.003251314325646534,0.008458014269911848,0.0019420871854173235,0.0330018429186341,0.0022133734860248894,0.018258173971115725,0.17899059789377436]
for idx, (f, i) in enumerate(zip(feature_idx, importance)):
    print(f'{feature_columns3[f]}: {i}')

# COMMAND ----------

# Evaluate the Model
evaluator3 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction")

# RMSE (Root Mean Squared Error)
rmse3 = evaluator3.evaluate(test_predictions3, {evaluator3.metricName: "rmse"})
print(f"Root Mean Squared Error (RMSE): {rmse3}")

# MSE (Mean Squared Error)
mse3 = evaluator3.evaluate(test_predictions3, {evaluator3.metricName: "mse"})
print(f"Mean Squared Error (MSE): {mse3}")

# MAE (Mean Absolute Error)
mae3 = evaluator3.evaluate(test_predictions3, {evaluator3.metricName: "mae"})
print(f"Mean Absolute Error (MAE): {mae3}")

# R2 (R-squared)
r23 = evaluator3.evaluate(test_predictions3, {evaluator3.metricName: "r2"})
print(f"R-squared (R2): {r23}")

# COMMAND ----------

###########MODEL 4#####################

# Has a 1:1 crash to no crash split


filtered_df_1 = joined_df.filter(col("crash") == 1)

# Count the number of rows with 1 in the "Crash" column
count_1 = joined_df.filter(col("crash") == 1).count()

# Duplicate rows with 0 in the "Crash" column
filtered_df_0 = joined_df.filter(col("crash") == 0).limit(count_1)

# Combine the DataFrames
DATA2 = filtered_df_1.union(filtered_df_0)




# Split the data into training and test sets (80% train, 20% test)
train_data4, test_data4 = DATA2.randomSplit([0.8, 0.2], seed=42)


feature_columns4 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio", "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS", "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded", "SPEED_LIMIT_DS", "ROAD_LANES_DS"]

# TRI_RAW_START
assembler4 = VectorAssembler(inputCols=feature_columns4, outputCol="features", handleInvalid="keep")
df_train4 = assembler4.transform(train_data4)
df_test4 = assembler4.transform(test_data4)

# Create a RandomForestRegressor model
rf4 = RandomForestRegressor(featuresCol="features", labelCol="Crashes Per Event", numTrees=5)

# COMMAND ----------

# Train the model
rf_model4 = rf4.fit(df_train4)

# COMMAND ----------

# Make predictions on the test data
test_predictions4 = rf_model4.transform(df_test4)

# COMMAND ----------

train_predictions4 = rf_model4.transform(df_train4)

# COMMAND ----------

# Evaluate the model
evaluator4 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
train_rmse4 = evaluator4.evaluate(train_predictions4)
#rmse = evaluator.evaluate(test_predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {train_rmse4}")

# Print feature importances
print("Feature Importances:")
print(rf_model4.featureImportances)

# COMMAND ----------

feature_idx = [0,1,2,3,4,5,6,7,8,9,11,12,14,16,17,18,25,26,27,29,31,32,34,37,40,41]
importance = [0.016045654655388798,0.012773835200955442,0.0434888339708118,0.05162633774739479,5.551063048927512e-05,0.006953675885834616,0.013318973990072822,0.008351205948356859,0.17110524795210302,0.08651626363905073,0.028473210197738182,0.02732715720706307,0.12722128708388875,0.00015478202765469275,0.001079630138922935,0.00995708123113783,0.0376762617866025,0.002721051478862394,0.28553637777548896,0.003253566959188993,0.04149733712541282,0.006392919695240039,0.007833625477110775,0.001062378652405655,0.0038231123286847244,0.005754681214139289]
for idx, (f, i) in enumerate(zip(feature_idx, importance)):
    print(f'{feature_columns4[f]}: {i}')

# COMMAND ----------

# Evaluate the Model
evaluator4 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction")

# RMSE (Root Mean Squared Error)
rmse4 = evaluator4.evaluate(test_predictions4, {evaluator4.metricName: "rmse"})
print(f"Root Mean Squared Error (RMSE): {rmse4}")

# MSE (Mean Squared Error)
mse4 = evaluator4.evaluate(test_predictions4, {evaluator4.metricName: "mse"})
print(f"Mean Squared Error (MSE): {mse4}")

# MAE (Mean Absolute Error)
mae4 = evaluator4.evaluate(test_predictions4, {evaluator4.metricName: "mae"})
print(f"Mean Absolute Error (MAE): {mae4}")

# R2 (R-squared)
r24 = evaluator4.evaluate(test_predictions4, {evaluator4.metricName: "r2"})
print(f"R-squared (R2): {r24}")

# COMMAND ----------

###########MODEL 5#####################

# Has a 1:10 crash to no crash split


filtered_df_1 = joined_df.filter(col("crash") == 1)

# Count the number of rows with 1 in the "Crash" column
count_1 = joined_df.filter(col("crash") == 1).count()

# Duplicate rows with 0 in the "Crash" column
filtered_df_0 = joined_df.filter(col("crash") == 0).limit(count_1*10)

# Combine the DataFrames
DATA3 = filtered_df_1.union(filtered_df_0)




# Split the data into training and test sets (80% train, 20% test)
train_data5, test_data5 = DATA3.randomSplit([0.8, 0.2], seed=42)


feature_columns5 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio", "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS", "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded", "SPEED_LIMIT_DS", "ROAD_LANES_DS"]

# TRI_RAW_START
assembler5 = VectorAssembler(inputCols=feature_columns5, outputCol="features", handleInvalid="keep")
df_train5 = assembler5.transform(train_data5)
df_test5 = assembler5.transform(test_data5)

# Create a RandomForestRegressor model
rf5 = RandomForestRegressor(featuresCol="features", labelCol="Crashes Per Event", numTrees=5)

# Train the model
rf_model5 = rf5.fit(df_train5)

# Make predictions on the test data
test_predictions5 = rf_model5.transform(df_test5)

train_predictions5 = rf_model5.transform(df_train5)

# Evaluate the model
evaluator5 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
train_rmse5 = evaluator5.evaluate(train_predictions5)
#rmse = evaluator.evaluate(test_predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {train_rmse5}")

# Print feature importances
print("Feature Importances:")
print(rf_model5.featureImportances)

# COMMAND ----------

feature_idx = [0,1,2,3,4,5,6,7,8,9,12,13,14,17,20,25,26,27,28,29,31,32,40,41]
importance = [0.11002960820143574,0.025524856982688814,0.032471479492524546,0.02177174153555174,0.00014840068541175155,1.2068292123591236e-07,0.38933473700119114,0.01746743922469606,0.13534905888957338,0.0020619439208816063,0.006619858902867233,7.282597779542458e-05,0.11218063053415943,0.0007832209122028217,0.0011522694833685488,0.008507240390373613,0.004511835388854638,0.024297324611487942,0.0001416197738599397,9.192515443349214e-05,0.013893630881199834,0.0002597011101919974,0.0009290748714118989,0.09239945539091718]
for idx, (f, i) in enumerate(zip(feature_idx, importance)):
    print(f'{feature_columns5[f]}: {i}')

# COMMAND ----------

# Evaluate the Model
evaluator5 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction")

# RMSE (Root Mean Squared Error)
rmse5 = evaluator5.evaluate(test_predictions5, {evaluator5.metricName: "rmse"})
print(f"Root Mean Squared Error (RMSE): {rmse5}")

# MSE (Mean Squared Error)
mse5 = evaluator5.evaluate(test_predictions5, {evaluator5.metricName: "mse"})
print(f"Mean Squared Error (MSE): {mse5}")

# MAE (Mean Absolute Error)
mae5 = evaluator5.evaluate(test_predictions5, {evaluator5.metricName: "mae"})
print(f"Mean Absolute Error (MAE): {mae5}")

# R2 (R-squared)
r25 = evaluator5.evaluate(test_predictions5, {evaluator5.metricName: "r2"})
print(f"R-squared (R2): {r25}")

# COMMAND ----------

###########MODEL 6#####################

#Model 3 features slimmed






# Split the data into training and test sets (80% train, 20% test)
train_data6, test_data6 = DATA.randomSplit([0.8, 0.2], seed=42)


feature_columns6 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded", "SPEED_LIMIT_DS", "ROAD_LANES_DS"]

# TRI_RAW_START
assembler6 = VectorAssembler(inputCols=feature_columns6, outputCol="features", handleInvalid="keep")
df_train6 = assembler6.transform(train_data6)
df_test6 = assembler6.transform(test_data6)

# Create a RandomForestRegressor model
rf6 = RandomForestRegressor(featuresCol="features", labelCol="Crashes Per Event", numTrees=5)

# Train the model
rf_model6 = rf6.fit(df_train6)

# Make predictions on the test data
test_predictions6 = rf_model6.transform(df_test6)

train_predictions6 = rf_model6.transform(df_train6)

# Evaluate the model
evaluator6 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
train_rmse6 = evaluator6.evaluate(train_predictions6)
#rmse = evaluator.evaluate(test_predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {train_rmse6}")

# Print feature importances
print("Feature Importances:")
print(rf_model6.featureImportances)

# COMMAND ----------

feature_idx = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,22,23,24,25,26,27,28,30,33,37,38]
importance = [0.09264970038606635,0.07173311467556255,0.07727391158671239,0.09042135625685316,0.007304886763842493,0.01611755750441838,0.11250151100806326,0.019087036271284264,0.01613076023759267,0.0023827532435615516,0.028817396344287326,4.83003592213185e-05,0.013656813987871223,0.006739980451709711,0.0008659762797036517,0.00015752826457089876,0.0006458028699960402,0.0003699629351176445,0.008113363369476882,0.3252351331926626,0.013092653973539259,0.014993320214580944,0.0007171013518449016,0.040611036727376176,9.37535183369516e-06,0.0002732713720306201,0.018021288733846452,0.022029106286373567]
for idx, (f, i) in enumerate(zip(feature_idx, importance)):
    print(f'{feature_columns6[f]}: {i}')

# COMMAND ----------

# Evaluate the Model
evaluator6 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction")

# RMSE (Root Mean Squared Error)
rmse6 = evaluator6.evaluate(test_predictions6, {evaluator6.metricName: "rmse"})
print(f"Root Mean Squared Error (RMSE): {rmse6}")

# MSE (Mean Squared Error)
mse6 = evaluator6.evaluate(test_predictions6, {evaluator6.metricName: "mse"})
print(f"Mean Squared Error (MSE): {mse6}")

# MAE (Mean Absolute Error)
mae6 = evaluator6.evaluate(test_predictions6, {evaluator6.metricName: "mae"})
print(f"Mean Absolute Error (MAE): {mae6}")

# R2 (R-squared)
r26 = evaluator6.evaluate(test_predictions6, {evaluator6.metricName: "r2"})
print(f"R-squared (R2): {r26}")


# COMMAND ----------

##########MODEL 7#####################

#Crash Binary Added




# Split the data into training and test sets (80% train, 20% test)
train_data7, test_data7 = DATA.randomSplit([0.8, 0.2], seed=42)

feature_columns7 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio", "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS", "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded", "SPEED_LIMIT_DS", "ROAD_LANES_DS", "crash"]

# TRI_RAW_START
assembler7 = VectorAssembler(inputCols=feature_columns7, outputCol="features", handleInvalid="keep")
df_train7 = assembler7.transform(train_data7)
df_test7 = assembler7.transform(test_data7)

# Create a RandomForestRegressor model
rf7 = RandomForestRegressor(featuresCol="features", labelCol="Crashes Per Event", numTrees=5)

# Train the model
rf_model7 = rf7.fit(df_train7)

# Make predictions on the test data
test_predictions7 = rf_model7.transform(df_test7)

train_predictions7 = rf_model7.transform(df_train7)

# Evaluate the model
evaluator7 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
train_rmse7 = evaluator7.evaluate(train_predictions7)
print(f"Root Mean Squared Error (RMSE) on test data: {train_rmse7}")

# Print feature importances
print("Feature Importances:")
print(rf_model7.featureImportances)


# COMMAND ----------

# Create a dictionary to associate feature names with their importances
feature_importance_dict = dict(zip(feature_columns7, rf_model7.featureImportances))

# Print feature importances
print("Feature Importances:")
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")

# COMMAND ----------

# Access feature importances
feature_importances = rf_model7.featureImportances

# Convert to a list and print
print("Feature Importances:")
print(list(feature_importances))

# COMMAND ----------

feature_idx = [0,1,2,3,5,6,7,8,9,25,26,27,29,30,31,40,41,42]
importance = [0.09845112095571883,0.07935055117583076,0.04139732034592923,0.43345233194408533,0.001959907316357533,0.00021937969570902982,0.004288009056008103,0.017891666888799746,0.021136055515824582,0.02770201708328576,0.016639317821907633,0.005569184997668415,0.0008789895306879078,0.00023030945021735406,0.038463959750615784,0.06078773336928417,0.000878385071898394,0.1507037600301713]
for idx, (f, i) in enumerate(zip(feature_idx, importance)):
    print(f'{feature_columns7[f]}: {i}')

# COMMAND ----------

# Evaluate the Model
evaluator7 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction")

# RMSE (Root Mean Squared Error)
rmse7 = evaluator7.evaluate(test_predictions7, {evaluator7.metricName: "rmse"})
print(f"Root Mean Squared Error (RMSE): {rmse7}")

# MSE (Mean Squared Error)
mse7 = evaluator7.evaluate(test_predictions7, {evaluator7.metricName: "mse"})
print(f"Mean Squared Error (MSE): {mse7}")

# MAE (Mean Absolute Error)
mae7 = evaluator7.evaluate(test_predictions7, {evaluator7.metricName: "mae"})
print(f"Mean Absolute Error (MAE): {mae7}")

# R2 (R-squared)
r27 = evaluator7.evaluate(test_predictions7, {evaluator7.metricName: "r2"})
print(f"R-squared (R2): {r27}")


# COMMAND ----------

#########Model 8###########

#hyperparameter tuning




# Split the data into training and test sets (80% train, 20% test)
train_data8, test_data8 = DATA.randomSplit([0.8, 0.2], seed=42)


# Define the feature columns
feature_columns8 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio", "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio", "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS", "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded", "SPEED_LIMIT_DS", "ROAD_LANES_DS", "crash"]

# Assemble features
assembler8 = VectorAssembler(inputCols=feature_columns8, outputCol="features", handleInvalid="keep")

# Transform the data
df_train8 = assembler8.transform(train_data8)
df_test8 = assembler8.transform(test_data8)

# Create a RandomForestRegressor model
rf8 = RandomForestRegressor(featuresCol="features", labelCol="Crashes Per Event")

# Define a parameter grid for tuning
param_grid = ParamGridBuilder() \
    .addGrid(rf8.numTrees, [5, 10, 20]) \
    .addGrid(rf8.maxDepth, [5, 10, 15]) \
    .build()

# Define the evaluator
evaluator8 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")

# Create a CrossValidator with 5-fold cross-validation
crossval = CrossValidator(estimator=rf8,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator8,
                          numFolds=5)

# Train the model using CrossValidator
cv_model = crossval.fit(df_train8)

# Make predictions on the test data
test_predictions8 = cv_model.transform(df_test8)

train_predictions8 = cv_model.transform(df_train8)

# Evaluate the model
train_rmse8 = evaluator8.evaluate(train_predictions8)
print(f"Root Mean Squared Error (RMSE) on test data: {train_rmse8}")

# Print best model's feature importances
best_model = cv_model.bestModel
print("Best Model's Feature Importances:")
print(best_model.featureImportances)


# COMMAND ----------

######FEATURE TESTING#################


from pyspark.ml.feature import ChiSqSelector


# Define feature columns (excluding the dependent variable 'Crashes Per Event')
feature_columns7 = ["Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio",
                    "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio",
                    "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS",
                    "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded",
                    "SPEED_LIMIT_DS", "ROAD_LANES_DS"]

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns7, outputCol="features", handleInvalid="keep")

# Transform the data to include the assembled features
assembled_data = assembler.transform(DATA)

# Create a ChiSqSelector
selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selected_features", labelCol="Crashes Per Event")

# Apply the selector to the data
selected_data = selector.fit(assembled_data).transform(assembled_data)

# Print selected features
selected_feature_indices = selector.selectedFeatures
selected_features = [feature_columns7[i] for i in selected_feature_indices]
print("Selected Features:")
for feature in selected_features:
    print(feature)


# COMMAND ----------

############MODEL 9############


from pyspark.ml.feature import VectorSlicer, ChiSqSelector
from pyspark.ml import Pipeline

# Split the data into training and test sets (80% train, 20% test)
train_data9, test_data9 = DATA.randomSplit([0.8, 0.2], seed=42)

feature_columns9 = [
    "Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio",
    "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio",
    "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS",
    "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded",
    "SPEED_LIMIT_DS", "ROAD_LANES_DS", "crash"
]

# Assemble features
assembler9 = VectorAssembler(inputCols=feature_columns9, outputCol="raw_features", handleInvalid="keep")

# Create a RandomForestRegressor model
rf9 = RandomForestRegressor(featuresCol="raw_features", labelCol="Crashes Per Event", numTrees=5)

# Create a feature selector for numeric features
numeric_slicer9 = VectorSlicer(inputCol="raw_features", outputCol="numeric_features",
                               names=["ARC_SLOPE_DS", "TRI_RAW_START", "ROAD_LANES_DS", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS","Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio","Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio"])

# Create a feature selector for categorical features
categorical_slicer9 = VectorSlicer(inputCol="raw_features", outputCol="categorical_features",
                                   names=["INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded", "crash"])

# Create a ChiSqSelector for categorical features
selector9 = ChiSqSelector(numTopFeatures=5, featuresCol="categorical_features",
                          outputCol="selected_categorical_features", labelCol="Crashes Per Event")

# Assemble final features
final_assembler9 = VectorAssembler(
    inputCols=["numeric_features", "selected_categorical_features"],
    outputCol="features"
)

# Create a pipeline
pipeline9 = Pipeline(stages=[assembler9, numeric_slicer9, categorical_slicer9, selector9, final_assembler9, rf9])

# Train the model
model9 = pipeline9.fit(train_data9)

# Make predictions on the test data
predictions9 = model9.transform(test_data9)

# Evaluate the model
evaluator9 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
rmse9 = evaluator9.evaluate(predictions9)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse9}")

# Print feature importances
print("Feature Importances:")
print(model9.stages[-1].featureImportances)




# COMMAND ----------

##########MODEL 10#####################

#XGBOOST

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

# Split the data into training and test sets (80% train, 20% test)
train_data_xgb, test_data_xgb = DATA.randomSplit([0.8, 0.2], seed=42)

feature_columns_xgb = [
    "Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio",
    "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio",
    "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS",
    "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded",
    "SPEED_LIMIT_DS", "ROAD_LANES_DS", "crash"
]

# Assemble features
assembler_xgb = VectorAssembler(inputCols=feature_columns_xgb, outputCol="features", handleInvalid="keep")

# Create an XGBoost model
xgb = GBTRegressor(featuresCol="features", labelCol="Crashes Per Event", maxDepth=5, maxBins=32, maxIter=20, seed=42)

# Assemble and transform the data
df_train_xgb = assembler_xgb.transform(train_data_xgb)
df_test_xgb = assembler_xgb.transform(test_data_xgb)

# Train the model
xgb_model = xgb.fit(df_train_xgb)

# Make predictions on the test data
test_predictions_xgb = xgb_model.transform(df_test_xgb)

train_predictions_xgb = xgb_model.transform(df_train_xgb)

# Evaluate the model
evaluator_xgb = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
train_rmse_xgb = evaluator_xgb.evaluate(train_predictions_xgb)
test_rmse_xgb = evaluator_xgb.evaluate(test_predictions_xgb)

print(f"Root Mean Squared Error (RMSE) on training data: {train_rmse_xgb}")
print(f"Root Mean Squared Error (RMSE) on test data: {test_rmse_xgb}")


# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator



# Evaluate the model on training data
train_predictions_xgb = xgb_model.transform(df_train_xgb)
train_rmse_xgb = evaluator_xgb.evaluate(train_predictions_xgb, {evaluator_xgb.metricName: "rmse"})
train_mse_xgb = evaluator_xgb.evaluate(train_predictions_xgb, {evaluator_xgb.metricName: "mse"})
train_mae_xgb = evaluator_xgb.evaluate(train_predictions_xgb, {evaluator_xgb.metricName: "mae"})
train_r2_xgb = evaluator_xgb.evaluate(train_predictions_xgb, {evaluator_xgb.metricName: "r2"})

print("Metrics on training data:")
print(f"Root Mean Squared Error (RMSE): {train_rmse_xgb}")
print(f"Mean Squared Error (MSE): {train_mse_xgb}")
print(f"Mean Absolute Error (MAE): {train_mae_xgb}")
print(f"R-squared (R2): {train_r2_xgb}")

# Evaluate the model on test data
test_predictions_xgb = xgb_model.transform(df_test_xgb)
test_rmse_xgb = evaluator_xgb.evaluate(test_predictions_xgb, {evaluator_xgb.metricName: "rmse"})
test_mse_xgb = evaluator_xgb.evaluate(test_predictions_xgb, {evaluator_xgb.metricName: "mse"})
test_mae_xgb = evaluator_xgb.evaluate(test_predictions_xgb, {evaluator_xgb.metricName: "mae"})
test_r2_xgb = evaluator_xgb.evaluate(test_predictions_xgb, {evaluator_xgb.metricName: "r2"})

print("\nMetrics on test data:")
print(f"Root Mean Squared Error (RMSE): {test_rmse_xgb}")
print(f"Mean Squared Error (MSE): {test_mse_xgb}")
print(f"Mean Absolute Error (MAE): {test_mae_xgb}")
print(f"R-squared (R2): {test_r2_xgb}")


# COMMAND ----------

#Feature Importance

# Get the feature importances from the XGBoost model
feature_importance_scores = xgb_model.featureImportances

# Create a mapping of feature names to their indices
feature_mapping = dict(enumerate(feature_columns_xgb))

# Display feature importances with feature names
print("Feature Importances:")
for index, importance in enumerate(feature_importance_scores):
    feature_name = feature_mapping.get(index, f"Feature_{index + 1}")
    print(f"{feature_name}: {importance}")


# COMMAND ----------

# Get the feature importances from the XGBoost model
feature_importance_scores = xgb_model.featureImportances

# Get the input columns used in the VectorAssembler
input_cols = assembler_xgb.getInputCols()

# Display feature importances with feature names
print("Feature Importances:")
for col_index, importance in enumerate(feature_importance_scores):
    col_name = input_cols[col_index]
    print(f"{col_name}: {importance}")

# COMMAND ----------

##########MODEL 11#####################

#XGBOOST2 NO CRASH


# Split the data into training and test sets (80% train, 20% test)
train_data_xgb2, test_data_xgb2 = DATA.randomSplit([0.8, 0.2], seed=42)

feature_columns_xgb2 = [
    "Acceleration Event Ratio", "Braking Event Ratio", "Excessive Speeding Ratio",
    "Phone Handling Event Ratio", "Potential Collision Event Ratio", "Speeding Event Ratio",
    "ARC_SLOPE_DS", "TRI_RAW_START", "ARC_LENGTH_DS", "CURVE_MAX_ANGLE_DS",
    "INTERACTION_POI_DS_encoded", "CIRCULATION_POI_DS_encoded", "ROAD_PRIORITY_DS_encoded",
    "SPEED_LIMIT_DS", "ROAD_LANES_DS"
]

# Assemble features
assembler_xgb2 = VectorAssembler(inputCols=feature_columns_xgb2, outputCol="features", handleInvalid="keep")

# Create an XGBoost model
xgb2 = GBTRegressor(featuresCol="features", labelCol="Crashes Per Event", maxDepth=5, maxBins=32, maxIter=20, seed=42)

# Assemble and transform the data
df_train_xgb2 = assembler_xgb2.transform(train_data_xgb2)
df_test_xgb2 = assembler_xgb2.transform(test_data_xgb2)

# Train the model
xgb_model2 = xgb2.fit(df_train_xgb2)

# Make predictions on the test data
test_predictions_xgb2 = xgb_model2.transform(df_test_xgb2)

train_predictions_xgb2 = xgb_model2.transform(df_train_xgb2)

# Evaluate the model
evaluator_xgb2 = RegressionEvaluator(labelCol="Crashes Per Event", predictionCol="prediction", metricName="rmse")
train_rmse_xgb2 = evaluator_xgb2.evaluate(train_predictions_xgb2)
test_rmse_xgb2 = evaluator_xgb2.evaluate(test_predictions_xgb2)

print(f"Root Mean Squared Error (RMSE) on training data: {train_rmse_xgb2}")
print(f"Root Mean Squared Error (RMSE) on test data: {test_rmse_xgb2}")


# COMMAND ----------


# Evaluate the model on training data
train_predictions_xgb2 = xgb_model2.transform(df_train_xgb2)
train_rmse_xgb2 = evaluator_xgb2.evaluate(train_predictions_xgb2, {evaluator_xgb2.metricName: "rmse"})
train_mse_xgb2 = evaluator_xgb2.evaluate(train_predictions_xgb2, {evaluator_xgb2.metricName: "mse"})
train_mae_xgb2 = evaluator_xgb2.evaluate(train_predictions_xgb2, {evaluator_xgb2.metricName: "mae"})
train_r2_xgb2 = evaluator_xgb2.evaluate(train_predictions_xgb2, {evaluator_xgb2.metricName: "r2"})

print("Metrics on training data:")
print(f"Root Mean Squared Error (RMSE): {train_rmse_xgb2}")
print(f"Mean Squared Error (MSE): {train_mse_xgb2}")
print(f"Mean Absolute Error (MAE): {train_mae_xgb2}")
print(f"R-squared (R2): {train_r2_xgb2}")

# COMMAND ----------

# Get the feature importances from the XGBoost model
feature_importance_scores2 = xgb_model2.featureImportances

# Get the input columns used in the VectorAssembler
input_cols2 = assembler_xgb2.getInputCols()

# Display feature importances with feature names
print("Feature Importances:")
for col_index, importance in enumerate(feature_importance_scores2):
    col_name = input_cols2[col_index]
    print(f"{col_name}: {importance}")

# COMMAND ----------


import pandas as pd
from IPython.display import display, HTML

# Create a Spark DataFrame with the counts
data = [("Actual Negative", "Predicted Negative", 468166),
        ("Actual Negative", "Predicted Positive", 1466),
        ("Actual Positive", "Predicted Negative", 0),
        ("Actual Positive", "Predicted Positive", 2215)]

columns = ["Actual", "Predicted", "Count"]
df = spark.createDataFrame(data, columns)

# Convert Spark DataFrame to Pandas DataFrame for better formatting
confusion_matrix_pd = df.toPandas()

# Pivot the DataFrame to create a confusion matrix
confusion_matrix_pd = confusion_matrix_pd.pivot(index='Actual', columns='Predicted', values='Count')

# Display the confusion matrix with color highlighting
cm_style = confusion_matrix_pd.style\
    .applymap(lambda x: 'color: white; background-color: #4CAF50', subset=pd.IndexSlice['Actual Negative', 'Predicted Negative'])\
    .applymap(lambda x: 'color: white; background-color: #FF5722', subset=pd.IndexSlice['Actual Negative', 'Predicted Positive'])\
    .applymap(lambda x: 'color: white; background-color: #FF5722', subset=pd.IndexSlice['Actual Positive', 'Predicted Negative'])\
    .applymap(lambda x: 'color: white; background-color: #4CAF50', subset=pd.IndexSlice['Actual Positive', 'Predicted Positive'])

# Display the styled confusion matrix
display(HTML(cm_style.render()))



# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Set a color palette similar to the one used for styling
color_palette = ['#4CAF50', '#FF5722']

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_pd, annot=True, fmt='d', cmap=sns.color_palette(color_palette))

# Set labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Save the plot as an image
plt.savefig('confusion_matrix.png')

# Show the plot
plt.show()


# COMMAND ----------

