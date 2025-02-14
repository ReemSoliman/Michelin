# Databricks notebook source
options = {
"sfUrl": "ab43836.west-europe.azure.snowflakecomputing.com",
"sfUser": "bstravinskas3@gatech.edu", # Email address of your personal Snowflake account
"sfPassword": "Rangers93!", # Password you choose when first login to Snowflake
"sfDatabase": "GATECH",
"sfSchema": "GROUP_7", # Replace * by your group number
"sfWarehouse": "GATECH_WH"
}

import pandas as pd


####NUMERIC MODEL REGRESSION#####


# Read the data from EVENTS table.
EVENTS = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "EVENTS") \
.load()
CRASHES_2019_2023_MAPMATCHED = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "CRASHES_2019_2023_MAPMATCHED") \
.load()
EVENTS = EVENTS.withColumnRenamed("arc_index_hash_start", "ARC_ID_HASH")
from pyspark.sql.functions import col, count
grouped_df = CRASHES_2019_2023_MAPMATCHED.groupBy("ARC_ID_HASH").agg(count("*").alias("count"))

# Show the resulting DataFrame
grouped_df.show()
from pyspark.sql import SparkSession


# Specify the list of columns to keep
columns_to_keep = [
    'ARC_ID_HASH',
    'SAMPLE_SPEED',
    'SOLAR_POSITION',
    'SPEED_CHANGE',
    'TRI_RAW_END',
    'TRI_RAW_START',
    'ROAD_ROUNDABOUT_DS',
    'ARC_LENGTH_DS',
    'TUNNEL_DS',
    'BRIDGE_DS',
    'SPEED_LIMIT_DS',
    'CIRCULATION_FULL_ARC_DS',
    'LANES_FORWARD_DS',
    'ROAD_LANES_DS',
    'ROAD_PRIORITY_DS',
    'CURVE_TOTAL_LENGTH_DS',
    'ARC_SLOPE_DS'
]

# Select only the columns from the specified list
EVENTS = EVENTS.select(columns_to_keep)



feature_columns = [
    'SAMPLE_SPEED',
    'SOLAR_POSITION',
    'SPEED_CHANGE',
    'TRI_RAW_END',
    'TRI_RAW_START',
    'ROAD_ROUNDABOUT_DS',
    'ARC_LENGTH_DS',
    'TUNNEL_DS',
    'BRIDGE_DS',
    'CIRCULATION_FULL_ARC_DS',
    'LANES_FORWARD_DS',
    'ROAD_LANES_DS',
    'ROAD_PRIORITY_DS',
    'CURVE_TOTAL_LENGTH_DS',
    'ARC_SLOPE_DS',
    'SPEED_LIMIT_DS'
]


for col_name in feature_columns:
    col_data_type = EVENTS.schema[col_name].dataType
    print(f"Column '{col_name}' has data type: {col_data_type}")

from pyspark.sql.functions import when, col

EVENTS = EVENTS.withColumn('ROAD_ROUNDABOUT_DS', when(col('ROAD_ROUNDABOUT_DS'), 1.0).otherwise(0.0))
EVENTS = EVENTS.withColumn('TUNNEL_DS', when(col('TUNNEL_DS'), 1.0).otherwise(0.0))
EVENTS = EVENTS.withColumn('BRIDGE_DS', when(col('BRIDGE_DS'), 1.0).otherwise(0.0))
EVENTS = EVENTS.withColumn('CIRCULATION_FULL_ARC_DS', when(col('CIRCULATION_FULL_ARC_DS'), 1.0).otherwise(0.0))

# Convert Decimal columns to DoubleType
decimal_columns = ['SPEED_CHANGE', 'SPEED_LIMIT_DS', 'LANES_FORWARD_DS', 'ROAD_LANES_DS', 'ROAD_PRIORITY_DS']
for col_name in decimal_columns:
    EVENTS = EVENTS.withColumn(col_name, col(col_name).cast('double'))

missing_values = {}
for col_name in EVENTS.columns:
    missing_count = EVENTS.where(col(col_name).isNull()).count()
    missing_values[col_name] = missing_count

# Print the count of missing values for each column
for col_name, missing_count in missing_values.items():
    print(f"Column '{col_name}' has {missing_count} missing values.")

#calculate the total number of missing values in the DataFrame
total_missing = sum(missing_values.values())
print(f"Total missing values in the DataFrame: {total_missing}")
EVENTS = EVENTS.na.drop(subset=feature_columns)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession



# Specify the feature columns
feature_columns = [
    'SAMPLE_SPEED',
    'SOLAR_POSITION',
    'SPEED_CHANGE',
    'TRI_RAW_END',
    'TRI_RAW_START',
    'ROAD_ROUNDABOUT_DS',
    'ARC_LENGTH_DS',
    'TUNNEL_DS',
    'BRIDGE_DS',
    'SPEED_LIMIT_DS',
    'CIRCULATION_FULL_ARC_DS',
    'LANES_FORWARD_DS',
    'ROAD_LANES_DS',
    'ROAD_PRIORITY_DS',
    'CURVE_TOTAL_LENGTH_DS',
    'ARC_SLOPE_DS'
]



# Assemble the features into a single vector column
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
EVENTS = assembler.transform(EVENTS)


# Join the EVENTS DataFrame with the count DataFrame using ARC_ID_HASH
joined_df = EVENTS.join(grouped_df, EVENTS.ARC_ID_HASH == grouped_df.ARC_ID_HASH, "left")


# Replace null values in the "count" column with 0 (indicating no occurrences)
joined_df = joined_df.na.fill(0, subset=["count"])


# Define the dependent variable column
dependent_variable_column = "count"

# Split the data into training and testing sets
(train_data, test_data) = joined_df.randomSplit([0.8, 0.2], seed=123)


# Create a Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol=dependent_variable_column)



# Fit the model to the training data
lr_model = lr.fit(train_data)


# Make predictions on the test data
predictions = lr_model.transform(test_data)


# Evaluate the model (e.g., calculate RMSE)
evaluator = RegressionEvaluator(labelCol=dependent_variable_column, predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)


#extract the model's coefficients and intercept
coefficients = lr_model.coefficients
intercept = lr_model.intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)


# Get the coefficients
coefficients = lr_model.coefficients

# Get the feature column names
feature_columns = train_data.columns
feature_columns.remove('ARC_ID_HASH')  # Replace with your actual dependent variable column name

# Create a list of (field name, coefficient) pairs
coefficients_with_names = [(col_name, coef) for col_name, coef in zip(feature_columns, coefficients)]

# Print the coefficients with field names
for field_name, coef in coefficients_with_names:
    print(f"{field_name}: {coef}")


# COMMAND ----------

####CATEGORICAL MODEL#####


# Read the data from EVENTS table.
EVENTS_CAT = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "EVENTS") \
.load()
CRASHES = spark.read \
.format("snowflake") \
.options(**options) \
.option("dbtable", "CRASHES_2019_2023_MAPMATCHED") \
.load()
EVENTS_CAT = EVENTS_CAT.withColumnRenamed("arc_index_hash_start", "ARC_ID_HASH")
selected_columns = [
    "ARC_ID_HASH",
    "context_tri_start",
    "context_weather",
    "event_type_desc",
    "circulation_poi_ds",
    "context_road_cycle"
]

# Create a copy of the DataFrame with selected columns
EVENTS_CAT = EVENTS_CAT.select(selected_columns)
import pandas as pd
import statsmodels.api as sm

# Create a 'count' field that counts the number of crashes per road segment
count_df = CRASHES.groupBy("ARC_ID_HASH").agg(count("*").alias("count"))

# Create a 'crash' field indicating whether a crash occurred (1) or not (0)
count_df = count_df.withColumn("crash", when(count_df["count"] > 0, 1).otherwise(0))

# Show the resulting DataFrame
count_df.show()
merged_df = EVENTS_CAT.join(count_df, on='ARC_ID_HASH', how='inner')

merged_df.show()
X = merged_df[['context_tri_start', 'context_weather', 'event_type_desc', 'circulation_poi_ds', 'context_road_cycle']]
y_count = merged_df['count']
y_crash = merged_df['crash']
categorical_columns = ['context_tri_start', 'context_weather', 'event_type_desc', 'circulation_poi_ds', 'context_road_cycle']


# Use StringIndexer to convert categorical columns to numerical
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_columns]

# Assemble all features, including encoded categorical and numeric features
assembler = VectorAssembler(inputCols=[col + "_index" for col in categorical_columns],
                            outputCol="features")

# Initialize a ZIP model for count with Poisson family
zip_model_count = GeneralizedLinearRegression(family="poisson", link="log", labelCol="count")

# Initialize a ZIP model for crash with Binomial family (logit link) for inflation
zip_model_crash = GeneralizedLinearRegression(family="binomial", link="logit", labelCol="crash")

# Create a pipeline for count model
count_model_pipeline = Pipeline(stages=indexers + [assembler, zip_model_count])

# Create a pipeline for crash model
crash_model_pipeline = Pipeline(stages=indexers + [assembler, zip_model_crash])

# Fit the count model
count_model = count_model_pipeline.fit(merged_df)
crash_model = crash_model_pipeline.fit(merged_df)
predictions = count_model.transform(merged_df)
evaluator = RegressionEvaluator(labelCol="count", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)
# Fit the pipeline
pipeline_model = count_model_pipeline.fit(merged_df)

# Access the Poisson regression model from the pipeline
poisson_model = pipeline_model.stages[-1]

# Access the model summary
poisson_summary = poisson_model.summary


# Get the model's deviance
deviance = poisson_summary.deviance

# Get the null deviance (deviance for the null model)
null_deviance = poisson_summary.nullDeviance

# Calculate the residual deviance
residual_deviance = null_deviance - deviance

# Get the degrees of freedom for the model
degrees_of_freedom = poisson_summary.degreesOfFreedom



# Print out the summary statistics

print("Deviance:", deviance)
print("Null Deviance:", null_deviance)
print("Residual Deviance:", residual_deviance)
print("Degrees of Freedom:", degrees_of_freedom)


coefficients = poisson_model.coefficients

# Get the standard errors for the coefficients
std_errors = poisson_model.summary.coefficientStandardErrors

# Now, you can access the coefficients and standard errors
print("Coefficients:", coefficients)
print("Standard Errors:", std_errors)
import numpy as np

# Get the deviance from the model summary
deviance = poisson_model.summary.deviance

# Get the number of estimated parameters (coefficients)
num_params = len(poisson_model.coefficients)

# Calculate AIC (Akaike Information Criterion)
aic = 2 * num_params - 2 * deviance

# Calculate BIC (Bayesian Information Criterion)
n = poisson_model.summary.numInstances
bic = -2 * deviance + num_params * (np.log(n) - np.log(2 * np.pi))

# Print AIC and BIC
print("AIC:", aic)
print("BIC:", bic)
