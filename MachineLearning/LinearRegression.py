# linearregression_pyspark.py
import findspark
findspark.init()  # Initialize Spark

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("LinearRegressionExample") \
    .getOrCreate()

# Read Excel file using pandas (PySpark doesn't natively read Excel)
try:
    pandas_df = pd.read_excel('BigChungus.xlsx')
except Exception as e:
    print(f"Error reading Excel file: {e}")
    raise

spark_df = spark.createDataFrame(pandas_df)

# Get all column names from the original pandas DataFrame
all_cols = pandas_df.columns.tolist()

# Define the label column
label_col = 'LABEL_Next_Month_Inflation'

# Initial feature columns, excluding the label, 'Unnamed: 0', and the specified columns
base_feature_cols = [c for c in all_cols if c != label_col and c != 'Unnamed: 0']

# Create new time-based features from 'Unnamed: 0' column
from pyspark.sql.functions import col, unix_timestamp, year, month
sprk_df_with_time_features = spark_df \
    .withColumn('Date_Unix', unix_timestamp(col('Unnamed: 0'))) \
    .withColumn('Year', year(col('Unnamed: 0'))) \
    .withColumn('Month', month(col('Unnamed: 0')))

# Define the final list of feature columns
feature_cols = ['Date_Unix', 'Year', 'Month'] + base_feature_cols

# Drop rows with any NaN values from the relevant columns (features and label)
all_relevant_cols = feature_cols + [label_col]
cleaned_df = sprk_df_with_time_features.dropna(subset=all_relevant_cols)

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(cleaned_df)

# Split into train and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Initialize and fit the model
lr = LinearRegression(featuresCol="features", labelCol="LABEL_Next_Month_Inflation")
model = lr.fit(train_data)

# Make predictions for evaluation on the test split
test_predictions = model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="LABEL_Next_Month_Inflation", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(test_predictions)
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")

evaluator_r2 = RegressionEvaluator(labelCol="LABEL_Next_Month_Inflation", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(test_predictions)
print(f"R-squared (R2) on test data = {r2}")

# Get feature coefficients
feature_names = assembler.getInputCols()
coefficients = model.coefficients
intercept = model.intercept

# Print coefficients
for feature, coef in zip(feature_names, coefficients.toArray()):
    print(f"{feature}: {coef}")
print(f"Intercept: {intercept}")

# Save predictions to Excel (only for the test data)
predictions_df = test_predictions.select('Unnamed: 0', 'LABEL_Next_Month_Inflation', 'prediction').toPandas()
try:
    predictions_df.to_excel('LinearRegressionOutput.xlsx', index=False)
    print("Predictions saved to LinearRegressionOutput.xlsx")
except Exception as e:
    print(f"Error saving Excel file: {e}")
    raise