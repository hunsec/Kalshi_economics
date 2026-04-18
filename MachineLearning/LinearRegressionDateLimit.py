# linearregression_pyspark.py
import findspark
findspark.init()  # Initialize Spark

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
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
from pyspark.sql.functions import unix_timestamp, year, month
sprk_df_with_time_features = spark_df \
    .withColumn('Date_Unix', unix_timestamp(spark_df['Unnamed: 0'])) \
    .withColumn('Year', year(spark_df['Unnamed: 0'])) \
    .withColumn('Month', month(spark_df['Unnamed: 0']))

# Define the date range for filtering
start_date_str = '2022-11-01'
end_date_str = '2025-12-31'

# Convert date strings to Unix timestamps for filtering
start_timestamp = unix_timestamp(col('Unnamed: 0')).cast('long').alias('start_ts')
end_timestamp = unix_timestamp(col('Unnamed: 0')).cast('long').alias('end_ts')

# Filter the data based on the date range
filtered_df = sprk_df_with_time_features.filter(
    (col('Unnamed: 0') >= start_date_str) & (col('Unnamed: 0') <= end_date_str)
)


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

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="LABEL_Next_Month_Inflation", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")

evaluator_r2 = RegressionEvaluator(labelCol="LABEL_Next_Month_Inflation", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print(f"R-squared (R2) on test data = {r2}")

# Get feature coefficients
feature_names = assembler.getInputCols()
coefficients = model.coefficients
intercept = model.intercept

# Print coefficients
for feature, coef in zip(feature_names, coefficients.toArray()):
    print(f"{feature}: {coef}")
print(f"Intercept: {intercept}")

# Save predictions to Excel
predictions_df = predictions.select('Unnamed: 0', 'LABEL_Next_Month_Inflation', 'prediction').toPandas()
try:
    predictions_df.to_excel('LinearRegressionOutput.xlsx', index=False)
    print("Predictions saved to LinearRegressionOutput.xlsx")
except Exception as e:
    print(f"Error saving Excel file: {e}")
    raise