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

# Read Excel file using pandas
try:
    pandas_df = pd.read_excel('BigChungus.xlsx')
except Exception as e:
    print(f"Error reading Excel file: {e}")
    raise

spark_df = spark.createDataFrame(pandas_df)

# Get all column names
all_cols = pandas_df.columns.tolist()

# Define the label column
label_col = 'LABEL_Next_Month_Inflation'

# Initial feature columns
base_feature_cols = [c for c in all_cols if c != label_col and c != 'Unnamed: 0']

# Create time-based features
from pyspark.sql.functions import unix_timestamp, year, month
sprk_df_with_time_features = spark_df \
    .withColumn('Date_Unix', unix_timestamp(spark_df['Unnamed: 0'])) \
    .withColumn('Year', year(spark_df['Unnamed: 0'])) \
    .withColumn('Month', month(spark_df['Unnamed: 0']))

# Define feature columns
feature_cols = ['Date_Unix', 'Year', 'Month'] + base_feature_cols

# Drop rows with NaNs in relevant columns
all_relevant_cols = feature_cols + [label_col]
cleaned_df = sprk_df_with_time_features.dropna(subset=all_relevant_cols)

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(cleaned_df)

# Split into train and test sets (train on all data)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Initialize and fit the model
lr = LinearRegression(featuresCol="features", labelCol="LABEL_Next_Month_Inflation")
model = lr.fit(train_data)

# Predict on the test data
predictions = model.transform(test_data)

# **Apply date filtering to predictions**
start_date_str = '2022-11-01'
end_date_str = '2025-12-31'
filtered_predictions = predictions.filter(
    (col('Unnamed: 0') >= start_date_str) & (col('Unnamed: 0') <= end_date_str)
)

# Evaluate the model only on the filtered predictions
evaluator = RegressionEvaluator(labelCol="LABEL_Next_Month_Inflation", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(filtered_predictions)
print(f"Root Mean Squared Error (RMSE) on test data (Nov 2022–Dec 2025) = {rmse}")

evaluator_r2 = RegressionEvaluator(labelCol="LABEL_Next_Month_Inflation", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(filtered_predictions)
print(f"R-squared (R2) on test data (Nov 2022–Dec 2025) = {r2}")

# Get feature coefficients
feature_names = assembler.getInputCols()
coefficients = model.coefficients
intercept = model.intercept

# Print coefficients
for feature, coef in zip(feature_names, coefficients.toArray()):
    print(f"{feature}: {coef}")
print(f"Intercept: {intercept}")

# Save filtered predictions to Excel
filtered_predictions_df = filtered_predictions.select('Unnamed: 0', 'LABEL_Next_Month_Inflation', 'prediction').toPandas()
try:
    filtered_predictions_df.to_excel('LinearRegressionPredictions_Nov2022_Dec2025.xlsx', index=False)
    print("Predictions saved to LinearRegressionPredictions_Nov2022_Dec2025.xlsx")
except Exception as e:
    print(f"Error saving Excel file: {e}")
    raise