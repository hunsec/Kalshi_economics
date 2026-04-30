# linearregression_pyspark_chronological.py
import findspark
findspark.init()  # Initialize Spark

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, to_date, year, month
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("LinearRegressionChronological") \
    .getOrCreate()

# Read Excel file using pandas
pandas_df = pd.read_excel('FRED_all.xlsx')
spark_df = spark.createDataFrame(pandas_df)

# Convert 'Unnamed: 0' to a proper date column
spark_df = spark_df.withColumn("Date", to_date(col("Unnamed: 0"), "yyyy-MM-dd"))

# Get all column names
all_cols = pandas_df.columns.tolist()

# Define the label column
label_col = 'LABEL_Next_Month_Inflation'

# Initial feature columns
base_feature_cols = [c for c in all_cols if c != label_col and c != 'Unnamed: 0' and c != 'Date']

# Create time-based features
time_feature_df = spark_df \
    .withColumn('Year', year(col("Date"))) \
    .withColumn('Month', month(col("Date")))

# Define feature columns
feature_cols = ['Year', 'Month'] + base_feature_cols

# Drop rows with NaNs in relevant columns
all_relevant_cols = feature_cols + [label_col]
cleaned_df = time_feature_df.dropna(subset=all_relevant_cols)

# Exclude data for November 2023, September 2023, and anything after December 2025
filtered_df = cleaned_df.filter(
    ~((col("Year") == 2023) & (col("Month") == 11)) &  # Exclude November 2023
    ~((col("Year") == 2023) & (col("Month") == 9)) &    # Exclude September 2023
    (col("Date") <= "2025-12-31")                          # Exclude data after December 2025
)

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(filtered_df)

# Chronological split: Train on data before November 2022, predict on data from November 2022 to December 2025
train_data = data.filter(col("Date") < "2022-12-01")
test_data = data.filter((col("Date") >= "2022-12-01") & (col("Date") <= "2025-12-31"))

# Initialize and fit the model
lr = LinearRegression(featuresCol="features", labelCol=label_col)
model = lr.fit(train_data)

# Predict on the test data
predictions = model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data (Dec 2022 - Dec 2025) = {rmse}")

evaluator_mae = RegressionEvaluator(labelCol="LABEL_Next_Month_Inflation", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)
print(f"Mean Absolute Error (MAE) on test data (Dec 2022 - Dec 2025) = {mae}")

evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print(f"R-squared (R2) on test data (Dec 2022 - Dec 2025) = {r2}")

# Get feature coefficients
feature_names = assembler.getInputCols()
coefficients = model.coefficients
intercept = model.intercept

# Print coefficients
for feature, coef in zip(feature_names, coefficients.toArray()):
    print(f"{feature}: {coef}")
print(f"Intercept: {intercept}")

# Save predictions to Excel
predictions_df = predictions.select("Unnamed: 0", label_col, "prediction").toPandas()
try:
    predictions_df.to_excel('LinearRegressionOutputDateLimit.xlsx', index=False)
    print("Predictions saved to LinearRegressionOutputDateLimit.xlsx")
except Exception as e:
    print(f"Error saving Excel file: {e}")
    raise