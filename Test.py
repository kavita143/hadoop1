# -------------------------------
# 1. START SPARK SESSION
# -------------------------------
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("LoanPrediction_NoPipeline") \
    .getOrCreate()


# -------------------------------
# 2. LOAD DATA
# -------------------------------
customer_df = spark.read.csv("/user/data/customer.csv", header=True, inferSchema=True)
loan_df = spark.read.csv("/user/data/loan.csv", header=True, inferSchema=True)


# -------------------------------
# 3. JOIN DATA
# -------------------------------
data = customer_df.join(loan_df, "customer_id")

# Clean null values
data = data.dropna()

data.show()


# -------------------------------
# 4. CONVERT LABEL (loan_status)
# -------------------------------
from pyspark.ml.feature import StringIndexer

label_indexer = StringIndexer(inputCol="loan_status", outputCol="label")
data = label_indexer.fit(data).transform(data)


# -------------------------------
# 5. CONVERT CATEGORICAL COLUMNS
# -------------------------------
# gender
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
data = gender_indexer.fit(data).transform(data)

# city
city_indexer = StringIndexer(inputCol="city", outputCol="city_index")
data = city_indexer.fit(data).transform(data)


# -------------------------------
# 6. CREATE FEATURE VECTOR
# -------------------------------
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[
        "age",
        "income",
        "cibil_score",   # ⭐ IMPORTANT
        "loan_amount",
        "loan_term",
        "gender_index",
        "city_index"
    ],
    outputCol="features"
)

data = assembler.transform(data)


# -------------------------------
# 7. FINAL DATASET
# -------------------------------
final_data = data.select("features", "label")

final_data.show(truncate=False)


# -------------------------------
# 8. TRAIN-TEST SPLIT
# -------------------------------
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)


# -------------------------------
# 9. TRAIN MODEL
# -------------------------------
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label")

model = lr.fit(train_data)


# -------------------------------
# 10. PREDICTIONS
# -------------------------------
predictions = model.transform(test_data)

predictions.select("features", "label", "prediction", "probability").show(truncate=False)


# -------------------------------
# 11. EVALUATION
# -------------------------------
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label")

accuracy = evaluator.evaluate(predictions)

print("Model Accuracy:", accuracy)


# -------------------------------
# 12. NEW CUSTOMER PREDICTION
# -------------------------------
new_customer = spark.createDataFrame([
    (45, "Male", 60000, "Chennai", 750, 200000, 36)
], ["age", "gender", "income", "city", "cibil_score", "loan_amount", "loan_term"])


# Apply SAME transformations manually

# gender
new_customer = gender_indexer.fit(data).transform(new_customer)

# city
new_customer = city_indexer.fit(data).transform(new_customer)

# features
new_customer = assembler.transform(new_customer)

# prediction
result = model.transform(new_customer)

result.select("prediction", "probability").show()


# -------------------------------
# 13. SAVE OUTPUT TO HDFS
# -------------------------------
predictions.write.mode("overwrite").csv("/user/output/loan_predictions", header=True)


# -------------------------------
# 14. STOP SPARK
# -------------------------------
spark.stop()