# -----------------------------------
# 1. START SPARK
# -----------------------------------
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CustomerStreaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# -----------------------------------
# 2. DEFINE SCHEMA (IMPORTANT)
# -----------------------------------
from pyspark.sql.types import *

schema = StructType([
    StructField("customer_id", IntegerType()),
    StructField("name", StringType()),
    StructField("city", StringType()),
    StructField("income", IntegerType())
])


# -----------------------------------
# 3. READ STREAM FROM FOLDER
# -----------------------------------
stream_df = spark.readStream \
    .schema(schema) \
    .csv("/user/stream_data/")


# -----------------------------------
# 4. COUNT CUSTOMERS
# -----------------------------------
customer_count = stream_df.groupBy().count()


# -----------------------------------
# 5. OUTPUT
# -----------------------------------
query = customer_count.writeStream \
    .format("console") \
    .outputMode("complete") \
    .start()

query.awaitTermination()