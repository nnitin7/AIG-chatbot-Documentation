from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, log
from pyspark.sql.types import FloatType
from pyspark.sql.udf import pandas_udf
import pandas as pd
import numpy as np

spark = SparkSession.builder.appName("HRETL").getOrCreate()
df = spark.read.parquet('s3://hr_bucket/raw_data.parquet')
df = df.filter(col('hire_date') > '2023-01-01').repartition(100, 'department')  # Prune for skew

@pandas_udf(FloatType())
def log_transform(series: pd.Series) -> pd.Series:
    return np.log(series + 1)

df = df.withColumn('log_salary', log_transform(col('salary')))
aggregated = df.groupBy('employee_id').agg(avg('salary').alias('avg_salary'), count('*').alias('record_count'))
aggregated = aggregated.filter(col('record_count') > 100)
aggregated.write.mode('overwrite').parquet('s3://hr_bucket/processed/')