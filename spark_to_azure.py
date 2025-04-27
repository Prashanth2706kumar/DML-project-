from pyspark.sql import SparkSession

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("SparkToAzureBlob") \
    .config("fs.azure.account.key.shankari.blob.core.windows.net", "PhwIb5218yzP2Vqa6pD81K/+UMLRF2BGq48qRT41fFTUu1w92jmvR599NWCnpzYUg+aWdYa91vHg+AStD+VTcg==") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

#df1 = spark.read.csv(r"D:\data_intensive\iot_equipment_monitoring_dataset.csv", header=True, inferSchema=True)
df2 = spark.read.csv(r"D:\data_intensive\smart_manufacturing_data.csv", header=True, inferSchema=True)

output_path = "wasbs://output@shankari.blob.core.windows.net/output"
#df1.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save(output_path)
df2.coalesce(1).write.format("csv").option("header", "true").mode("append").save(output_path)
print("Data successfully written to Azure Blob Storage!")   