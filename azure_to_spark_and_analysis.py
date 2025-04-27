from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, min, max, to_timestamp
from pymongo import MongoClient
import pandas as pd

spark = SparkSession.builder \
    .appName("ReadFromAzureBlobAndWriteToMongoDB") \
    .config("fs.azure.account.key.shankari.blob.core.windows.net", "PhwIb5218yzP2Vqa6pD81K/+UMLRF2BGq48qRT41fFTUu1w92jmvR599NWCnpzYUg+aWdYa91vHg+AStD+VTcg==") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def process_and_upload(df, mongo_uri, db_name, collection_name, log_file):
    with open(log_file, "a") as log:
        try:
            log.write("Analysis started...\n")
            log.write("Data loaded from Azure successfully\n")
            df.show(5)
            df = df.dropDuplicates()
            print(df.dropDuplicates())
            quantiles = df.approxQuantile(["temperature"], [0.25, 0.75], 0.05)
            Q1, Q3 = quantiles[0][0], quantiles[0][1]
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df.filter((col("temperature") >= lower_bound) & (col("temperature") <= upper_bound))
            print(quantiles)
            df = df.withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
            p_df = df.toPandas()
            client = MongoClient(mongo_uri)
            db = client[db_name]
            collection = db[collection_name]
            data = p_df.to_dict(orient="records")
            collection.insert_many(data)

            log.write("data loaded to mongo db succesfully finished successfully.\n")
        except Exception as e:
            log.write(f"Error: {e}\n")


file_path1 = "wasbs://output@shankari.blob.core.windows.net/output/part-00000-dcc97a4b-a91b-4e50-80a4-67f71d82179f-c000.csv"
file_path2 = "wasbs://output@shankari.blob.core.windows.net/output/part-00000-434ab652-4b75-4833-83dd-cfc0e0fc9103-c000.csv"

df1 = spark.read.csv(file_path1, header=True, inferSchema=True)
df2 = spark.read.csv(file_path2, header=True, inferSchema=True)


process_and_upload(
    df1,
    mongo_uri="mongodb+srv://Prashanth:kumar@cluster0.ukgh7xo.mongodb.net/?retryWrites=true&w=majority",
    db_name="prashanth",
    collection_name="prashanth",
    log_file="process_log_db_2.txt"
)

process_and_upload(
    df2,
    mongo_uri="mongodb+srv://Prashanth:kumar@cluster0.ukgh7xo.mongodb.net/?retryWrites=true&w=majority",
    db_name="prashanth",
    collection_name="db2",
    log_file="process_log_db_1.txt"
)