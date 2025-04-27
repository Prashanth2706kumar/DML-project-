from pyspark.sql.functions import mean, stddev, min, max, corr, count
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pymongo import MongoClient
import pandas as pd
from pyspark.sql import SparkSession

client = MongoClient("mongodb+srv://Prashanth:kumar@cluster0.ukgh7xo.mongodb.net/?retryWrites=true&w=majority")
db = client["prashanth"]
collection1 = db["prashanth"] 
collection2 = db["db2"] 

data1 = list(collection1.find())
data2 = list(collection2.find())

pandas_df1 = pd.DataFrame(data1)
pandas_df2 = pd.DataFrame(data2)
pandas_df1 = pandas_df1.drop(columns=["_id"])
pandas_df2 = pandas_df2.drop(columns=["_id"])
pandas_df1.to_csv("db_1_local_file.csv", index=False)
pandas_df2.to_csv("db_2_local_file.csv", index=False)
spark = SparkSession.builder.appName("PandasToSpark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df1 = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("db_1_local_file.csv")
df2 = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("db_2_local_file.csv")
def perform_statistical_summary(df):
    print("Descriptive Statistics:")
    df.select(
        mean("temperature").alias("Mean Temperature"),
        stddev("temperature").alias("Stddev Temperature"),
        min("temperature").alias("Min Temperature"),
        max("temperature").alias("Max Temperature"),
        mean("vibration").alias("Mean Vibration"),
        stddev("vibration").alias("Stddev Vibration"),
        min("vibration").alias("Min Vibration"),
        max("vibration").alias("Max Vibration"),
        mean("pressure").alias("Mean Pressure"),
        stddev("pressure").alias("Stddev Pressure"),
        min("pressure").alias("Min Pressure"),
        max("pressure").alias("Max Pressure")
    ).show()
    temperature_data = df.select("temperature").toPandas()["temperature"]
    plt.hist(temperature_data, bins=20, color='skyblue', edgecolor='black')
    plt.title("Temperature Distribution")
    plt.xlabel("Temperature")
    plt.ylabel("Frequency")
    plt.show()
    import pandas as pd
    df_pd = df.select("timestamp", "temperature", "vibration", "pressure").toPandas()
    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp"])
    
    # Plot Time-Series Data
    plt.figure(figsize=(10, 6))
    plt.plot(df_pd["timestamp"], df_pd["temperature"], label="Temperature", color="red")
    plt.plot(df_pd["timestamp"], df_pd["vibration"], label="Vibration", color="blue")
    plt.plot(df_pd["timestamp"], df_pd["pressure"], label="Pressure", color="green")
    plt.title("Time-Series Trends")
    plt.xlabel("Timestamp")
    plt.ylabel("Sensor Values")
    plt.legend()
    plt.show()
    if("downtime_risk" in df.columns):
        print("Correlation Analysis:")
        correlation_results = {
            "Temperature vs Downtime Risk": df.select(corr("temperature", "downtime_risk")).collect()[0][0],
            "Vibration vs Downtime Risk": df.select(corr("vibration", "downtime_risk")).collect()[0][0],
            "Pressure vs Downtime Risk": df.select(corr("pressure", "downtime_risk")).collect()[0][0]
        }
        
        for key, value in correlation_results.items():
            print(f"{key}: {value:.4f}")

        correlation_df = df.select("temperature", "vibration", "pressure", "downtime_risk").toPandas()
        correlations = correlation_df.corr()
        
        # Create Heatmap
        sns.heatmap(correlations, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
        # Scatter Plot
        plt.scatter(df_pd["temperature"], correlation_df["downtime_risk"], color='purple', alpha=0.5)
        plt.title("Temperature vs Downtime Risk")
        plt.xlabel("Temperature")
        plt.ylabel("Downtime Risk")
        plt.show()
        # Frequency Counts for Categorical Data
        print("Frequency Counts:")
        df.groupBy("machine_status").count().show()
        df.groupBy("failure_type").count().show()
        df.groupBy("maintenance_required").count().show()


df2 = df2.withColumn("downtime_risk",
    when(col("downtime_risk") < 0.33, 0).   # Low
    when((col("downtime_risk") >= 0.33) & (col("downtime_risk") < 0.66), 1).  # Medium
    otherwise(2)  # High
)

perform_statistical_summary(df1)
perform_statistical_summary(df2)

def save_statistical_summary(df, filename):
    summary_df = df.select(
        mean("temperature").alias("Mean Temperature"),
        stddev("temperature").alias("Stddev Temperature"),
        min("temperature").alias("Min Temperature"),
        max("temperature").alias("Max Temperature"),
        mean("vibration").alias("Mean Vibration"),
        stddev("vibration").alias("Stddev Vibration"),
        min("vibration").alias("Min Vibration"),
        max("vibration").alias("Max Vibration"),
        mean("pressure").alias("Mean Pressure"),
        stddev("pressure").alias("Stddev Pressure"),
        min("pressure").alias("Min Pressure"),
        max("pressure").alias("Max Pressure")
    ).toPandas()
    
    summary_df.to_csv(filename, index=False)    
save_statistical_summary(df1, "df1_summary.csv")
save_statistical_summary(df2, "df2_summary.csv")


feature_columns = ["temperature", "vibration", "pressure", "energy_consumption"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_ml = assembler.transform(df2).select("features", "downtime_risk")

train_data, test_data = df_ml.randomSplit([0.8, 0.2])

rf = RandomForestClassifier(labelCol="downtime_risk", featuresCol="features")
rf_model = rf.fit(train_data)


predictions = rf_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="downtime_risk", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

assembler = VectorAssembler(inputCols=["temperature", "vibration", "pressure"], outputCol="features")
df_rul = assembler.transform(df2).select("features", "predicted_remaining_life")

feature_importance = pd.DataFrame(list(zip(feature_columns, rf_model.featureImportances)), columns=["Feature", "Importance"])
feature_importance.to_csv("feature_importance.csv", index=False)


train_data, test_data = df_rul.randomSplit([0.8, 0.2])

lr = LinearRegression(labelCol="predicted_remaining_life", featuresCol="features")
lr_model = lr.fit(train_data)
#predicted_value=lr_model.predict(train_data)

test_results = lr_model.evaluate(test_data)
print(f"RMSE: {test_results.rootMeanSquaredError}")


assembler = VectorAssembler(inputCols=["Normalized_Temp", "Normalized_Vibration", "Anomaly_Score"], outputCol="features")
df_anomaly = assembler.transform(df1).select("features")


kmeans = KMeans(k=3, seed=1)
kmeans_model = kmeans.fit(df_anomaly)
centers = kmeans_model.clusterCenters()
cluster_centers = pd.DataFrame(kmeans_model.clusterCenters(), columns=["Center_Temp", "Center_Vibration", "Center_Anomaly"])
cluster_centers.to_csv("cluster_centers.csv", index=False)

print(f"Cluster Centers: {centers}")