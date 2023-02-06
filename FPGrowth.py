import os

os.environ["PYSPARK_PYTHON"] = r"C:\Users\miuta\PycharmProjects\pythonProject2\venv\Scripts\pyspark"

os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\miuta\PycharmProjects\pythonProject2\venv\Scripts\pyspark"

os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17.0.2"

import time
import numpy as np
import pandas as pd
import datetime as dt
from pyspark.sql.types import StringType
from pyspark.sql.functions import array
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SQLContext


spark = SparkSession.builder.appName('Basket Analysis').master('local[*]').getOrCreate()
df = pd.read_csv('./data/processed-data.csv')

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

spark_dff = sqlContext.createDataFrame(df)
df_new = spark_dff.withColumn("Description", array(spark_dff["Description"]))
start_time = time.time()

fpGrowth = FPGrowth(itemsCol="Description", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(df_new)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df_new).show()


print('Execution time: %s seconds' % (time.time() - start_time))