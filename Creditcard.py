# Databricks notebook source
import pyspark

# COMMAND ----------

df = sqlContext.sql("SELECT * FROM default_of_credit_card_clients_1_csv")
print("Total number of rows: %d" % df.count())


# COMMAND ----------

df.show(10)

# COMMAND ----------

from pyspark import SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CreditCard").master("local[*]") .getOrCreate()




# COMMAND ----------

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

transformed_df = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

splits = [0.7, 0.3]
training_data, test_data = transformed_df.randomSplit(splits, 13579)

print("Number of training set rows: %d" % training_data.count())
print("Number of test set rows: %d" % test_data.count())

# COMMAND ----------

from pyspark.mllib.tree import RandomForest
from time import *

start_time = time()

model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={},numTrees=25, featureSubsetStrategy="auto", impurity="gini",maxDepth=4, maxBins=32, seed=13579)

end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)

# COMMAND ----------

predictions = model.predict(test_data.map(lambda x: x.features))
labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
print("Model accuracy: %.3f%%" % (acc * 100))

# COMMAND ----------

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

from time import *
start_time = time()

gbt = GradientBoostedTrees.trainClassifier(training_data,
                                             categoricalFeaturesInfo={}, numIterations=3)

end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)

# COMMAND ----------

predictions = gbt.predict(test_data.map(lambda x: x.features))
labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
print("Model accuracy: %.3f%%" % (acc * 100))

# COMMAND ----------

from pyspark.mllib.classification import SVMWithSGD, SVMModel
from time import *
start_time = time()
svm = SVMWithSGD.train(training_data, iterations=10)
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)

# COMMAND ----------

predictions = svm.predict(test_data.map(lambda x: x.features))
labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
print("Model accuracy: %.3f%%" % (acc * 100))





