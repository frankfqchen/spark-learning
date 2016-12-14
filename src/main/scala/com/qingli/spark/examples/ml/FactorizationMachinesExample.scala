package com.qingli.spark.examples.ml

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.fm.FactorizationMachines
import org.apache.spark.sql.SparkSession

/**
  * Created by Qing Lee on 16-12-14.
  */
object FactorizationMachinesExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("FactorizationMachinesExample")
      .master("local[4]")
      .getOrCreate()

    val train = spark.read.format("libsvm").load("data/a9a.train")
    val test = spark.read.format("libsvm").load("data/a9a.test")

    val fm = new FactorizationMachines()
      .setTask("classification")
      .setSolver("gd")
      .setInitialStd(0.01)
      .setStepSize(0.01)
      .setUseBiasTerm(true)
      .setUseLinearTerms(true)
      .setNumFactors(10)
      .setRegParams((0, 1e-3, 1e-4))
      .setTol(1e-3)
      .setMaxIter(100)
      .setMiniBatchFraction(1)

    val fmModel = fm.fit(train)
    val result = fmModel.transform(test)
    val predictionAndLabel = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    println("Accuracy: " + evaluator.evaluate(predictionAndLabel))

    spark.stop()
  }

}
