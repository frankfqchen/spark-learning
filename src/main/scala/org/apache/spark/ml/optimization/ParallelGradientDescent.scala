package org.apache.spark.ml.optimization

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.{Gradient, Optimizer, Updater}
import org.apache.spark.rdd.RDD

/**
  * Created by Qing Lee on 16-12-13.
  */
class ParallelGradientDescent private[spark](private var gradient: Gradient,
                                             private var updater: Updater) extends Optimizer with Logging {
  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = ???
}
