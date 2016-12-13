package org.apache.spark.ml.optimization

import breeze.linalg.norm
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.{Gradient, Optimizer, Updater}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by Qing Lee on 16-12-13.
  */
class ParallelStochasticGradientDescent private[spark](private var gradient: Gradient,
                                                       private var updater: Updater) extends Optimizer with Logging {
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.1
  private var convergenceTol: Double = 0.0001

  /**
    * Set the initial step size of parallel SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    *
    * @param stepSize
    * @return
    */
  def setStepSize(stepSize: Double): this.type = {
    require(stepSize > 0, s"Initial step size must be positive, but got ${stepSize}")

    this.stepSize = stepSize
    this
  }


  /**
    * Set the number of iterations for parallel SGD. Default 100.
    *
    * @param numIterations
    * @return
    */
  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations >= 0, s"Number of iterations must be non negative, but got ${numIterations}")

    this.numIterations = numIterations
    this
  }


  /**
    * Set the regularization parameter. Default 0.1
    *
    * @param regParam
    * @return
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0, s"Regularization parameter must be non negative, but got ${regParam}")

    this.regParam = regParam
    this
  }


  /**
    * Set the convergence tolerance. Default 0.0001.
    *
    * @param convergenceTol
    * @return
    */
  def setConvergenceTol(convergenceTol: Double): this.type = {
    require(convergenceTol >= 0 && convergenceTol <= 1.0, s"Convergence tolerance must be in range [0, 1], but got ${convergenceTol}")

    this.convergenceTol = convergenceTol
    this
  }


  /**
    * Set the gradient function (of loss function on one single data sample) to be used by Parallel SGD.
    *
    * @param gradient
    * @return
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a give direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determine what kind or regularization is used, if any.
    *
    * @param updater
    * @return
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
    *
    * @param data
    * @param initialWeights
    * @return
    */
  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = ParallelStochasticGradientDescent.runParallelSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      initialWeights,
      convergenceTol)

    weights
  }
}

object ParallelStochasticGradientDescent extends Logging {

  private def isConverged(previousWeights: Vector,
                          currentWeights: Vector,
                          convergenceTol: Double): Boolean = {

    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    val vecDiff: Double = norm(previousBDV - currentBDV)

    vecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }

  def runParallelSGD(data: RDD[(Double, Vector)],
                     gradient: Gradient,
                     updater: Updater,
                     stepSize: Double,
                     numIterations: Int,
                     regParam: Double,
                     initialWeights: Vector,
                     convergenceTol: Double): (Vector, Array[Double]) = {

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numSamples = data.count()

    if (numSamples == 0) {
      logWarning("No data found, ParallelStochasticGradientDescent.runParallelSGD return the initial weights.")

      return (initialWeights, stochasticLossHistory.toArray)
    }

    // Initialize weights as a column vector.
    var weights = Vectors.dense(initialWeights.toArray)

    var converged = false
    var iter = 1

    while (!converged && iter <= numIterations) {
      val bcWeights = data.context.broadcast(weights)

      val (avgWeights, avgRegVal, lossSum, batchSize) = data.mapPartitions { part =>
        var localWeights = bcWeights.value
        var localRegVal = 0.0
        var localLossSum = 0.0
        var j = 1
        while (part.hasNext) {
          val (label, vector) = part.next()
          val (localGrad, localLoss) = gradient.compute(vector, label, localWeights)
          val update = updater.compute(localWeights, localGrad, stepSize, j, regParam)

          localWeights = update._1
          localRegVal = update._2
          localLossSum += localLoss
          j += 1
        }

        Iterator.single((localWeights, localRegVal, localLossSum, j))
      }.treeReduce { case ((w1, rv1, ls1, c1), (w2, rv2, ls2, c2)) =>
        val avgWeights = (w1.asBreeze * c1.toDouble + w2.asBreeze * c2.toDouble) / (c1 + c2).toDouble
        val avgRegVal = (rv1 * c1.toDouble + rv2 * c2.toDouble) / (c1 + c2).toDouble

        (Vectors.fromBreeze(avgWeights), avgRegVal, ls1 + ls2, c1 + c2)
      }

      stochasticLossHistory.append(lossSum / batchSize + avgRegVal)
      weights = avgWeights
      previousWeights = currentWeights
      currentWeights = Some(weights)

      if (previousWeights.isDefined && currentWeights.isDefined) {
        converged = isConverged(previousWeights.get, currentWeights.get, convergenceTol)
      }

      iter += 1
    }

    logInfo(("ParallelStochasticGradientDescent.runParallelSGD finished. " +
      "Last 10 stochastic losses %s").format(stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)
  }
}
