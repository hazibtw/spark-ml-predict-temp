package com.mapr.mlib
import org.apache.log4j.{Logger}

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.evaluation.RegressionMetrics


import org.apache.spark.mllib.linalg.Vectors

object PredcitWeatherTemp {
  
   @transient lazy val logger = Logger.getLogger(getClass.getName)
   
   //year, month, day, week,temp_2,temp_1,average,actual,forecast_noaa,forecast_acc,forecast_under,friend
  // 2016,  1,      1, Fri,   45,     45,   45.6,    45,        43,           50,       44,          29
  // 2016,  7,      25,Mon,75,80,77.1,85,75,82,76,81
   //(85.0,[2016.0,5.0,26.0,0.0,71.0,78.0,72.2,70.0,74.0,72.0,84.0])

  case class Weather(
      year:Integer,
      month:Integer,day:Integer,week:String,temp_2:Double,temp_1:Double,
     average:Double,actual:Double,forecast_noaa:Double,forecast_acc:Double,forecast_under:Double,
     friend:Double
  )
  
  def parseWeather(line:String):Weather={
     
     val fields = line.split(',')  
    val weather:Weather = Weather(fields(0).toInt, fields(1).toInt, fields(2).toInt, fields(3),
        fields(4).toDouble, fields(5).toDouble, fields(6).toDouble, fields(7).toDouble,
         fields(8).toDouble, fields(9).toDouble, fields(10).toDouble, fields(11).toDouble
    
    )
    return weather     
}
   
   def main(args:Array[String]){
     
     val name = "Predict weather Application"
    logger.info(s"Starting up $name")
    
    val conf = new SparkConf().setAppName(name).setMaster("local[*]").set("spark.cores.max", "2")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    
    val sqlContext = new SQLContext(sc);
    import sqlContext._
    import sqlContext.implicits._
    
    val weatherData = sc.textFile("hdfs://localhost:8020/spark/mlib/randomforest/weather/temps.csv")
    
    
    val header = weatherData.first()

  val filterData = weatherData.filter(x => x != header)
  
    
    val weatherDF = filterData.map(parseWeather).cache()
    
    
    val weekIndex =
    weatherDF
      .map(_.week).distinct
      .collect
      .zipWithIndex
      .toMap
      
      val cols =Array("actual","year","month","day","week","temp_2","temp_1",
        "average","forecast_noaa","forecast_acc","forecast_under","friend")
        
        // year|month|day| week|temp_2|temp_1|average|actual|forecast_noaa|forecast_acc|forecast_under|friend|weekIndex
      
      val features = weatherDF map { weather =>
    val actual = weather.actual.toDouble
    val year = weather.year.toDouble
    val month = weather.month.toInt - 1
    val day = weather.day.toInt - 1
    val week = weekIndex(weather.week).toDouble
    val temp2 = weather.temp_2.toDouble
    val temp1 = weather.temp_1.toDouble
    val average = weather.average.toDouble
    val forecast_noaa = weather.forecast_noaa.toDouble
    val forecast_acc = weather.forecast_acc.toDouble
    val forecast_under = weather.forecast_under.toDouble
    val friend = weather.friend.toDouble
    
   
    Array(actual, year, month, day, week, temp2, temp1,average, forecast_noaa, forecast_acc,forecast_under,friend)
  }
    
    val labeled = features map { x =>
    LabeledPoint(x(0), Vectors.dense(x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9),x(10),x(11)))
  }
    
    val splitData = labeled.randomSplit(Array(0.75,0.25));
    
    val (trainingData, testData) = (splitData(0), splitData(1))
trainingData.take(1)

    
    val categoricalFeaturesInfo = ((1 -> 12) :: (2 -> 31) :: (3 -> weekIndex.size) :: Nil).toMap
    
   

    
    val numClasses = 2
val numTrees = 1000 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "variance"
val maxDepth = 3
val maxBins = 32


val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)






                                            
  model.toDebugString
println(model.toDebugString)
    
    // Evaluate model on test instances and compute test error
val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
    val metrics = new RegressionMetrics(labelsAndPredictions)


    
val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSE)
//println("Learned regression forest model:\n" + model.toDebugString)

println(s"MSE = ${metrics.meanSquaredError}")
println(s"RMSE = ${metrics.rootMeanSquaredError}")

// R-squared
println(s"R-squared = ${metrics.r2}")

// Mean absolute error
println(s"MAE = ${metrics.meanAbsoluteError}")

// Explained variance
println(s"Explained variance = ${metrics.explainedVariance}")

    
    
    
   }
  
}