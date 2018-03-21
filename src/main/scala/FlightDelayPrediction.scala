import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by vsinha on 3/20/2018.
  */
object FlightDelayPrediction {

  case class Flight(dofM: String, dofW: String, carrier: String, tailnum: String, flnum: Int, org_id: String, origin: String,
                    dest_id: String, dest: String, crsdeptime: Double, deptime: Double, depdelaymins: Double, crsarrtime: Double,
                    arrtime: Double, arrdelay: Double, crselapsedtime: Double, dist: Int)

  def parseFlight(str: String): Flight = {
    val line = str.split(",")
    Flight(line(0), line(1), line(2), line(3), line(4).toInt, line(5), line(6), line(7), line(8), line(9).toDouble,
      line(10).toDouble, line(11).toDouble, line(12).toDouble, line(13).toDouble, line(14).toDouble, line(15).toDouble, line(16).toInt)
  }

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir", "C:\\hadoop")

    val sparkConf = new SparkConf()
      .setAppName("FlightDelayPrediction")
      .setMaster("local[2]")

    val sc = new SparkContext(sparkConf)

    // read csv file
    val data = sc.textFile("src/main/resources/flight_data.csv")

    // remove header
    val header = data.first()
    val textRDD = data.filter(row => row != header)

    val flightsRDD = textRDD.map(parseFlight).cache()

    // next step is to transform the non-numeric features into numeric values

    // transform non-numeric carrier value to numeric value
    var carrierMap: Map[String, Int] = Map()
    var index: Int = 0
    flightsRDD.map(flight => flight.carrier).distinct.collect.foreach(x => { carrierMap += (x -> index); index += 1 })

    // transform non-numeric origin value to numeric value
    var originMap: Map[String, Int] = Map()
    var index1: Int = 0
    flightsRDD.map(flight => flight.origin).distinct.collect.foreach(x => { originMap += (x -> index1); index1 += 1 })

    // transform non-numeric destination value to numeric value
    var destMap: Map[String, Int] = Map()
    var index2: Int = 0
    flightsRDD.map(flight => flight.dest).distinct.collect.foreach(x => { destMap += (x -> index2); index2 += 1 })

    // creating the features array
    val mlprep = flightsRDD.map(flight => {
      val monthday = flight.dofM.toInt - 1 // category // -1 because feature starts with 0
      val weekday = flight.dofW.toInt - 1 // category // -1 because feature starts with 0
      val crsdeptime1 = flight.crsdeptime.toInt
      val crsarrtime1 = flight.crsarrtime.toInt
      val carrier1 = carrierMap(flight.carrier) // category
      val crselapsedtime1 = flight.crselapsedtime
      val origin1 = originMap(flight.origin) // category
      val dest1 = destMap(flight.dest) // category
      val delayed = if (flight.depdelaymins > 40) 1.0 else 0.0
      Array(delayed.toDouble, monthday.toDouble, weekday.toDouble, crsdeptime1.toDouble, crsarrtime1.toDouble, carrier1.toDouble, crselapsedtime1.toDouble, origin1.toDouble, dest1.toDouble)
    })

    // create Labeled Points
    // first parameter is label or target variable which is 'delayed' in our case
    // second parameter is a vector of features
    val mldata = mlprep.map(x => LabeledPoint(x(0), Vectors.dense(x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8))))

    // split the data into training and test data set

    // mldata0 is 85% not delayed flights
    val mldata0 = mldata.filter(x => x.label == 0).randomSplit(Array(0.85, 0.15))(0)
    // mldata1 is %100 delayed flights
    val mldata1 = mldata.filter(x => x.label != 0)
    // mldata2 is mix of delayed and not delayed
    val mldata2 = mldata0 ++ mldata1

    // split mldata2 into training and test data
    val splits = mldata2.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // next step is to train the model

    /* categoricalFeaturesInfo specifies which features are categorical and how many categorical values each of those features can take.
     This is given as a map from feature index to the number of categories for that feature.*/
    var categoricalFeaturesInfo = Map[Int, Int]()
    categoricalFeaturesInfo += (0 -> 31) // day of month
    categoricalFeaturesInfo += (1 -> 7)  // day of week
    categoricalFeaturesInfo += (4 -> carrierMap.size)
    categoricalFeaturesInfo += (6 -> originMap.size)
    categoricalFeaturesInfo += (7 -> destMap.size)

    val numClasses = 2 //delayed(1) and not-delayed(0)
    val impurity = "gini"
    val maxDepth = 9
    val maxBins = 7000

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // print the decision tree
    println(model.toDebugString)

    // test the model
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    // calculate wrong and correct prediction percentage

    val wrongPrediction = labelAndPreds.filter{
      case (label, prediction) => label != prediction
    }

    val wrongCount = wrongPrediction.count()

    val correctPrediction = labelAndPreds.filter{
      case (label, prediction) => label == prediction
    }

    val correctCount = correctPrediction.count()

    println("Wrong Count: " + wrongCount)
    println("Wrong Percentage: " + (wrongCount.toDouble/testData.count()) * 100)
    println("Correct Count: " + correctCount)
    println("Correct Percentage: " + (correctCount.toDouble/testData.count()) * 100)


  }
}
