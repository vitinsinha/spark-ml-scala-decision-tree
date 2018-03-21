import _root_.sbtassembly.AssemblyPlugin.autoImport._
import _root_.sbtassembly.PathList

name := "spark-ml-scala-decision-tree"

version := "1.0"

scalaVersion := "2.11.8"

resolvers  += "MavenCentralRepository" at "http://central.maven.org/maven2"

libraryDependencies ++= {
  val sparkVer = "2.2.1"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer,
    "org.apache.spark" %% "spark-mllib" % sparkVer,
    "org.apache.spark" %% "spark-sql" % sparkVer
  )
}

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

    