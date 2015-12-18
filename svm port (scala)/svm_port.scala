/*

created on 12/17/15 (at marlton with al; samsung holiday party later; will need to port SVMs to java and since JAMA is tricky to use, lets try out breeze)

goal: implement SVM to scala. built with sbt.

miscellaneous notes:

dont need build.sbt for sbt compiling, unless there are dependencies.

breeze seems to be a better documented library than JAMA. reference: https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet

do not use underscores in variable names (as in python) since underscores are reserved in scala: https://stackoverflow.com/questions/23123290/error-expected-but-identifier-found

if youre unsure about type, create in scala repl, which will output type

*/

import breeze.linalg._

object SVM{

	/*
	loadData() returns data points from csv file.
	*/
	def loadData() : DenseMatrix[Double] = { // via sbt console, we see that csv is read as a densematrix[double]

		// reference on reading CSV files: https://www.snip2code.com/Snippet/4415/Principal-Component-Analysis-with-Breeze

		import breeze.linalg._
		import java.io.File // required for reading files

		val data= csvread(new File("data.csv")) // use java.io to create File buffer

		return data
	}


	/*
	setSupportVectors() initializes support vectors, which are previously identified during libsvm training.
	*/
	def setSupportVectors() : List[DenseVector[Double]] = {

		val sv1 = new DenseVector(Array(4.0, 7))
		val sv2 = new DenseVector(Array(7.0, 2.0))
		val sv3 = new DenseVector(Array(9.0, 5.0))
		val list_sv = List(sv1, sv2, sv3)

		return list_sv
	}


	/*
	setSupportVectors() initializes support vectors, which are previously identified during libsvm training.
	*/
	def setCoefficients() : List[DenseVector[Double]] = {
		// define coefficients and then bundle in a list
		val coef1 = new DenseVector(Array(-0.04985403))
		val coef2 = new DenseVector(Array(-0.13855825))
		val coef3 = new DenseVector(Array(0.18841229))
		val list_coef = List(coef1, coef2, coef3)

		return list_coef
	}
	
	def main(args: Array[String]){ // not using "extend Apps" means main() is required.
		
		// call helper functions to initialize values
		println("initializing parameters...")
		val list_sv = setSupportVectors()
		val list_coef = setCoefficients()
		val intercept = -5.31708909

		// load data into memory
		println("reading csv...")
		val data = loadData()

		// generate predictions
		println("predicting...")

		for (idx <- 0 to 5){
			var test = data(idx,0 to 1).t // NB transpose operation after slice
			calculateScore(test, list_sv, list_coef, intercept)
		}
		
	}


	/*
	calculateScore() is the linear SVM decision function
	*/
	def calculateScore(testx: DenseVector[Double], list_svx: List[DenseVector[Double]], list_coefx: List[DenseVector[Double]], interceptx: Double){

		var t1 = list_coefx(0) * (testx dot list_svx(0))
		var t2 = list_coefx(1) * (testx dot list_svx(1))
		var t3 = list_coefx(2) * (testx dot list_svx(2))
		var score = t1 + t2 + t3 + interceptx

		var result = score(0) > 0 // deference from densevector, then convert to boolean

		println("predicted score is: " + score + " and the prediction is " + result)

	}
	

}