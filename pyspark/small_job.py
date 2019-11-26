from pyspark.sql import SparkSession

if __name__ == "__main__":

	sess = SparkSession \
		.builder \
		.appName("Test small job on spark") \
		.getOrCreate()

	sc = sess.sparkContext

	print(f"UI url: {sc.uiWebUrl}")

	some_list = [
		"One",
		"Two",
		"Three",
		"Four",
		"Foo",
	]

	distributed_list = sc.parallelize(some_list)

	print(distributed_list.map(lambda x: x.lower()).collect())

	sess.stop() 