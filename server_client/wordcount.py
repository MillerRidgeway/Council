def word_count(spark_session, file_name):
    # df=spark_session.read.csv('/Ass3/MOE1',header=True)
    lines = spark_session.sparkContext.textFile(file_name)
    words = lines.flatMap(lambda line: line.split(" "))
    wordCounts = words.countByValue()

    for word, count in wordCounts.items():
        print("{} : {}".format(word, count))
    return
