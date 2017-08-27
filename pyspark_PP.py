import sys
from pyspark.sql import SparkSession, functions, types
import string, re
from pyspark.sql.window import Window
from pyspark.sql.functions import col, rank, desc, udf, to_json
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql.types import IntegerType


spark = SparkSession.builder.appName('reddit averages').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+


schema = types.StructType([ # commented-out fields won't be read
    types.StructField('body', types.StringType(), False),
    types.StructField('subreddit', types.StringType(), False),
])


def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]

    comments = spark.read.json(in_directory, schema=schema)
    
    comments.cache() 
   
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation + '0123456789'),)

    # NLP processing code adapted from https://spark.apache.org/docs/latest/ml-features.html
    regexTokenizer = RegexTokenizer(inputCol="body", outputCol="words", minTokenLength=3, pattern=wordbreak)
    # alternatively, pattern="\\w+", gaps(False)

    countTokens = udf(lambda words: len(words), IntegerType())

    regexTokenized = regexTokenizer.transform(comments)
    docs = regexTokenized.select("body", "words", "subreddit")

    docs.cache()

    #extra_stop_words = ["www","http","gt"]

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    docs = remover.transform(docs).withColumn("tokens", countTokens(col("filtered")))

    docs = docs.drop("body")
    docs = docs.drop("words") 

    docs.groupBy("subreddit").agg(functions.avg("tokens")).show()

    # threshold for post length
    lthresh = 60
    uthresh = 100
    docs = docs.filter(docs['tokens'] > lthresh)
    docs = docs.filter(docs['tokens'] < uthresh)


    logs = docs.groupBy("subreddit").agg(functions.count("*")).show()


    #adds rank per subreddit type into a new column called rank
    ranked = docs.withColumn("rank", rank().over(Window.partitionBy("subreddit").orderBy(desc("tokens"))))

    #ranked.cache()

    group_size = 230
    #take group_size biggest docs from each group type
    ranked = ranked.filter(ranked['rank'] <= group_size)
    
    #convert arrays to columns so we can write csv
    for i in range(uthresh):
        ranked = ranked.withColumn('{0}'.format(i), ranked.filtered.getItem(i))

    #drop filtered so we can write to csv
    ranked = ranked.drop('filtered')
    ranked = ranked.drop('rank')
    ranked.show()

    ranked.write.csv(out_directory, mode='overwrite')


if __name__=='__main__':
    main()
