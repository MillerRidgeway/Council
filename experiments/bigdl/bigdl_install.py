from bigdl.util.common import *
from pyspark import SparkContext
from bigdl.nn.layer import *
import bigdl.version

# create sparkcontext with bigdl configuration
sc = SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[*]"))
init_engine() # prepare the bigdl environment 
bigdl.version.__version__ # Get the current BigDL version
linear = Linear(2, 3) # Try to create a Linear layer