from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from pyspark import SparkContext, SQLContext
from pyspark.sql.types import StringType
from sparkdl import KerasImageFileTransformer

#Load context
sc = SparkContext()
sqlContext = SQLContext(sc)

def loadAndPreprocessKerasInceptionV3(uri):
  # this is a typical way to load and prep images in keras
  image = img_to_array(load_img(uri, target_size=(299, 299)))  # image dimensions for InceptionV3
  image = np.expand_dims(image, axis=0)
  return preprocess_input(image)

transformer = KerasImageFileTransformer(inputCol="uri", outputCol="predictions",
                                        modelFile='model-full.h5',  # local file path for model
                                        imageLoader=loadAndPreprocessKerasInceptionV3,
                                        outputMode="vector")

files = [os.path.abspath(os.path.join("myimages/", f)) for f in os.listdir("myimages/") if f.endswith('.jpg')]
uri_df = sqlContext.createDataFrame(files, StringType()).toDF("uri")

keras_pred_df = transformer.transform(uri_df)