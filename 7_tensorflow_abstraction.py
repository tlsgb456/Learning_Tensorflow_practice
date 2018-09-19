import tensorflow as tf
from tensorflow.contrib import learn

model = learn.DNNRegressor()
model.fit()
model.evaluate()
model.predict()
