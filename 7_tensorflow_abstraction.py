import tensorflow as tf
from tensorflow.contrib import learn

# model = learn.DNNRegressor()
# model.fit()
# model.evaluate()
# model.predict()

# -- 7.2.1 --
from sklearn import datasets, metrics, preprocessing
boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target

NUM_STEPS = 200
MINIBATCH_SIZE = 506

# some represent of data is instanced which named 'feature_columns'.
feature_columns = learn.infer_real_valued_columns_from_input(x_data)

# use estimator
reg = learn.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
    )
reg.fit(x_data, boston.target, steps=NUM_STEPS, batch_size=MINIBATCH_SIZE)

# return MSE loss value
MSE = reg.evaluate(x_data, boston.target, steps=1)

print(MSE)

