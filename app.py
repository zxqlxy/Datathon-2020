import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from keras.callbacks import ModelCheckpoint, History
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
import random
# import tensorflow as tf
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Flatten, Conv2D

import warnings

#data = pd.read_csv('training.csv')


# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10, activation='softmax')
#
#   def call(self, x):
#     x = self.d1(x)
#     return self.d2(x)
#
# # Create an instance of the model
# model = MyModel()

def main():
    options = pd.DataFrame()
    data = np.array([])

    st.sidebar.title("File options")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

    st.title("Data Visualization 101")
    st.markdown(
        """
    Our demo will be running on [Chevron's dataset](https://datathon.rice.edu/static/chevronChallenge.zip) as default. You can add 
    csv file as you want.

    """
    )

    if uploaded_file is not None:

        st.header("Data Exploration")
        st.markdown(
            """
            If you choose one or more parameters to explore, it will generate default summary. If you choose two parameters,
            it will plot graph for the first parameter over the second parameter.
            """)
        agree = st.checkbox("show raw data")
        if agree:
            st.write(data)

        dhead = data.head(10)
        columns = checktype(dhead)

        options = st.sidebar.multiselect(
            'Choose parameters to explore',
            data.columns)

        if options:
            st.write(data[options].describe())

        cor_btn = st.sidebar.checkbox('Show Correlation Plot')
        if cor_btn:
            plot_correlation(data, columns)

    if len(options) == 2 and not (isinstance(data[options[0]][1], str) and isinstance(data[options[0]][1], str)):
        linear = st.checkbox('Plot Linear Regression')
        graphbtn = st.sidebar.button("Show graph")
        st.markdown(
            """
        The graph below is the scatter point graph of variable a vs variable b, if desired, you can plot the 
        regression line on the plot by checking the box above

        """
        )
        plt.plot(data[options[0]], data[options[1]], '.')
        if linear:
            X = data[options[0]].values.reshape(-1, 1)  # values converts it into a numpy array
            Y = data[options[1]].values.reshape(-1, 1)
            linear_regressor = LinearRegression()  # create object for the class
            linear_regressor.fit(X, Y)  # perform linear regression
            Y_pred = linear_regressor.predict(X)  # make predictions
            plt.plot(X, Y_pred, color='red')
        plt.title(options[0] + ' vs ' + options[1])
        plt.xlabel(options[0])
        plt.ylabel(options[1])
        st.pyplot()
        plt.clf()



    if len(data)>0:
	    st.sidebar.title('Nerual Nets')

	    test_train_split_slider = st.sidebar.slider('validation_split', 0.0, 1.0, 0.01, 0.2)
	    training = st.sidebar.multiselect(
	        'Choose training to explore',
	        columns)
	    target = st.sidebar.selectbox(
	        'Choose a target',
	        columns)
	    number = st.number_input('Training epoch')
	    trainbtn = st.sidebar.button("Train model")

	    if trainbtn:
	        neural_nets(model, data, training, target)


def checktype(df: pd.DataFrame):
    values = df.dtypes
    typeDic = values.to_dict()
    numDic = dict(filter(lambda elem: elem[1] == "float64" or elem[1] == "int64", typeDic.items())) or elem[1] == "object"
    return list(numDic.keys())

def checktype_object(df: pd.DataFrame):
    values = df.dtypes
    typeDic = values.to_dict()
    numDic = dict(filter(lambda elem: elem[1] == "object", typeDic.items()))
    return list(numDic.keys())


@st.cache(suppress_st_warning=True)
def plot2(x, y, linear):
    if linear:
        X = x.values.reshape(-1, 1)  # values converts it into a numpy array
        Y = y.values.reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        plt.plot(X, Y_pred, color='red')
    st.write("no")
    plt.plot(x, y, '.')
    st.pyplot()
    plt.clf()


def plot3(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for i in range(x):
        ax.scatter(x[i], y[i], z[i], marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


def plot_PCA(num_data, option='standard'):
    pca = decomposition.PCA()
    if option == 'standard':
        standard = StandardScaler()
        standard.fit(num_data)  # Standardize the data
        stand = standard
        processed = standard.transform(num_data)
        pca.fit(num_data)
        newdata = pca.transform(num_data)
    elif option == 'normal':
        pca.fit(num_data)
        newdata = pca.transform(num_data)
    # print(pca.get_covariance())
    plt.matshow(pca.components_, cmap='viridis')
    # plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=65, ha='left')
    plt.tight_layout()
    st.pyplot()
    plt.clf()


# plot_PCA();
@st.cache(suppress_st_warning=True)
def plot_correlation(data, columns):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(len(columns)), columns, fontsize=14, rotation=45)
    plt.yticks(range(len(columns)), columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    st.pyplot()


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


def corelation_coefficient(data):
    df = pd.DataFrame([data[options[0]], data[options[1]]], columns=options)
    df.corr(method=histogram_intersection)




# @tf.function
#
# def neural_nets(model, data, training, target):
    # training = data[data.columns[5]]
    # target = data[data.columns[5]]
    #
    train = data[training]
    target = data[target]

    # model = Sequential()
    #
    # # The Input Layer :
    # model.add(Dense(128, kernel_initializer='normal', input_dim=train.shape[1], activation='relu'))
    #
    # # The Hidden Layers :
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    #
    # # The Output Layer :
    # model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    #
    # # Compile the network :
    # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    #
    # # model = NN_model()
    #
    # model.summary()


    # checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    # checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # history = History()
    # callbacks_list = [checkpoint, history]
    # NN_model.fit(train, target, epochs=20, batch_size=32, validation_split=0.2, callbacks=callbacks_list)
    # print(History)
    #
    # train_ds = tf.data.Dataset.from_tensor_slices(
    #     (train, target)).shuffle(10000).batch(32)
    #
    # test_ds = tf.data.Dataset.from_tensor_slices((train, target)).batch(32)
    #
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    #
    # optimizer = tf.keras.optimizers.Adam()
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    #
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    #
    # @tf.function
    # def train_step(model, images, labels):
    #     with tf.GradientTape() as tape:
    #         # training=True is only needed if there are layers with different
    #         # behavior during training versus inference (e.g. Dropout).
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)
    #
    # @tf.function
    # def test_step(model, images, labels):
    #     # training=False is only needed if there are layers with different
    #     # behavior during training versus inference (e.g. Dropout).
    #     predictions = model(images, training=False)
    #     t_loss = loss_object(labels, predictions)
    #
    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)
    #
    # EPOCHS =5
    # for epoch in range(EPOCHS):
    #   # Reset the metrics at the start of the next epoch
    #   train_loss.reset_states()
    #   train_accuracy.reset_states()
    #   test_loss.reset_states()
    #   test_accuracy.reset_states()
    #
    #   for images, labels in train_ds:
    #       print(images, labels)
    #       train_step(model, images, labels)
    #
    #   for test_images, test_labels in train_ds:
    #     test_step(model, test_images, test_labels)
    #
    #   template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    #   print(template.format(epoch+1,
    #                         train_loss.result(),
    #                         train_accuracy.result()*100,
    #                         test_loss.result(),
    #                         test_accuracy.result()*100))
# corelation_coefficient(data)

# progress_bar = st.progress(0)
# status_text = st.empty()
# chart = st.line_chart(np.random.randn(10, 2))

# for i in range(100):
#     # Update progress bar.
#     progress_bar.progress(i)

#     new_rows = np.random.randn(10, 2)

#     # Update status text.
#     status_text.text(
#         'The latest random number is: %s' % new_rows[-1, 1])

#     # Append data to the chart.
#     chart.add_rows(new_rows)

#     # Pretend we're doing some computation that takes time.
#     time.sleep(0.1)

# status_text.text('Done!')
# st.balloons()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
