import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

st.write(data.describe())

# for col in data.columns: 
#     print(col) 
#     st.button(col)

options = st.multiselect(
    'Choose two parameter to plot',
    # ('Yellow', 'Red')
    data.columns)


# st.write('You selected:', options)

def plot2(x, y):
    plt.plot(x, y)
    st.pyplot()


def plot2(x, y):
    plt.plot(x, y, '.')
    st.pyplot()


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


if len(options) == 2:
    plot2(data[options[0]], data[options[1]])
elif len(options) == 3:
    plot3(data[options[0]], data[options[1]], data[options[2]])


def plot_PCA(option='standard'):
    pca = decomposition.PCA()
    processed = process_all(0, 2000)
    if option == 'standard':
        standard = StandardScaler()
        standard.fit(processed)  # Standardize the data
        stand = standard
        processed = standard.transform(processed)
        pca.fit(processed)
        newdata = pca.transform(processed)
    elif option == 'normal':
        pca.fit(processed)
        newdata = pca.transform(processed)
    # print(pca.get_covariance())
    plt.matshow(pca.components_, cmap='viridis')
    # plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
    plt.colorbar()
    plt.xticks(range(6), ['time', 'bins', 'element', 'Amplitude', 'Doppler', 'FWHM'], rotation=65, ha='left')
    plt.tight_layout()
    st.pyplot()
    plt.clf()


def plot_Reg(data):
    X = data[options[0]].values.reshape(1, -1)  # values converts it into a numpy array
    Y = data[options[1]].values.reshape(1, -1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    st.pyplot()


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


def corelation_coefficient(data):
    df = pd.DataFrame([data[options[0]], data[options[1]]], columns=options)
    df.corr(method=histogram_intersection)


plot_Reg(data)
corelation_coefficient(data)
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
