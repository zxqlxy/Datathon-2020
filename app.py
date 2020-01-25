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
import random

import warnings


def main():
	st.sidebar.title("File options")
	uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
	if uploaded_file is not None:
		data = pd.read_csv(uploaded_file)
		agree = st.checkbox("show raw data")
		if agree:
			st.write(data)

	st.title("Data Visualization 101")
	st.markdown(
	        """## Important Notes

	We will run a demo that uses the Chevron Dataset to train and test a machine learning model.

	""")

	if uploaded_file is not None:

		st.write(data.describe())
		dhead = data.head(10)
		columns = checktype(dhead)
		# for col in data.columns: 
		#     print(col) 
		#     st.button(col)

		options = st.sidebar.multiselect(
			     'Choose two parameter to plot',
		         # ('Yellow', 'Red')
			     columns)
		# st.write('You selected:', options)
		linear = st.checkbox('Plot Linear Regression')
		st.sidebar.button("Show graph")

		if len(options) == 2:
			if linear:
				plot2(data[options[0]], data[options[1]], linear)
			else:
				st.write("not linear")
				plt.clf()
				plt.plot(data[options[0]], data[options[1]], '.')
				st.pyplot()
				st.write("plz plot")

		elif len(options) == 3:
			plot3(data[options[0]], data[options[1]], data[options[2]])

		# if linear:
		# 	plot2(data[options[0]], data[options[1]], linear)
		#
		# else:

		num_data = data[columns]

		# plot_Reg(num_data, options)
		plot_correlation(data, columns)


		test_train_split_slider = st.sidebar.slider('training data %', 0, len(data), math.floor(0.2 * len(data)))
		trainbtn = st.sidebar.button("Train model")



def checktype(df : pd.DataFrame):

	values = df.dtypes
	typeDic = values.to_dict()
	numDic = dict(filter(lambda elem: elem[1] == "float64" or elem[1] == "int64", typeDic.items()))
	return list(numDic.keys())

@st.cache(suppress_st_warning=True)
def plot2(x,y, linear):
	if linear:
		X = x.values.reshape(-1, 1)  # values converts it into a numpy array
		Y = y.values.reshape(-1, 1)
		linear_regressor = LinearRegression()  # create object for the class
		linear_regressor.fit(X, Y)  # perform linear regression
		Y_pred = linear_regressor.predict(X)  # make predictions
		plt.plot(X, Y_pred, color='red')
	st.write("no")
	plt.plot(x,y, '.')
	st.pyplot()
	plt.clf()


def plot3(x, y,z):
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
	plt.title('Correlation Matrix', fontsize=16)
	st.pyplot()
	# plt.clf()



def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


def corelation_coefficient(data):
    df = pd.DataFrame([data[options[0]], data[options[1]]], columns=options)
    df.corr(method=histogram_intersection)


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