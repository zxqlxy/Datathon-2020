import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import random

st.sidebar.title("File options")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

st.title("Data Visualization 101")
st.markdown(
        """## Important Notes

This app requires the Awesome Streamlit package.
The Awesome Streamlit package can be installed using

`pip install awesome-streamlit`
""")

st.sidebar.button("Show graph")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
	    data = pd.read_csv(uploaded_file)
	    st.write(data)

st.write(data.describe())

def checktype(df : pd.DataFrame):

	values = df.dtypes
	typeDic = values.to_dict()
	numDic = dict(filter(lambda elem: elem[1] == "float64" or elem[1] == "int64", typeDic.items()))
	return list(numDic.keys())
dhead = data.head(10)
columns = checktype(dhead)
# for col in data.columns: 
#     print(col) 
#     st.button(col)

options = st.multiselect(
	     'Choose two parameter to plot',
         # ('Yellow', 'Red')
	     columns)
# st.write('You selected:', options)


def plot2(x,y):
	plt.plot(x,y)
	st.pyplot()

def plot2(x,y):
	plt.plot(x,y, '.')
	st.pyplot()


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
if len(options) == 2:
	plot2(data[options[0]], data[options[1]])
elif len(options) == 3:
	plot3(data[options[0]], data[options[1]], data[options[2]])


num_data = data[columns]
def plot_PCA(option='standard'):
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

plot_PCA();

def plot_correlation():
	f = plt.figure(figsize=(19, 15))
	plt.matshow(data.corr(), fignum=f.number)
	plt.xticks(range(len(columns)), columns, fontsize=14, rotation=45)
	plt.yticks(range(len(columns)), columns, fontsize=14)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=14)
	plt.title('Correlation Matrix', fontsize=16)
	st.pyplot()
	plt.clf()


plot_correlation()
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