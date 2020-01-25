import pandas as pd

def checktype(df : pd.DataFrame):

	values = df.dtypes
	typeDic = values.to_dict()
	numDic = dict(filter(lambda elem: elem[1] == "float64" or elem[1] == "int64", typeDic.items()))
	print(numDic)

data = pd.read_csv("wine_data.csv")
dhead = data.head(10)
checktype(dhead)