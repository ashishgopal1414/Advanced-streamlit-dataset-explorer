# Core Packages
import streamlit as st

# EDA Packages
import pandas as pd 
import numpy as numpy


# Data Viz Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import missingno as msno 


# Machine Learning Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sklearn


def main():
	""" Semi Supervised Machine Learning App with Streamlit """

	st.title("Semi-Auto Machine Learning Application")
	st.text("By IMRAN S M")

	activities = ["EDA","Plotting","Model Building","About"]

	choice = st.sidebar.selectbox("Select Activities",activities)


	if choice == "EDA":
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload Dataset : ",type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

            # Show Shape
			if st.checkbox("Show Shape"):
				st.write(df.shape)


			# Show Columns
			if st.checkbox("Show Columns"):
				all_columns = df.columns = df.columns.to_list()
				st.write(all_columns)



            # Show Summary
			if st.checkbox("Show Summary"):
				st.write(df.describe())




            # Show Value Counts
			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())



			# Show Select Columns To Show
			if st.checkbox("Select Columns to Show"):
				selected_columns = st.multiselect("Select Columns ",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			# Show Percentage of Missing Values
			if st.checkbox("Show Percentage of Missing Values"):
				st.write((df.isna().mean().round(4) * 100))





	if choice == "Plotting":
		st.subheader("Data Visualization")

		# File Select And Upload
		data = st.file_uploader("Upload Dataset : ",type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

		# Correlation with Seaborn
		if st.checkbox("Correlation With Seaborn"):
			st.write(sns.heatmap(df.corr()),annot=True)
			st.pyplot()


        # Pie Chart 
		if st.checkbox("Pie Chart"):
			all_columns = df.columns.to_list()
			columns_to_plot = st.selectbox("Select 1 Column ",all_columns)
			pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pie_plot)
			st.pyplot()


		# Missing Value Plot
		if st.checkbox("Plot Heatmap"):
			heat = msno.matrix(df)
			st.write(heat)
			st.pyplot()





	if choice == "Model Building":
		st.subheader("Model Selection And Accuracy")

		# File Select And Upload
		data = st.file_uploader("Upload Dataset : ",type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


		# Model Building
		X = df.iloc[:,0:-1]
		Y = df.iloc[:,-1]
		seed = 7


		# Model
		models = []
		models.append(("Logistic Regression",LogisticRegression()))
		models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis()))
		models.append(("K-Nearest Neighbors",KNeighborsClassifier()))
		models.append(("Classification And Regression Tree",DecisionTreeClassifier()))
		models.append(("Naive Bayes",GaussianNB()))
		models.append(("Support Vector Machine",SVC()))


		# Evaluate Model Accuracy


		# List
		model_names = []
		model_mean = []
		model_std = []
		all_models = []
		scoring = "Accuracy"

		for name, model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=seed)
			cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold)
			model_names.append(name)
			model_mean.append(cv_results.mean())
			model_std.append(cv_results.std())

			accuracy_results = {"model_name": name,"model_accuracy":cv_results.mean(), "standard_deviation":cv_results.std()}
			all_models.append(accuracy_results)

		if st.checkbox("Metrics as Table"):
			metDf = pd.DataFrame(zip(model_names,model_mean,model_std))
			metDf.columns = ["Model","Accuracy","Standard Deviation"]
			st.dataframe(metDf)


        # JSON Format
		if st.checkbox("Metric as JSON Format"):
			st.json(all_models)








	if choice == "About":
		st.sidebar.header("About App")
		st.sidebar.info(" Advanced Data Science Explorer Application ")
		st.title("")
		st.title("")

		

		st.sidebar.header("About Developer")
		st.sidebar.info("www.linkedin.com/in/imran-s-m")
		
		st.subheader("About Me")
		st.text("Name: IMRAN S M")
		st.markdown("LinkedIn: https://www.linkedin.com/in/imran-s-m")
		st.markdown("GitHub: https://github.com/immu0001")





















if __name__ == '__main__':
	main()
