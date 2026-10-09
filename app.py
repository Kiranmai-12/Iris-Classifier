import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
st.title("Iris Flower Classifier")
d=load_iris(); model=RandomForestClassifier(random_state=42).fit(d.data,d.target)
vals=[st.number_input(label,0.1,10.0,default) for label,default in [("Sepal length cm",5.1),("Sepal width cm",3.5),("Petal length cm",1.4),("Petal width cm",0.2)]]
if st.button("Predict"): st.success(d.target_names[model.predict([vals])[0]].title())
