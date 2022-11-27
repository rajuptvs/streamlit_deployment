import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
import pickle

def get_metrics():
    st.write("Testing out definition")
def app():
    label_dict={0:"Active Life", 1:"Automotive",2:"Beauty & Spas",3:"Restaurants",4:"Shopping"}
    st.set_page_config(page_title="Tagging Demo")
    option = st.selectbox("Please pick the model you want to test.",('Linear Support Vector','Naive Bayes','SVM with SGD'))
    if option == 'Linear Support Vector':
        with open('models/tagging_model.pickle', 'rb') as f:
            pipe = pickle.load(f)
    elif option == 'Naive Bayes':
        with open('models/tagging_model_nb.pickle', 'rb') as f:
            pipe = pickle.load(f)
    elif option == 'SVM with SGD':
        with open('models/tagging_model_sgd.pickle', 'rb') as f:
            pipe = pickle.load(f)
    with open('models/tagging_model.pickle', 'rb') as f:
        pipe = pickle.load(f)
    st.header("Tagging Prediction using "+option)
    review=st.text_input("Enter your review for the prediction")
    st.write("Your review is:",review)
    
    get_metrics()

    if st.button("Predict"):
        pred=pipe.predict([review])
        st.write("Your review can be tagged as :",label_dict.get(pred[0]))
        
if __name__=='__main__':
    app()
