import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix, ConfusionMatrixDisplay
import pickle
def get_metrics(pipe):
    test_data=pd.read_csv("datasets/testing_data_rating.csv")
    st.write('Accuracy of this model ' + str(pipe.score(test_data["text"].to_list(),test_data["rating"].to_list())))
    st.write("Classification Report : ")
    report=classification_report(test_data["rating"].to_list(),pipe.predict(test_data["text"].to_list()),output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)
    cm=confusion_matrix(test_data["categories"].to_list(),pipe.predict(test_data["text"].to_list()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    st.pyplot(disp)
def app():
    ### this would give you the title on the browser tab
    st.set_page_config(page_title="Rating Demo")
    option = st.selectbox("Please pick the model you want to test.",('Linear Support Vector','Linear Support Vector with Oversampling'))
    if option == 'Linear Support Vector':
        with open('models/svcpipe.pickle', 'rb') as f:
            pipe = pickle.load(f)
    elif option == 'Linear Support Vector with Oversampling':
        with open('models/svc_oversample.pickle', 'rb') as f:
            pipe = pickle.load(f)
    st.header("Rating Prediction using" + option)
    review=st.text_input("Enter your review for the prediction")
    st.write("Your review is:",review)
    
    if st.button("Predict"):
        pred=pipe.predict([review])
        st.write("Your review is rated:",pred[0])
    
    get_metrics(pipe)

        
if __name__=='__main__':
    app()
