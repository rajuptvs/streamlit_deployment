import streamlit as st
from multiapp import MultiApp
from apps import rating_model,tagging_model

app = MultiApp()


# Add all your application here
app.add_app("Rating Prediction", rating_model.app)
app.add_app("Tagging Prediction", tagging_model.app)



# The main app
app.run()