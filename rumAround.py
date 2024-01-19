import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from helpers import heatMap, formatFlavorProfile, makeScatter, makeHist, makeBar
import pickle
import sklearn
import plotly.express as px
from wordcloud import WordCloud
from PIL import Image
import os

# set page layout
st.set_page_config(
        page_title="Rum Around!",
        page_icon="🥃",
        layout="wide",
    )

# set title
with st.container():
    with st.columns(3)[1]:
        st.title('Rum Around the World!')

# define tabs
tab1, tab2, tab3 = st.tabs(["Exploring the Data Set", "Flavor Heat Maps", 'Predictive Models'])

# load rum data set
rum = pd.read_csv('Data/rum_12_2023.csv')

# load flavors
flavors = pd.read_csv("Data/flavors.csv")

# load word cloud data
wordcloudDF = pd.read_csv('Data/taste_counts.csv')
d = {taste:count for taste, count in zip(wordcloudDF['Taste'], wordcloudDF['Count'])}

# load models
model_rating = pickle.load(open('Models/taste_to_score.sav', 'rb'))
model_country = pickle.load(open('Models/taste_to_country.sav', 'rb'))

with tab1:
    st.write(" ")
    with st.container():
        colA, colB, colC, colD = st.columns(4)
        colA.markdown(f"<h4 style='text-align: center'>Unique Rums in the Data Set: {len(rum):,}</h4>", unsafe_allow_html=True)
        #colA.plotly_chart(makeHist(rum, 'Rating'), use_container_width=True)
        colB.markdown(f"<h4 style='text-align: center'>Number of Reviews: {int(sum(rum['Number_Reviews'].dropna())):,}</h4>", unsafe_allow_html=True)
        #colB.plotly_chart(makeHist(rum, 'Rating'), use_container_width=True)
        colC.markdown(f"<h4 style='text-align: center'>Unique Countries/Territories: {len(rum['Country'].value_counts().index)}</h4>", unsafe_allow_html=True)
        #colC.plotly_chart(makeHist(rum, 'Rating'), use_container_width=True)
        colD.markdown(f"<h4 style='text-align: center'>Unique Tasting Notes: {224}</h4>", unsafe_allow_html=True)
    st.write(" ")
    with st.container():
        colX, colY, colZ = st.columns([2,3,3])
        with colX:
            _, col5, _ = st.columns([1,5,1])
            col5.image('Img/wc.png')

        with colY:
            colY2, _ = st.columns([12,1])
            colY2.write("##")
            colY2.write("##")
            
                    
            colY2.plotly_chart(makeHist(rum, 'Rating'), use_container_width=True)
        with colZ:
            colZ2, _ = st.columns([12,1])
            colZ2.write("##")
            colZ2.write("##")
            colZ2.write("##")
            colZ2.plotly_chart(makeBar(rum), use_container_width=True)
    
with tab2:
    st.write("Pick a Country to see it's Flavor Profile Compares to that of all Rums in the Database!")

    # drop down options
    min_obs = 150
    drop1_options = sorted(list(rum['Country'].value_counts().index[rum['Country'].value_counts().values >= min_obs]))
    drop1_options.remove('unknown')
    option1 = st.selectbox(
    ' ',
    [c.title() for c in drop1_options],
    index = None
    )

    if option1 != None:
        st.pyplot(heatMap(country=option1.lower()))

with tab3:
    st.write("Create a Flavor Profile for your Rum by Selecting 3 to 8 Flavors.")
    st.write("Country is predicted using a Categorical Naive Bayes Classifier and rating is predicted using a Lasso Regressor.")
    
    flavor_profile = st.multiselect(

    ' ',

    sorted(list(flavors['0'].values)),

    max_selections = 8)

    if len(flavor_profile) >= 3:
        predicted_country = model_country.predict(formatFlavorProfile(flavor_profile, flavors['0'].values))[0].title()
        st.write("#")
        st.write(f'The Predicted Country for a Rum with this Flavor Profile: {predicted_country}')
        st.write(f'The Predicted Rating for a Rum with this Flavor Profile: {round(model_rating.predict(formatFlavorProfile(flavor_profile, flavors["0"].values))[0], 1)}/10')
        st.write("#")

        with st.container():
            col1, col2 = st.columns(2)
            df = rum[rum['Country'] == predicted_country.lower()].sort_values(by=['Number_Reviews'], ascending=False)[['Name','Distillery', 'Age', 'Price', 'Rating', 'Number_Reviews']].reset_index(drop=True)
            
            col1.markdown(f"<h6 style='text-align: center'>Top 25 Rums from {predicted_country} sorted by Reviews</h6>", unsafe_allow_html=True)
            col1.write(df.head(25))
            col2.write(makeScatter(df, predicted_country.title())) 
                

        

        

