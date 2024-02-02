#######################################################
## Creating app to check property values in SP       ##
#######################################################

# Importing packages
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import json

# Importing model
model = load("project_streamlit/data/model/model.joblib")

# Data path
data_path = "project_streamlit/data/sao-paulo-properties-april-2019.csv"

# Load data function
@st.cache_data
def load_data():

    data = pd.read_csv(data_path)
    
    return data

# Loading data
df = load_data()
df['District'] = df['District'].apply(lambda x: x.split('/')[0])
mean_latlon = df.loc[:, ["Latitude", "District", "Longitude"]].groupby("District").aggregate("mean")

# Unique district names
district_names = df.District.unique().tolist()

## Sidebar
st.sidebar.header("Calculate your own sale price/rent:")

param_type = st.sidebar.selectbox("Rent or Sale?", options=df["Negotiation Type"].unique().tolist(), format_func=str.capitalize)
param_dist = st.sidebar.selectbox("District", options=district_names)
param_lat = st.sidebar.number_input("Approximate latitude:", value=mean_latlon.loc[param_dist, "Latitude"], format="%.15f")
param_lon = st.sidebar.number_input("Approximate longitude:", value=mean_latlon.loc[param_dist, "Longitude"], format="%.15f")
param_condo = st.sidebar.number_input("Condo price", min_value=np.float64(0))
param_size = st.sidebar.number_input("Size m²:", min_value=0)
param_toilets = st.sidebar.number_input("Number of toilets:", min_value=0)
param_rooms = st.sidebar.number_input("Number of rooms:", min_value=1)
param_suites = st.sidebar.number_input("Number of suites:", min_value=0)
param_park = st.sidebar.number_input("Number of parking spaces:", min_value=0)
param_elev = st.sidebar.number_input("Number of elevators:", min_value=0)
param_furn = st.sidebar.toggle("Furnished?", value=False)
param_pool = st.sidebar.toggle("Pool available?", value=False)
param_new = st.sidebar.toggle("New?", value=False)

update_button = st.sidebar.button("Send values")

if update_button:
    # Column names
    regular_columns = df.columns.values.tolist()
    regular_columns.remove("District")
    regular_columns.remove("Price")
    negotiation_columns = [f"Negotiation Type_{x}" for x in df["Negotiation Type"].unique().tolist()]
    regular_columns.remove("Negotiation Type")
    regular_columns.remove("Property Type")
    district_columns = [f"District_{x}" for x in district_names]
    args_list = regular_columns + district_columns + negotiation_columns
    args_list.append("Property Type_apartment")

    # To dict
    args_dict = dict.fromkeys(args_list, 0)

    # Updating dict with selected values
    args_dict.update({"Condo": param_condo,
                      "Size": param_size,
                      "Rooms": param_rooms,
                      "Toilets": param_toilets,
                      "Suites": param_suites,
                      "Parking": param_park,
                      "Elevator": param_elev,
                      "Furnished": int(param_furn),
                      "Swimming Pool": int(param_pool),
                      "New": int(param_new),
                      "Latitude": param_lat,
                      "Longitude": param_lon,
                      f"District_{param_dist}": 1,
                      f"Negotiation Type_{param_type}": 1})

    args_array = np.asarray(list(args_dict.values())).reshape(1, -1)
    predict = model.predict(args_array)[0]
    update_button = False

## Main
st.title("Property Value Prediction - São Paulo/BR")
st.markdown(f"""
            This is an application to infer Apartment sale or rent prices in São Paulo - BR, by using machine learning. Simply enter your desired parameters to look for an apartment and predict the possible price of sale/rent. This app uses a dataset that covers apartment prices until 2019, so bear that in mind when comparing actual prices.
            """)
st.subheader("", divider='rainbow')

if st.toggle("View Dataframe", value=False):
    st.write(df)

try:
    if predict:
        st.subheader("", divider='rainbow')
        st.header(f"Predicted value: :red[{predict}]")
except:
    st.subheader("", divider='rainbow')
    st.header("No predicted value yet!")



