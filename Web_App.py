import numpy as np
import pickle
import pandas as pd
import streamlit as st

# loading saved model
loaded_model = pickle.load(open('F:/projects/Earth_Quake_Prediction_Web_App_using_streamlit/Model/trained_model.sav', 'rb'))

# creating a function for prediction
def earth_quake_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return {
        "Magnitude": prediction[0][0],
        "Longitude": prediction[0][1],
        "Latitude": prediction[0][2]
    }

def main():
    # giving a Title
    st.title('Earth Quake Prediction Web App')

    # getting the input data from user
    # Define a list of years
    years = list(range(1900, 2051))

    # Create the dropdown menu using selectbox
    selected_year = st.selectbox('Select a year', years)

    # Define a list of months
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Create the dropdown menu using selectbox
    selected_month = st.selectbox('Select a month', months)

    # Define a list of days
    days = list(range(1, 32))  # Assuming 1 to 31 for days

    # Create the dropdown menu using selectbox
    selected_day = st.selectbox('Select a day', days)

    # code for prediction
    prediction = ''

    # creating button for prediction
    if st.button('Result'):
        prediction = earth_quake_prediction([selected_year, selected_month, selected_day])

        # Displaying the prediction results
        st.success("Prediction:")
        st.write("Magnitude:", prediction["Magnitude"])
        st.write("Longitude:", prediction["Longitude"])
        st.write("Latitude:", prediction["Latitude"])

        # Rename 'Latitude' column to 'LAT'
        data = {
            "LAT": [prediction["Latitude"]],
            "LON": [prediction["Longitude"]],
        }
        df = pd.DataFrame(data)
        st.map(df, use_container_width=True, zoom=4)

if __name__ == '__main__':
    main()

