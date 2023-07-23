import pickle
import numpy as np
import sklearn


#loading saved model
loaded_model = pickle.load(open('F:/projects/Earth quake model by streamlit/Model/trained_model.sav','rb'))
input_data= [1994,2,7]
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = loaded_model.predict(input_data_reshaped)
print("Magnitude:",prediction[0][0])
print("Longitude:",prediction[0][1])
print("Latitude:",prediction[0][2])
