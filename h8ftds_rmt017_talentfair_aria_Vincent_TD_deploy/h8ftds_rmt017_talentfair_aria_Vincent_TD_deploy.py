import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import pickle

# Load All Files
with open('model_ran.pkl', 'rb') as file_1:
  model_ran = pickle.load(file_1)

with open('model_scaler.pkl', 'rb') as file_2:
  model_scaler = pickle.load(file_2)

with open('model_pca.pkl', 'rb') as file_3:
  model_pca = pickle.load(file_3)

def run():
    # membuat title
    st.title('PLANT NUTRITION PREDICTION')
    
    # tambah deskripsi
    st.write('Talent Fair 2023')
    st.write('Made by: Vincent Timothy Djaja')
    st.write('Batch: dsft-rmt-017')

    # tambah gambar
    image = Image.open('aria.jpg')
    st.image(image, caption='aria agriculture')

    # subheader
    st.subheader('EXPLORATORY DATA ANALYSIS')

    # Import dataframe
    data = pd.read_excel('aria_data.xlsx')

    # Histogram
    pilihan = st.selectbox('Silakan pilih kolom di bawah ini untuk lihat histogramnya:', 
    ('v1','v2','v3','v4','v5','v6','v7','v8'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data[pilihan],bins=30,kde=True)
    st.pyplot(fig)


    with st.form(key='form_parameters'):
      v1 = st.number_input('v1', min_value=0, max_value=10000, value=537)
      v2 = st.number_input('v2', min_value=0, max_value=10000, value=207)
      v3 = st.number_input('v3', min_value=0, max_value=10000, value=577)
      v4 = st.number_input('v4', min_value=0, max_value=10000, value=365)
      v5 = st.number_input('v5', min_value=0, max_value=10000, value=492)
      v6 = st.number_input('v6', min_value=0, max_value=10000, value=218)
      v7 = st.number_input('v7', min_value=0, max_value=10000, value=646)
      v8 = st.number_input('v8', min_value=0, max_value=10000, value=4274)
      st = st.selectbox('sample type', ('Lab 1' , 'Lab 2'), index=1)

      st.markdown('---')

      submitted = st.form_submit_button('Predict Target')
   
    df_inf = {
       'v1':v1, 
       'v2':v2,
       'v3':v3, 
       'v4':v4,
       'v5':v5, 
       'v6':v6,
       'v7':v7, 
       'v8':v8, 
       'sample_type' : st
    } 
    df_inf = pd.DataFrame([df_inf])
    
    if submitted:
            
      # Drop target
      df_inf.drop(['target'],axis=1,inplace=True)

      # Change lab 1 and lab 2
      df_inf['sample_type'].replace(['lab 1','lab 2'],[1,2],inplace=True)
      
      # Feature Scaling
      df_inf_scaled = model_scaler.transform(df_inf)
      df_inf_scaled
      
      # PCA
      df_pca = model_pca.transform(df_inf_scaled)
      df_pca

      # Predict inference
      y_pred_inf = model_ran.predict(df_pca)

      st.write('# Predict : ', str(int(y_pred_inf)))    

if __name__ == '__main__':
    run()