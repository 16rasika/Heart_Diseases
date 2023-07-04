import streamlit as st
import pandas as pd
import pickle
import sklearn
import seaborn as sns
import pandas as pd
import numpy as np 

from PIL import Image

#image = Image.open('heart_disease_image.jfif')

image = Image.open('Keep-Your-Heart-Healthy.png')

st.image(image,width=200)
#('images/dvd_store.jpg',use_column_width='always')
st.title ('Heart Diseases Prediction App')


#This app predicts if patient has a heart diseases

#Data obtined : (kaggle)https://www.kaggle.com/datasets/priyanka841/heart-disease-prediction-uci?select=heart.csv

st.header("Please,fill information to predict your heart condition'")
#st.header("Please, fill below information to predict your heart condition'")


#collects user input features

def user_input_features():
    age = st.number_input("Enter your Age:", step=1)
    st.write('You selected:', age)
    st.divider()
    #st.write('Male'=1, age)
    sex_string=st.radio('Pick your gender',['Male','Female'])##################### change converted 
    #sex_string=st.radio(...)
    if sex_string == 'Male':
        sex = 1
    else:
        sex = 0 
    #sex = st.selectbox('Select your Gender',(Male,Female))
    st.write('You selected:',sex_string)###########
    st.divider()
    
    cp_string = st.radio('Select Chest pain type',['Low','High','Medium','VeryLow'])#########
    cp_dict={"Low":1,"VeryLow":0, "High":3, "Medium":2}
    cp=cp_dict[cp_string] # this will access the value
    st.write('You selected:', cp_string)################
    st.divider()
    
    tres = st.number_input('Enter your Resting Blood Pressure:',step=1)
    st.write('You selected:', tres)
    st.divider()
    
    fbs_string = st.radio('Select Fasting Blood Sugar',['Low','High','Medium']) #################  change
    fbs_dict={"Low":0, "High":2, "Medium":1}
    fbs=fbs_dict[fbs_string]
    st.write('You selected:', fbs)
    st.divider()
    
    tha = st.number_input('Enter your Maximum heart rate achieved:',step=1)
    st.write('You selected:', tha)
    st.divider()
    
    thal_string=st.radio('Select your thal(Blood disorder) level',['Low','High','Medium'])######change
    thal_dict={"Low":0, "High":2, "Medium":1}
    thal=thal_dict[thal_string]

    st.write('You selected:', thal)
    
    heart={'age':age,
           'sex':sex,
           'cp':cp,
           'trestbps':tres,
           'fbs':fbs,
           'thalach':tha,
           'thal':thal
          }
    features=pd.DataFrame(heart,index=[0]) 
    st.button(":green[Submit]")
          
    return features,tres,thal,cp
input_df,tres,thal,cp = user_input_features()
#combines user input features with entire dataset
#heart_dataset = pd.read_csv('heart.csv')

def function_heart(tres,thal):
    if tres>=140:
      return (":orange[You might have Heart Disease]")
    elif thal>=1: 
      return(":red[Take Care! You have Heart Disease]")
    elif cp>=2:
      return(":red[Take Care! You have Heart Disease]")
    elif fbs== high:
      return(":red[Take Care! You have Heart Disease]")
      #image = Image.open('take_care.jpg')
     # st.image(image, use_column_width='always')
    else:    
       return(":green[According to model You are on low risk!]")

#st.write('The current best movie title is', title)
#st.write(Healthy_heart)
#st.title(":blue[Ursula's secret movie algorithm]") just for reference


url   = 'https://drive.google.com/file/d/1kvjnST4S2j_KOpyPO805RABKe-wF6nLL/view?usp=drive_link'# new heart csv heart_dataset
path  = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
heart_dataset =  pd.read_csv(path) # heart.csv


heart_dataset = heart_dataset.loc[:,['age','sex','cp','trestbps','fbs','thalach','thal','target']]


heart_dataset = heart_dataset.drop(columns='target',axis=1)

df = pd.concat([input_df,heart_dataset],axis=0)

# feature selection

#df = pd.get_dummies(df,columns= ['sex','cp','tresbps','fbs','thalach','thal'])

df = df[:1]  # for selecting one row from input features

#st.write(input_df)
#df = pd.get_dummies(df,columns= ['sex','cp','tresbps','fbs','thalach','thal'])


#loaded_model = pickle.load(open('models/trained_pipe_knn.sav', 'rb')) # from lms
#load_clf = pickle.load(open('Decision_Tree_Classifier.pickle','rb'))
load_clf = pickle.load(open('Random_forest_Classifier.pickle','rb'))  


#Apply model to make prediction
prediction =load_clf.predict(df)
prediction_proba =load_clf.predict_proba(df)

#Healthy_heart=function_heart(tres,thal)
#st.markdown(Healthy_heart)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probabilty')
st.write(prediction_proba)

if (prediction_proba[0,1] >= 0.2) & (prediction_proba[0,1] <=0.8):     
    st.write (":orange[You might have Heart Disease]")
elif prediction == 0:  
   st.write(":green[According to model You are on low risk!]")
else:  
   st.write(":red[Take Care! You have Heart Disease]") 

