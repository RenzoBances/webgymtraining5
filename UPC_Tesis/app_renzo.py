import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import base64
from random import randrange
import pandas as pd
import main
import json
import Exercises.Curls
import Exercises.Squats
import Exercises.Extensions
import Exercises.Crunches
import Exercises.Rows
import Exercises.BenchPress
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(
    page_title="STARTER TRAINING -UPC",
    page_icon ="img/upc_logo.png",
)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_upc=get_base64_of_bin_file('img/upc_logo_50x50.png')
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 336px;        
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 336px;
        margin-left: -336px;
        background-color: #00F;
    }}
    [data-testid="stVerticalBlock"] {{
        flex: 0;
        gap: 0;
    }}
    #starter-training{{
        padding: 0;
    }}
    #div-upc{{
        #border: 1px solid #DDDDDD;
        background-image: url("data:image/png;base64,{img_upc}");
        position: fixed !important;
        right: 14px;
        bottom: 14px;
        width: 50px;
        height: 50px;
        background-size: 50px 50px;
    }}
    .css-10trblm{{
        color: #FFF;
        font-size: 40px;
        font-family: 'PROGRESS PERSONAL USE';
        src: url(fonts/ProgressPersonalUse-EaJdz.ttf);       
    }}
    #.main {{
        background: linear-gradient(135deg,#a8e73d,#09e7db,#092de7);
        background-size: 180% 180%;
        animation: gradient-animation 3s ease infinite;
        }}
        @keyframes gradient-animation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    .block-container{{
        max-width: 100%;
    }}
    </style>
    """ ,
    unsafe_allow_html=True,
)

st.title('STARTER TRAINING')
st.sidebar.markdown('---')
st.markdown("<div id='div-upc'></span>", unsafe_allow_html=True)
st.sidebar.title('The Training App')
st.sidebar.markdown('---')

app_mode = st.sidebar.selectbox('Choose your training:',
    ['(home)','Glute Bridge','Abs', 'Lunges', 'Push Up', 'Squats']
)

if app_mode =='(home)':
    a=0
elif app_mode =='Squats':
        st.markdown('### CORE Basic Squats')
        st.markdown("<hr/>", unsafe_allow_html=True)
        exercise_to_do = {}
        user_input_rep = st.text_input("Please enter rep amount: " +app_mode)
        user_input_sets = st.text_input("Please enter set amount: " + app_mode)
        exercise_to_do[app_mode] = {"reps":user_input_rep,"sets":user_input_sets}
        options = st.button("Click me to begin.")
        if options:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown('## Are you ready?')
            with st.spinner('Starting in 5 seconds...'):
                time.sleep(5)
                #st.success('Done!')
            #st.write(exercise_to_do)
            #webcam = st.checkbox('Start Webcam')
            trainer, user = st.columns(2)
            with trainer:        
                st.markdown("Expert", unsafe_allow_html=True)
                video_trainer_file="videos_experto/squats.mp4"
                st.video(video_trainer_file, format="video/mp4", start_time=0)
            with user:
                    main.start(exercise_to_do)
                    app_mode = '(home)'

elif app_mode =='Abs':
        app_mode ='Crunches'
        st.markdown('### CORE Basic Abs')
        st.markdown("<hr/>", unsafe_allow_html=True)
        exercise_to_do = {}
        user_input_rep = st.text_input("Please enter rep amount: " +app_mode)
        user_input_sets = st.text_input("Please enter set amount: " + app_mode)
        exercise_to_do[app_mode] = {"reps":user_input_rep,"sets":user_input_sets}
        options = st.button("Click me to begin.")
        if options:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown('## Are you ready?')
            with st.spinner('Starting in 5 seconds...'):
                time.sleep(5)
                #st.success('Done!')
            #st.write(exercise_to_do)
            #webcam = st.checkbox('Start Webcam')
            trainer, user = st.columns(2)
            with trainer:        
                st.markdown("Expert", unsafe_allow_html=True)
                video_trainer_file="videos_experto/crunches.mp4"
                st.video(video_trainer_file, format="video/mp4", start_time=0)
            with user:
                    main.start(exercise_to_do)

elif app_mode =='Lunges':
    a=0
elif app_mode =='Push Up':
        app_mode ='Extensions'
        st.markdown('### Tronco Superior Basic Push Up')
        st.markdown("<hr/>", unsafe_allow_html=True)
        exercise_to_do = {}
        user_input_rep = st.text_input("Please enter rep amount: " +app_mode)
        user_input_sets = st.text_input("Please enter set amount: " + app_mode)
        exercise_to_do[app_mode] = {"reps":user_input_rep,"sets":user_input_sets}
        options = st.button("Click me to begin.")
        if options:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown('## Are you ready?')
            with st.spinner('Starting in 5 seconds...'):
                time.sleep(5)
                #st.success('Done!')
            #st.write(exercise_to_do)
            #webcam = st.checkbox('Start Webcam')
            trainer, user = st.columns(2)
            with trainer:        
                st.markdown("Expert", unsafe_allow_html=True)
                video_trainer_file="videos_experto/Extensions.mp4"
                st.video(video_trainer_file, format="video/mp4", start_time=0)
            with user:
                    main.start(exercise_to_do)
elif app_mode =='Glute Bridge':
    a=0
else:
    a=0