import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

#title
st.title("Inson, Gullar va Mevalarni aniqlab beruvchi model")

#rasmni yuklash
file = st.file_uploader('Rasmni yuklash', type=['png', 'jpg', 'jpeg', 'svg', 'gif'])
if file:
    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('person_nature_model.pkl')

    #UZB function
    def nom(name):
        if name=='Person':
            return 'Inson'
        elif name=='Fruit':
            return 'Meva'
        else:
            return 'Gul'

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Aniqlangan obyekt: {pred}")
    st.info(f"Aniqlik darajasi: {probs[pred_id]*100:.1f}%")
    st.image(file)

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)