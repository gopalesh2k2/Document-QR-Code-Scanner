import streamlit as st
import utlis
import qrscan as qs
import numpy as np
from PIL import Image, ImageOps

st.title('Document Scanner/QR Code Scanner')
option = st.selectbox("What service you require ?", ("Document Scanner", "QR Code Scanner"))
if option == "Document Scanner":
    st.write('upload image')
    uploaded = st.file_uploader('Upload your image', type=("png", "jpg", "jpeg"))
    if uploaded is not None:
        img = Image.open(uploaded)
        st.write("Successfully Uploaded!!")
        img = ImageOps.exif_transpose(img)
        img = np.array(img)
        final, step = utlis.scanImage(img)
        st.image(step)
        st.image(final)
        if st.button("DOWNLOAD"):
            utlis.download(final)
            st.write("Downloaded Successfully!!")
elif option == "QR Code Scanner":
    cam = st.selectbox("You want to upload or use webCam", ("Upload", "webCam"))
    if cam == "Upload":
        st.write('upload image')
        uploaded = st.file_uploader('Upload your image', type=("png", "jpg", "jpeg"))
        if uploaded is not None:
            img = Image.open(uploaded)
            st.write("Successfully Uploaded!!")
            img = ImageOps.exif_transpose(img)
            img = np.array(img)
            final = qs.scanCode(False, img)
            st.write("out")
            st.image(final)
    elif cam == "webCam":
        final = qs.scanCode(True)






