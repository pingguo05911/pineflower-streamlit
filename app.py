import streamlit as st
from PIL import Image

st.title("松花物候期识别系统")
st.write("基础系统运行正常！")

uploaded_file = st.file_uploader("上传图像", type=['jpg', 'png', 'jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    st.success("✅ 系统工作正常！")
