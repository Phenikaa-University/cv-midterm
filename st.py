"""Create streamlit app for predict ditgit from image"""

import streamlit as st
from PIL import Image
import requests
import cv2

from inferences import Predict
from data.data_process import split_digit_from_img, makeContours

predict = Predict()

# Set title
st.title("Digit Recognizer")

# Upload image
st.write("Upload image")
image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Display image
if image is not None:
    img = Image.open(image)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    # Save image to path
    path = {
        "images": "data/test/digit_test.png",
        "lines": "data/lines/",
        "words": "data/words/"
    }
    img.save(path["images"])
    
    results = split_digit_from_img(path)

    # Predict image
    if st.button("Predict"):
        # Print predict result
        st.write("Predict result:")
        numbers = ""
        for img in results:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            img = cv2.resize(binary, (28, 28))
            res = predict(img)
            digit = int(res.argmax())
            numbers += str(digit)
        st.write(numbers)