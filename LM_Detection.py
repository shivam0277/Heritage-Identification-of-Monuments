# import streamlit as st
# import PIL
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import pandas as pd
# from geopy.geocoders import Nominatim

# model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
# # model_url = 'on_device_vision_classifier_landmarks_classifier_asia_V1_1'

# # label_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
# labels = 'landmarks_classifier_asia_V1_label_map.csv'
# df = pd.read_csv(labels)
# labels = dict(zip(df.id, df.name))

# def image_processing(image):
#     img_shape = (321, 321)
#     classifier = tf.keras.Sequential(
#         [hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")])
#     img = PIL.Image.open(image)
#     img = img.resize(img_shape)
#     img1 = img
#     img = np.array(img) / 255.0
#     img = img[np.newaxis]
#     result = classifier.predict(img)
#     return labels[np.argmax(result)],img1

# def get_map(loc):
#     geolocator = Nominatim(user_agent="Your_Name")
#     location = geolocator.geocode(loc)
#     return location.address,location.latitude, location.longitude

# def run():
#     # st.title("Landmark Recognition")
#     st.title("Heritage Identification of Monuments using Deep Learning")
#     img = PIL.Image.open('logo.png')
    
#     img = img.resize((256,256))
#     st.image(img)
#     img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
#     if img_file is not None:
#         save_image_path = './Uploaded_Images/' + img_file.name
#         with open(save_image_path, "wb") as f:
#             f.write(img_file.getbuffer())
#         prediction,image = image_processing(save_image_path)
#         st.image(image)
#         st.header("📍 **Predicted Landmark is: " + prediction + '**')
#         try:
#             address, latitude, longitude = get_map(prediction)
#             st.success('Address: '+address )
#             loc_dict = {'Latitude':latitude,'Longitude':longitude}
#             st.subheader('✅ **Latitude & Longitude of '+prediction+'**')
#             st.json(loc_dict)
#             data = [[latitude,longitude]]
#             df = pd.DataFrame(data, columns=['lat', 'lon'])
#             st.subheader('✅ **'+prediction +' on the Map**'+'🗺️')
#             st.map(df)
#         except Exception as e:
#             st.warning("No address found!!")
# run()

import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
labels_file = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels_file)
labels = dict(zip(df.id, df.name))

def classify_landmark(image_path):
    img = PIL.Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]

    # Load the model
    module = hub.load(model_url)
    input_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    predictions = module(input_tensor)

    predicted_label_id = np.argmax(predictions[0])
    predicted_label = labels[predicted_label_id]

    return predicted_label, predictions[0][predicted_label_id]

def get_location(landmark_name):
    geolocator = Nominatim(user_agent="HeritageApp")
    location = geolocator.geocode(landmark_name)
    return location.address, location.latitude, location.longitude

def main():
    st.title("Heritage Monuments Identification")
    st.markdown("Upload an image of a monument to identify and explore its heritage.")
    
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        with st.spinner('Analyzing...'):
            predicted_label, confidence = classify_landmark(uploaded_file)
        
        st.success(f"Predicted Landmark: {predicted_label} (Confidence: {confidence:.2f})")
        
        try:
            address, latitude, longitude = get_location(predicted_label)
            st.subheader("Location Details")
            st.write(f"Address: {address}")
            st.write(f"Latitude: {latitude}, Longitude: {longitude}")
            
            st.subheader("Location on Map")
            st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
        except Exception as e:
            st.warning("Location details not available.")
    
if __name__ == "__main__":
    main()
