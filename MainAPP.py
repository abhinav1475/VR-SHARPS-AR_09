import streamlit as st
import gettext
import os
import tensorflow as tf
import numpy as np

# Set up translation directories and files
locales_dir = 'locale'
button_labels = {
    'en': {
        'home': 'Home',
        'about': 'About',
        'disease_recognition': 'Disease Recognition',
        'predict': 'Predict',
        'show_image': 'Show Image',
        'dashboard': 'Dashboard',
        'select_page': 'Select Page',
        'welcome': 'Welcome to the AI-Driven Crop Disease Management System! 🌱💻',
        'about_project': 'About Project',
        'about_dataset': 'About Dataset',
        'plant_disease_recognition': 'PLANT DISEASE RECOGNITION SYSTEM',
        'how_it_works': """
            ### How It Works
            1. **Upload Data**: Go to the **Disease Management** page and upload an image of your crop along with environmental data.
            2. **Analysis**: Our AI-driven system will process the information using sophisticated algorithms to identify disease symptoms and assess contributing factors.
            3. **Results**: View predictions and receive tailored recommendations for disease management.
        """,
        'our_prediction': 'Our Prediction',
    },
    'te': {
        'home': 'హోమ్',
        'about': 'గురించి',
        'disease_recognition': 'వైరస్ గుర్తింపు',
        'predict': 'అనుమానించండి',
        'show_image': 'చిత్రాన్ని చూపించు',
        'dashboard': 'డాష్‌బోర్డ్',
        'select_page': 'పేజీని ఎంచుకోండి',
        'welcome': 'AI ఆధారిత పంట వ్యాధి నిర్వహణ వ్యవస్థకు స్వాగతం! 🌱💻',
        'about_project': 'ప్రాజెక్ట్ గురించి',
        'about_dataset': 'డేటా సెట్ గురించి',
        'plant_disease_recognition': 'పంట వ్యాధి గుర్తింపు వ్యవస్థ',
        'how_it_works': """
            ### ఇది ఎలా పనిచేస్తుంది
            1. **డేటాను అప్‌లోడ్ చేయండి**: **వ్యాధి నిర్వహణ** పేజీకి వెళ్లి మీ పంట యొక్క చిత్రం మరియు పర్యావరణ డేటాను అప్‌లోడ్ చేయండి.
            2. **విశ్లేషణ**: మా AI-ఆధారిత సిస్టమ్ సొగసైన ఆల్గోరిథమ్స్‌ని ఉపయోగించి వ్యాధి లక్షణాలను గుర్తించడానికి మరియు సంబంధిత కారకాల్ని అంచనా వేస్తుంది.
            3. **ఫలితాలు**: ఫలితాలను చూడండి మరియు వ్యాధి నిర్వహణ కోసం ప్రత్యేక సిఫార్సులను స్వీకరించండి.
        """,
        'our_prediction': 'మా అంచనా',
    }
}

# Load Translation Files
def load_translation(language):
    try:
        lang = gettext.translation('messages', localedir=locales_dir, languages=[language])
        lang.install()
        return lang.gettext  # Returns translation function
    except FileNotFoundError:
        return lambda x: x  # Fallback to English if translation not found

# Step 1: Language Selection Page
st.session_state.setdefault("language_selected", False)
if not st.session_state.language_selected:
    st.title("Select Language")
    language_choice = st.radio("Choose Language:", ["English", "Telugu"])
    language_code = 'en' if language_choice == "English" else 'te'
    
    if st.button("Continue"):
        st.session_state.language_selected = True
        st.session_state.language = language_code
        _ = load_translation(language_code)

# Step 2: Main App after Language Selection
else:
    language_code = st.session_state.language
    _ = load_translation(language_code)

    # Sidebar and Main Page Content
    st.sidebar.title(button_labels[language_code]['dashboard'])
    app_mode = st.sidebar.selectbox(button_labels[language_code]['select_page'], [button_labels[language_code]['home'], button_labels[language_code]['about'], button_labels[language_code]['disease_recognition']])

    # TensorFlow Model Prediction
    def model_prediction(test_image):
        model = tf.keras.models.load_model("/content/drive/MyDrive/Plant_Disease_Dataset/trained_plant_disease_model.keras")
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # return index of max element

    # Home Page
    if app_mode == button_labels[language_code]['home']:
        st.header(button_labels[language_code]['plant_disease_recognition'])
        image_path = "/content/drive/MyDrive/Plant_Disease_Dataset/Screenshot 2024-10-27 175010.png"
        st.image(image_path, use_column_width=True)
        st.markdown(button_labels[language_code]['welcome'])
        st.markdown(button_labels[language_code]['how_it_works'])

    # About Project Page
    elif app_mode == button_labels[language_code]['about']:
        st.header(button_labels[language_code]['about_project'])
        st.markdown(button_labels[language_code]['about_dataset'])

    # Disease Recognition Page
    elif app_mode == button_labels[language_code]['disease_recognition']:
        st.header(button_labels[language_code]['disease_recognition'])
        test_image = st.file_uploader(button_labels[language_code]['show_image'])
        if test_image is not None and st.button(button_labels[language_code]['show_image']):
            st.image(test_image, width=400, use_column_width=True)

        # Predict button
        if st.button(button_labels[language_code]['predict']):
            st.snow()
            st.write(button_labels[language_code]['our_prediction'])
            result_index = model_prediction(test_image)
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            st.success(_("Model is Predicting it's a {}").format(class_name[result_index]))
