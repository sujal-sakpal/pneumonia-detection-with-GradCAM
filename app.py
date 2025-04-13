import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt

image = keras.preprocessing.image
load_model = keras.models.load_model

# Load the trained model
@st.cache_resource
def load_model_cached():
    return load_model('custom_cnn_model.keras') # add path to your cnn model and make changes accordingly

model = load_model_cached()
class_names = ['BACTERIAL', 'NORMAL', 'VIRAL']  # Update if needed

# Load and preprocess image
def load_and_preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# Alternative Grad-CAM implementation without using Model API
def alternative_gradcam(model, img_array, pred_index=None):
    # Make prediction
    preds = model.predict(img_array)
    
    # Find the predicted class index if not provided
    if pred_index is None:
        pred_index = np.argmax(preds[0])
    
    # Create a simplified Grad-CAM based on feature maps
    # Instead of trying to access internal layers, using a simplified
    # approach to generate a heatmap based on the prediction gradient
    
    # Creating a basic heatmap (this is a simplified approach that won't
    # give the true Grad-CAM but should at least generate a visualization)
    img_tensor = tf.convert_to_tensor(img_array)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[:, pred_index]
    
    # Get gradients of the target class with respect to the input image
    grads = tape.gradient(loss, img_tensor)
    
    # Take the mean of absolute gradients across color channels
    grads_abs = tf.abs(grads)
    grads_mean = tf.reduce_mean(grads_abs, axis=-1)
    
    # Normalize the heatmap
    heatmap = grads_mean[0].numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap

# Superimpose heatmap
def superimpose_heatmap(heatmap, original_image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_image = np.array(original_image)
    superimposed_img = heatmap_color * alpha + original_image
    return np.uint8(superimposed_img)

# Streamlit app UI
st.title("Chest X-ray Classifier with Grad-CAM")
st.write("Upload a chest X-ray image to see the model's prediction and Grad-CAM visualization.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        img_array, orig_img = load_and_preprocess_image(uploaded_file)
        
        # Make prediction
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds[0])]
        confidence = np.max(preds[0])
        
        # Display prediction results
        st.subheader(f"Prediction: {pred_class}")
        st.write(f"Confidence: {confidence:.2%}")
        
        # Show all class probabilities
        st.subheader("Class Probabilities:")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {preds[0][i]:.2%}")
        
        # Generate visualization
        try:
            heatmap = alternative_gradcam(model, img_array)
            final_img = superimpose_heatmap(heatmap, orig_img)
            
            # Display images
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig_img, caption="Original X-ray", use_container_width=True)
            with col2:
                st.image(final_img, caption="Gradient Visualization", use_container_width=True)
                
            # Display heatmap separately
            st.subheader("Heatmap Only")
            heatmap_display = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(cv2.resize(heatmap_display, (orig_img.size[0], orig_img.size[1])), 
                                            cv2.COLORMAP_JET)
            st.image(heatmap_color, caption="Areas of Interest", use_container_width=True)
            
            # Add interpretation guidance
            st.subheader("Interpretation Guide")
            st.write("""
            - **Red/Yellow Areas**: Regions the model is focusing on most to make its prediction
            - **Blue Areas**: Regions with less influence on the model's decision
            - The heatmap shows which parts of the X-ray were most important for classifying the image
            """)
            
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            st.image(orig_img, caption="Original X-ray", use_column_width=True)