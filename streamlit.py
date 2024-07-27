import streamlit as st
import os
import boto3
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from zipfile import ZipFile
from model import create_model


# Function to download the model weights from S3
def download_weights_from_s3(bucket_name, object_key, weights_path):
    weights_dir = os.path.dirname(weights_path)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        st.info(f"Created directory {weights_dir} for weights.")

    if not os.path.exists(weights_path):
        st.info("Weights not found locally. Downloading from S3...")
        s3 = boto3.client('s3')
        s3.download_file(Bucket=bucket_name, Key=object_key, Filename=weights_path)
        st.success("Weights downloaded.")
    else:
        st.info("Weights found locally. Loading...")

# Function to load the model, ensuring it's downloaded if not present
def load_model(weights_path):
    bucket_name = 'segmentation-model-bucket'
    object_key = 'weights/weights.h5'
    download_weights_from_s3(bucket_name, object_key, weights_path)
    model = create_model(img_height=352, img_width=352, input_chanels=3, out_classes=1, starting_filters=34)
    model.load_weights(weights_path)
    return model

def clear_temp_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def generate_data_batches(data_dir, batch_size, target_size, seed=None):
    rescale_factor = 1.0 / 255.0
    if seed is not None:
        np.random.seed(seed)

    test_data_generator = ImageDataGenerator(rescale=rescale_factor)
    return test_data_generator.flow_from_directory(
        data_dir + '/app_images',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
        seed=seed
    )

def process_and_display_images(loaded_model, uploaded_files, temp_dir):
    try:
        clear_temp_directory(temp_dir)
        masks_dir = os.path.join(temp_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        batch_size = 16
        test_images_gen = generate_data_batches('./temp_app_data/', batch_size, (352, 352), seed=123)
        total_files = len(uploaded_files)
        steps_needed = np.ceil(total_files / batch_size)
        global_index = 0

        for _ in range(int(steps_needed)):
            batch = next(test_images_gen)
            predictions = loaded_model.predict(batch)
            binary_predictions = (predictions > 0.5).astype(np.uint8)

            for j, pred_image in enumerate(binary_predictions):
                if global_index >= len(uploaded_files):
                    break

                pred_pil_image = Image.fromarray(pred_image.squeeze() * 255)
                col1, col2 = st.columns(2)
                with col1:
                    original_image = Image.open(uploaded_files[global_index])
                    st.image(original_image, caption='Original Image', use_column_width=True)
                with col2:
                    st.image(pred_pil_image, caption='Predicted Mask', use_column_width=True)

                original_file_name = os.path.basename(uploaded_files[global_index].name)
                original_name_without_ext = os.path.splitext(original_file_name)[0]
                mask_filename = f"mask_{original_name_without_ext}.jpg"
                mask_path = os.path.join(masks_dir, mask_filename)
                pred_pil_image.save(mask_path, "JPEG")
                global_index += 1

        zip_path = os.path.join(temp_dir, "masks.zip")
        with ZipFile(zip_path, 'w') as zipf:
            for mask_file in os.listdir(masks_dir):
                zipf.write(os.path.join(masks_dir, mask_file), arcname=mask_file)

        with open(zip_path, "rb") as f:
            st.download_button("Download All Masks as ZIP", f.read(), "masks.zip", "application/zip")

        shutil.rmtree('./temp_app_data')
        
    except Exception as e:
        st.error(f"An error occurred while processing the images: {e}")

def main():
    st.title("Polyp Segmentation Tool")
    st.image("https://production-media.paperswithcode.com/datasets/Screenshot_from_2021-05-05_23-44-10.png", use_column_width=True)
    st.header("How to Use This Tool")
    st.write("""
        This tool allows healthcare professionals to upload colonoscopy images and receive automated polyp segmentation.
        - **Original Image**: Shows the uploaded colonoscopy image.
        - **Predicted Mask**: Shows the segmentation of the polyp.
        Supported file formats: jpg, jpeg.
    """)
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['jpg', 'jpeg'])
    temp_dir = './temp_app_data/app_images/test'
    if uploaded_files:
        if st.button('Process Images'):
            with st.spinner('Processing images...'):
                model_path = 'your_path_weights'
                model = load_model(model_path)
                process_and_display_images(model, uploaded_files, temp_dir)
                st.success('Processing complete!')

if __name__ == "__main__":
    main()
