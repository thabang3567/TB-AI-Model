import streamlit as st
from PIL import Image
import os
import glob
import random

st.set_page_config(page_title="Spectrogram Visualizer", layout="wide")
st.title("🫁 Cough Spectrogram Visualizer")
st.write("Comparing the acoustic frequency patterns across the 3 respiratory classes.")

base_dir = r"C:\Users\Thabang Moloko\PyCharmMiscProject\Chest Diseases Dataset"

# Removed Pneumonia
class_folders = {
    "Healthy": os.path.join(base_dir, "9. Normal", "CSI"),
    "Tuberculosis": os.path.join(base_dir, "5. Tuberculosis", "CSI"),
    "COVID-19": os.path.join(base_dir, "1. COVID-19", "CSI")
}

if st.button("🔀 Load Random Samples", type="primary"):
    pass

st.markdown("---")
# Changed to 3 columns
cols = st.columns(3)

for col, (class_name, folder_path) in zip(cols, class_folders.items()):
    with col:
        st.subheader(class_name)
        valid_extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG')
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        if image_files:
            sample_image_path = random.choice(image_files)
            try:
                img = Image.open(sample_image_path)
                st.image(img, use_container_width=True)
                st.caption(f"📁 **Dataset Size:** {len(image_files)} images")
                st.caption(f"📄 `{os.path.basename(sample_image_path)}`")
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.error(f"No valid images found in folder.")
            st.caption(f"`{folder_path}`")