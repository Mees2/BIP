import streamlit as st
from pathlib import Path
import json
from PIL import Image

# --- CONFIG ---
SCRIPT_DIR = Path(__file__).parent
REPORT_PATH = SCRIPT_DIR / 'glaucoma_analysis_report.json'
GRADCAM_DIR = SCRIPT_DIR  # GradCAM images are saved in the script dir

# --- LOAD DATA ---
@st.cache_data
def load_report():
    with open(REPORT_PATH, 'r') as f:
        return json.load(f)

def get_gradcam_path(image_filename):
    # Try both glaucoma and normal naming
    base = image_filename.split('.')[0]
    for prefix in ['gradcam_glaucoma_', 'gradcam_normal_']:
        for i in range(1, 10):
            candidate = GRADCAM_DIR / f'{prefix}{i}.png'
            if candidate.exists() and base in candidate.name:
                return candidate
    # Fallback: try to find by matching base name
    for img in GRADCAM_DIR.glob('gradcam_*.png'):
        if base in img.name:
            return img
    return None

# --- MAIN INTERFACE ---
st.set_page_config(page_title="Glaucoma AI Review", layout="wide")
st.title("Glaucoma AI Review Interface")
st.markdown("""
This tool allows you to review retinal images, GradCAM visualizations, and AI explanations for glaucoma screening. Use the arrows or select an image to step through the cases.
""")

report = load_report()
image_filenames = [r['image_filename'] for r in report]

# --- IMAGE SELECTION ---
idx = st.slider("Select image index", 0, len(report)-1, 0)
selected = report[idx]

# --- DISPLAY ---
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Original Image")
    orig_path = SCRIPT_DIR / 'data' / 'test' / selected['primary_prediction']['class'] / selected['image_filename']
    if orig_path.exists():
        st.image(str(orig_path), caption="Original", use_column_width=True)
    else:
        st.warning(f"Original image not found: {orig_path}")
    st.subheader("GradCAM Visualization")
    gradcam_path = get_gradcam_path(selected['image_filename'])
    if gradcam_path and gradcam_path.exists():
        st.image(str(gradcam_path), caption="GradCAM", use_column_width=True)
    else:
        st.warning("GradCAM image not found.")

with col2:
    st.subheader("AI Analysis Report")
    st.markdown(f"**Image:** `{selected['image_filename']}`")
    st.markdown(f"**Primary Finding:** `{selected['primary_prediction']['class'].upper()}`")
    st.markdown(f"**Confidence:** `{selected['primary_prediction']['confidence']} ({selected['primary_prediction']['confidence_level']})`")
    st.markdown(f"**Description:** {selected['primary_prediction']['description']}")
    st.markdown(f"**Alternative Finding:** `{selected['alternative_finding']['class']} ({selected['alternative_finding']['confidence']})`")
    st.markdown(f"**Model Consistency:** `{selected['model_consistency']['status']}` - {selected['model_consistency']['message']}")
    st.markdown(f"**Clinical Recommendation:** {selected['clinical_recommendation']}")
    st.markdown(f"**Explanation Summary:**\n{selected['explanation_summary']}")
    st.markdown(f"**Agreement with Surrogate:** `{selected['agreement_with_surrogate']}`")
    st.markdown(f"**Confidence Assessment:** {selected['confidence_assessment']}")

st.markdown("---")
st.markdown("Use the slider above to browse through the images and their explanations.")
