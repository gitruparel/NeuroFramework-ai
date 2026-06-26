"""Streamlit Research Dashboard application client."""

import streamlit as st


def main():
    st.set_page_config(page_title="MRI Analysis Platform", layout="wide")
    st.title("AI-Powered Structural MRI Analysis Platform")
    st.write("Upload structural MRI scans (T1w, T2w) to generate automated disease classifications, segmentations, and reports.")

    uploaded_file = st.file_uploader("Choose an MRI Volume (.nii, .nii.gz)", type=["nii", "gz"])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        if st.button("Run Diagnostic Pipeline"):
            st.info("Executing Preprocessing and Model Inference on uploaded scan...")


if __name__ == "__main__":
    main()
