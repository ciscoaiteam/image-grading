import streamlit as st
import requests
import base64
from io import BytesIO

# Page config
st.set_page_config(page_title="Image Analysis for Grading WTF?", layout="wide")

# Inject Tailwind CSS and custom styles
st.markdown(
    """
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
      .drag-area {
          border: 2px dashed #6366f1;
          transition: all 0.3s ease;
      }
      .drag-area.active {
          border-color: #4338ca;
          background-color: rgba(99, 102, 241, 0.1);
      }
    </style>
    """, unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="bg-indigo-600 text-white rounded-lg shadow-md mb-8 py-6">
      <h1 class="text-3xl md:text-4xl font-bold text-center">Image Analysis for Grading WTF?</h1>
    </div>
    """, unsafe_allow_html=True
)

# Layout: two columns
col1, col2 = st.columns(2)

with col1:
    # Upload section container
    st.markdown("<div class='bg-white p-6 rounded-lg shadow-md'>", unsafe_allow_html=True)
    st.markdown("<h2 class='text-2xl font-semibold mb-4'>Upload Image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(label="", type=["png", "jpg", "jpeg"], key="uploader")

    if uploaded_file:
        image_bytes = uploaded_file.read()
        if st.button("Submit for Analysis"):
            with st.spinner("Analyzing..."):
                try:
                    files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                    response = requests.post(
                        "http://localhost:8000/predict",
                        files=files
                    )
                    response.raise_for_status()
                    data = response.json()
                    st.session_state['result'] = {
                        'bytes': image_bytes,
                        'classification_prob': data['classification_prob'],
                        'regression_score': data['regression_score']
                    }
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Results section container
    st.markdown("<div class='bg-white p-6 rounded-lg shadow-md'>", unsafe_allow_html=True)
    st.markdown("<h2 class='text-2xl font-semibold mb-4'>Analysis Results</h2>", unsafe_allow_html=True)

    if 'result' in st.session_state:
        res = st.session_state['result']
        # Convert image to base64 for inline HTML results
        img_base64 = base64.b64encode(res['bytes']).decode('utf-8')
        # Display predictions first
        result_html = f"""
        <div class='space-y-6'>
          <div class='grid grid-cols-1 md:grid-cols-2 gap-4'>
            <div class='bg-indigo-50 p-4 rounded-lg'>
              <h3 class='text-lg font-medium mb-2'>Classification Probability</h3>
              <div class='flex items-center justify-center'>
                <span class='text-2xl font-bold text-indigo-600'>{res['classification_prob']:.2f}</span>
              </div>
            </div>
            <div class='bg-indigo-50 p-4 rounded-lg'>
              <h3 class='text-lg font-medium mb-2'>Regression Score</h3>
              <div class='flex flex-col items-center'>
                <span class='text-2xl font-bold text-indigo-600'>{res['regression_score']:.2f}</span>
                <div class='w-full bg-gray-200 rounded-full h-2.5 mt-2'>
                  <div class='bg-green-600 h-2.5 rounded-full' style='width: {res['regression_score']:.2f}%;'></div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
        # Then display the image at the bottom
        st.markdown("<h3 class='text-lg font-medium mt-4 mb-2'>Processed Image</h3>", unsafe_allow_html=True)
        st.image(res['bytes'], use_container_width=True)
    else:
        st.markdown(
            "<div class='text-center text-gray-500'><p>No image analyzed yet. Upload an image to see results.</p></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
