import streamlit as st
import cv2
import tempfile
from agent_core import ImageAgent, Pipeline

st.set_page_config(layout="wide")
st.title("🧠 AI Image Detector (Gemini Powered)")

@st.cache_resource
def load_system():
    pipeline = Pipeline()
    agent = ImageAgent(pipeline)
    return agent

agent = load_system()

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded.read())
    image_path = temp.name

    col1, col2 = st.columns(2)

    # ===== INPUT =====
    with col1:
        st.image(image_path, caption="Input", use_container_width=True)

    # ===== RUN MODEL =====
    res = agent.handle("predict", image_path)

    # ===== GRADCAM =====
    with col2:
        cam = res["raw"]["cam"]
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224,224))

        heatmap = cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        st.image(overlay, caption="Grad-CAM", use_container_width=True)

    st.write(f"Confidence: {res['raw']['confidence']:.2f}")
    st.write(f"Risk: {res['raw']['risk']}")

    # ===== CHAT =====
    st.subheader("💬 Chat")
    user_input = st.text_input("Ask anything...")

    if user_input:
        response = agent.handle(user_input, image_path)

        # ===== TEXT =====
        st.write(response["text"])

        # ===== SIMILAR =====
        if "giống" in user_input.lower():
            st.subheader("🔍 Similar Images")

            paths = response["raw"].get("similar_paths", [])
            labels = response["raw"].get("similar_labels", [])

            cols = st.columns(5)

            for i in range(min(5, len(paths))):  # 🔥 tránh overflow
                with cols[i]:
                    st.image(paths[i], use_container_width=True)
                    if i < len(labels):
                        st.caption(labels[i])