import streamlit as st
import torch
import sys
from torchvision import transforms
from PIL import Image
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.model import SimpleCNN

model_path = base_dir / "models"
example_dir = Path(__file__).parent / "examples"

# CIFAR-10 class
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# transform 
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# load model
@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path / "cnn_cifar10_v1.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.set_page_config(page_title="Image Classifier", layout="wide")

st.title("Image Classifier (CIFAR-10)")
st.write("Upload an image and let the model classify it.")

st.sidebar.title("About")
st.sidebar.write("""
Model: CNN (PyTorch)  
Dataset: CIFAR-10  
Accuracy: ~71%  
""")

st.sidebar.write("""
Model trained on low-resolution images (32x32),
so performance on real-world images may vary.
""")

example_images = {
    "Dog": example_dir / "dog.png",
    "Car": example_dir / "car.jpg",
    "Cat": example_dir / "cat.jpg"
}

st.sidebar.subheader("Try example images")

selected_example = st.sidebar.selectbox(
    "Choose example",
    ["None"] + list(example_images.keys())
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns([2, 1])

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif selected_example != "None":
    example_path = example_dir / example_images[selected_example]
    image = Image.open(example_path).convert("RGB")

col1, col2 = st.columns([2, 1])

if image is not None:
    with col1:
        st.image(image, caption="Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        with st.spinner("Classifying..."):
            outputs = model(img)

        probs = torch.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3)

    with col2:
        st.subheader("Predictions")

        for i in range(3):
            class_name = classes[top3_idx[0][i]]
            confidence = top3_prob[0][i].item()

            st.write(class_name)
            st.progress(confidence)
            st.caption(f"{confidence:.2f}")