import streamlit as st
import torch
import sys
from torchvision import transforms
from PIL import Image
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.cnn_model import SimpleCNN
from src.resnet import get_resnet18

model_path = base_dir / "models"
example_dir = Path(__file__).parent / "examples"

# CIFAR-10 class
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# transform 
cnn_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# load model (cached)
@st.cache_resource
def load_models():
    # Load CNN checkpoint (weights + metadata)
    cnn = SimpleCNN()
    cnn_checkpoint = torch.load(model_path / "cnn_cifar10_v4.pth", map_location="cpu")
    cnn.load_state_dict(cnn_checkpoint['model_state_dict'])
    cnn.eval()

    # Load ResNet checkpoint (weights + metadata)
    resnet = get_resnet18()
    resnet_checkpoint = torch.load(model_path / "resnet_cifar10_v1.pth", map_location="cpu")
    resnet.load_state_dict(resnet_checkpoint['model_state_dict'])
    resnet.eval()

    return cnn, cnn_checkpoint, resnet, resnet_checkpoint

# prediction function
def predict(model, image, transform):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        with st.spinner("Classifying..."):
            outputs = model(img)

        probs = torch.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3)

    return top3_prob, top3_idx

# 
cnn_model, cnn_checkpoint, resnet_model, resnet_checkpoint = load_models()

# layout
st.set_page_config(page_title="Image Classifier", layout="wide")

st.title("Image Classifier (CIFAR-10)")
st.write("Upload an image and let the model classify it.")

cnn_accuracy = cnn_checkpoint['accuracy']
cnn_epoch = cnn_checkpoint['epoch']
resnet_accuracy = resnet_checkpoint['accuracy']
resnet_epoch = resnet_checkpoint['epoch']

st.sidebar.title("Model Info")

st.sidebar.markdown("### CNN")
st.sidebar.metric("Accuracy", f"{cnn_accuracy:.4f}")
st.sidebar.caption(f"Epoch: {cnn_epoch}")

st.sidebar.markdown("---")

st.sidebar.markdown("### ResNet")
st.sidebar.metric("Accuracy", f"{resnet_accuracy:.4f}")
st.sidebar.caption(f"Epoch: {resnet_epoch}")

st.sidebar.write("""
Model trained on low-resolution images (32x32),
so performance on real-world images may vary.
""")

# Example images
example_images = {
    "Dog": "dog.png",
    "Car": "car.jpg",
    "Cat": "cat.jpg"
}

st.sidebar.subheader("Try example images")

selected_example = st.sidebar.selectbox(
    "Choose example",
    ["None"] + list(example_images.keys())
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

st.subheader("Predictions")
col1, col2, col3 = st.columns([2, 1, 1])

# action
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif selected_example != "None":
    example_path = example_dir / example_images[selected_example]
    image = Image.open(example_path).convert("RGB")

if image is not None:
    with col1:
        st.image(image, caption="Image", use_container_width=True)

    cnn_probs, cnn_idx = predict(cnn_model, image, cnn_transform)
    resnet_probs, resnet_idx = predict(resnet_model, image, resnet_transform)

    with col2:
        st.subheader("CNN")

        for i in range(3):
            class_name = classes[cnn_idx[0][i]]
            confidence = cnn_probs[0][i].item()

            st.write(class_name)
            st.progress(confidence)
            st.caption(f"{confidence:.2f}")

    with col3:
        st.subheader("ResNet")

        for i in range(3):
            class_name = classes[resnet_idx[0][i]]
            confidence = resnet_probs[0][i].item()

            st.write(class_name)
            st.progress(confidence)
            st.caption(f"{confidence:.2f}")

    cnn_pred = classes[cnn_idx[0][0]]
    resnet_pred = classes[resnet_idx[0][0]]

    if cnn_pred != resnet_pred:
        st.warning(f"Models disagree! CNN: {cnn_pred} vs ResNet: {resnet_pred}")
    else:
        st.success(f"Both models agree: {cnn_pred}")

