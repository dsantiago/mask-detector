import numpy as np
import streamlit as st
from PIL import Image
import cv2

from retinaface.pre_trained_models import get_model as get_detector
from facemask_detection.pre_trained_models import get_model as get_classifier

import torch
import albumentations as A

st.set_option("deprecation.showfileUploaderEncoding", False)

@st.cache
def cached_detector(device="cpu"):
    print("## CASHING DETECTOR")
    m = get_detector("resnet50_2020-07-20", max_size=1024, device=device)
    m.eval()
    return m

@st.cache
def cached_classifier():
    print("## CASHING CLASSIFIER")
    m = get_classifier("tf_efficientnet_b0_ns_2020-07-29")
    m.eval()
    return m

@st.cache
def face_annotations(image, model):
    face_detector= cached_detector(device="cpu")
    
    with torch.no_grad():
        annotations = face_detector.predict_jsons(image)
    
    return annotations

@st.cache
def mask_annotations(image):
    mask_classifier = cached_classifier()

    transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_CUBIC),
            A.CenterCrop(height=224, width=224),
            A.Normalize(),
        ]
    )

    crop_transformed = transform(image=image)['image']
    model_input = torch.from_numpy(np.transpose(crop_transformed, (2, 0, 1)))  
    prediction = [mask_classifier(model_input.unsqueeze(0))[0].item()]  

    return prediction


def drawbbox(image, bbox, pred=None):
    vis_image = image.copy()
    x_min, y_min, x_max, y_max = bbox

    if pred is None:
        color = (0, 255, 0)
        vis_image = cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
        return vis_image

    color = (255, 0, 0) # red if no mask
    text = f"no mask {pred:.2f}"

    if pred > 0.5:
        color = (0, 255, 0) # green if with mask
        text = f"mask {pred:.2f}"

    x_min = np.clip(x_min, 0, x_max - 1)
    y_min = np.clip(y_min, 0, y_max - 1)

    vis_image = cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
    vis_image = cv2.putText(vis_image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return vis_image


def main():
    model = cached_detector("cpu")

    st.title("Detect faces")
    st.write("Diogo Santiago - https://github.com/dsantiago/mask-detector")
    st.write("---")
    st.sidebar.title('Features')

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    use_mask = st.sidebar.checkbox("Check Mask?")

    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        image = image[:, :, :3] # Fix to PNG's with 4 channels

        st.subheader("Original")
        st.image(image, caption="Before", use_column_width=True)
        
        subtxt = "Detecting faces..."

        if use_mask:
            subtxt = "Detecting faces and masks..."
        
        st.subheader(subtxt)
        annotations = face_annotations(image, model)

        if not annotations[0]["bbox"]:
            st.write("No faces detected")
            return
    
        visualized_image = image.copy()
           
        for annotation in annotations:
            x_min, y_min, x_max, y_max = annotation['bbox']
            x_min = np.clip(x_min, 0, x_max)
            y_min = np.clip(y_min, 0, y_max)

            prediction = None

            if use_mask:
                crop = image[y_min:y_max, x_min:x_max]
                prediction = mask_annotations(crop)[0]

            visualized_image = drawbbox(visualized_image, [x_min, y_min, x_max, y_max], prediction)

        st.image(visualized_image, caption="After", use_column_width=True)


if __name__ == "__main__":
    main()
