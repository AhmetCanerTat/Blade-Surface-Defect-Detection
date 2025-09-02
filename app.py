
import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
from model import get_efficientnetv2_s_model_layer_added
import numpy as np
import cv2
from grad_cam import apply_grad_cam_efficientnetv2s
from preprocessing import preprocess_image_cv2,expand_channel

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER


# Label mapping (should match training)
unique_labels = sorted(['Good', 'Nick', 'Scratch'])  # Update if your classes are different
idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}

# Model
model = get_efficientnetv2_s_model_layer_added(len(unique_labels), freeze=False)
model.load_state_dict(torch.load('best_model_2.pth', map_location='cpu'))
model.eval()
device = torch.device('cpu')

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization if used in training
    ])
   
    return transform(image_bytes).unsqueeze(0)


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        prediction = None
        gradcam_url = None
        uploaded_url = None
        file = request.files['file']
        if file:
            # Save uploaded image
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            uploaded_url = url_for('static', filename=f'uploads/{filename}')

            # Preprocessing (CLAHE pipeline)
            img_cv2 = cv2.imread(upload_path)
            _, _, _, clahe = preprocess_image_cv2(img_cv2, resize_shape=(256, 256))
            clahe = expand_channel(clahe)
            img_tensor = transforms.ToTensor()(clahe).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
            prediction = idx_to_label.get(pred, str(pred))

            # Grad-CAM (use merged image for visualization)
            gradcam_img = apply_grad_cam_efficientnetv2s(model, clahe, pred, device)
            # Resize Grad-CAM to original uploaded image size
            orig_img = Image.open(upload_path)
            gradcam_img_pil = Image.fromarray(gradcam_img).resize(orig_img.size, resample=Image.BILINEAR)
            gradcam_filename = f'gradcam_{filename}'
            gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
            gradcam_img_pil.save(gradcam_path)
            gradcam_url = url_for('static', filename=f'gradcam/{gradcam_filename}')
        return render_template('index.html', prediction=prediction, gradcam_url=gradcam_url, uploaded_url=uploaded_url)
    # For GET requests, always return clean page
    return render_template('index.html', prediction=None, gradcam_url=None, uploaded_url=None)

if __name__ == '__main__':
    app.run(debug=True)