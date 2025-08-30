import os
from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 5)  # Assuming 5 output classes
    model.load_state_dict(torch.load("model/retinal_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define image transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_label = None
    image_path = None

    if request.method == "POST":
        file = request.files['file']
        if file:
            try:
                image = Image.open(file).convert("RGB")
                # Get original filename extension
                ext = os.path.splitext(file.filename)[1]

# Predict first (so we have the label)
                input_tensor = transform_pipeline(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    prediction = int(predicted.item())
                    prediction_label = CLASS_NAMES.get(prediction, "Unknown")

# Create new filename
                    new_filename = f"{prediction_label.replace(' ', '_')}_{file.filename}"
                    save_path = os.path.join(UPLOAD_FOLDER, new_filename)
                    image.save(save_path)
                    image_path = f"/static/uploads/{new_filename}"

                

                input_tensor = transform_pipeline(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    prediction = int(predicted.item())

                    prediction_label = CLASS_NAMES.get(prediction, "Unknown")
            except Exception as e:
                prediction_label = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction_label, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
