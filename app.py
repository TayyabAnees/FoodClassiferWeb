import os
import time

from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load model
def load_model():
    model = models.googlenet(pretrained=False, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("googlenet_food_best.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Prediction function
def predict_image(image):
    time.sleep(4);
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    class_names = ["Non-Food", "Food"]
    return class_names[pred_class.item()], confidence.item() * 100


# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        image_file = request.files["image"]
        if image_file.filename == "":
            return redirect(request.url)

        filename = image_file.filename
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)

        image = Image.open(image_path).convert("RGB")
        label, confidence = predict_image(image)

        return render_template("result.html", image_filename=filename, label=label, confidence=confidence)

    # folder path code stays the same...
    elif "folder_path" in request.form:
        folder_path = request.form["folder_path"]
        results = []

        if not os.path.isdir(folder_path):
            return render_template("result.html", error="Invalid folder path!")

        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(folder_path, fname)
                try:
                    image = Image.open(fpath).convert("RGB")
                    label, confidence = predict_image(image)
                    results.append((fname, label, confidence))
                except:
                    continue

        return render_template("result.html", folder_results=results)

    return redirect(url_for("index"))


@app.route("/delete_image", methods=["POST"])
def delete_image():
    data = request.get_json()
    filename = data.get("filename")
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "not found"})


if __name__ == "__main__":
    app.run(debug=True)
