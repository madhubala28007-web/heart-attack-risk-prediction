from joblib import load
from flask import Flask, render_template_string, request
import os
from predict import predict_image
from joblib import load   # pickle use panna 'import pickle' use pannunga

app =Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")  # unga model file peru ennavo adhu podunga
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Heart Attack Risk Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #eef3f8;
            margin: 0;
            padding: 0;
        }
        header {
            background: #b71c1c;
            color: white;
            padding: 25px;
            text-align: center;
        }
        .container {
            background: white;
            max-width: 650px;
            margin: 40px auto;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px #cccccc;
            text-align: center;
        }
        input, button {
            margin: 15px;
            padding: 12px;
        }
        button {
            background: #b71c1c;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .result {
            background: #fff3f3;
            padding: 15px;
            margin-top: 20px;
            border-radius: 8px;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #555;
        }
    </style>
</head>
<body>

<header>
    <h1>Heart Attack Risk Prediction</h1>
    <p>Retinal Eye Image Analysis Using Machine Learning</p>
</header>

<div class="container">
    <h2>Upload Retinal Eye Image</h2>

    <p>Choose a retinal fundus image for analysis.</p>

    <form action="/predict" method="POST"
          enctype="multipart/form-data">

        <input type="file" name="image"
               accept=".jpg,.jpeg,.png" required>

        <br>

        <button type="submit">Analyze Image</button>
    </form>

    {% if result %}
    <div class="result">
        <h3>{{ result }}</h3>
        <p>{{ message }}</p>
    </div>
    {% endif %}

    <p>
        Note: This is a demonstration website.
        It does not provide a medical diagnosis.
    </p>
</div>

<footer>
    BCA Final Year Project | Python Flask
</footer>

</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files.get("image")

    if not image or image.filename == "":
        return render_template_string(
            HTML,
            result="No Image Selected",
            message="Please upload an image."
        )

    allowed_extensions = (".jpg", ".jpeg", ".png")

    if not image.filename.lower().endswith(allowed_extensions):
        return render_template_string(
            HTML,
            result="Invalid File",
            message="Please upload a JPG or PNG image."
        )

    filename = os.path.basename(image.filename)
    image.save(os.path.join(UPLOAD_FOLDER, filename))
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    prob, category, res = predict_image(saved_path)

    return render_template_string(
        HTML,
        result=f"Risk Category: {category}",
        message=f"Probability of High Risk: {prob:.2%}"
    )
     
    
if __name__== "__main__":
    app.run(debug=True) 