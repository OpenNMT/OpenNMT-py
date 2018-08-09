from flask import Flask, request, jsonify
app = Flask(__name__)
from ocr_model import Img2LatexModel
import io
import os
from PIL import Image
import numpy as np

model = Img2LatexModel()

@app.route("/", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))


            # classify the input image and then initialize the list
            # of predictions to return to the client
            data = {}
            data["prediction"] = model.predict([image])[0]
            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
