#!/usr/bin/env python
import os
import onmt
from flask import Flask, jsonify, request


AVAILABLE_MODEL_PATH = "./available_models"
STATUS_OK = "ok"
STATUS_ERROR = "error"

models = onmt.ServerModels()
app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/models', methods=['GET'])
def get_models():
    global models
    out = {}
    try:
        model_list = os.listdir(AVAILABLE_MODEL_PATH)
        out['available'] = model_list
    except Exception as e:
        out['available'] = str(e)

    loaded = []
    for (id, model) in models.models.items():
        if model is not None:
            loaded += [{"model_id": id,
                        "model": model.opt.model, "gpu": model.opt.gpu}]
    out['loaded'] = loaded
    return jsonify(out)


@app.route('/load_model', methods=['POST'])
def load_model():
    global models
    out = {}
    data = request.get_json(force=True)

    try:
        model_name = data['model']
        model_path = os.path.join(AVAILABLE_MODEL_PATH, model_name)
        data['model'] = model_path
    except KeyError:
        out['error'] = "Parameter 'model' is required"
        out['status'] = STATUS_ERROR
    else:
        model_id, load_time = models.load(data)
        out['status'] = STATUS_OK
        out['model_id'] = model_id
        out['load_time'] = load_time

    return jsonify(out)


@app.route('/unload_model/<int:model_id>', methods=['GET'])
def unload_model(model_id):
    global models
    out = {"model_id": model_id}

    try:
        models.unload(model_id)
        out['status'] = STATUS_OK
    except Exception as e:
        out['status'] = STATUS_ERROR
        out['error'] = str(e)

    return jsonify(out)


@app.route('/translate/<int:model_id>', methods=['POST'])
def translate(model_id):
    data = request.get_json(force=True)
    out = {'model_id': model_id}

    try:
        text = data['text']
    except KeyError:
        out['error'] = "Parameter 'text' is required"
        out['status'] = STATUS_ERROR
    else:
        try:
            translation, times = models.run(model_id, text)
            out['result'] = translation
            out['status'] = STATUS_OK
            out['time'] = times
        except onmt.ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

    return jsonify(out)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
