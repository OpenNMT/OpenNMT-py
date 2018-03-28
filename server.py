#!/usr/bin/env python
import os
import onmt
from flask import Flask, jsonify, request
from onmt.translate import TranslationServer

AVAILABLE_MODEL_PATH = "./available_models"
STATUS_OK = "ok"
STATUS_ERROR = "error"

translation_server = TranslationServer()
app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/models', methods=['GET'])
def get_models():
    global translation_server
    out = {}
    try:
        model_list = os.listdir(AVAILABLE_MODEL_PATH)
        out['available'] = model_list
    except Exception as e:
        out['available'] = str(e)

    loaded = []
    for (id, model) in translation_server.models.items():
        if model is not None:
            loaded += [{"model_id": id,
                        "model": model.opt.model, "gpu": model.opt.gpu}]
    out['loaded'] = loaded
    return jsonify(out)


@app.route('/load_model', methods=['POST'])
def load_model():
    global translation_server
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
        model_id, load_time = translation_server.load_model(data)
        out['status'] = STATUS_OK
        out['model_id'] = model_id
        out['load_time'] = load_time

    return jsonify(out)


@app.route('/unload_model/<int:model_id>', methods=['GET'])
def unload_model(model_id):
    global translation_server
    out = {"model_id": model_id}

    try:
        translation_server.unload_model(model_id)
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
            translation, times = translation_server.run_model(model_id, text)
            out['result'] = translation
            out['status'] = STATUS_OK
            out['time'] = times
        except onmt.ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

    return jsonify(out)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
