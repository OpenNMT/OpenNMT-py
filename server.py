#!/usr/bin/env python
import os
import argparse

from flask import Flask, jsonify, request
from onmt.translate import TranslationServer, ServerModelError

ROOT = "/translator"
AVAILABLE_MODEL_PATH = "./available_models"
STATUS_OK = "ok"
STATUS_ERROR = "error"


def prefix_route(route_function, prefix='', mask='{0}{1}'):
    def newroute(route, *args, **kwargs):
        return route_function(mask.format(prefix, route), *args, **kwargs)
    return newroute


app = Flask(__name__)
app.route = prefix_route(app.route, ROOT)
translation_server = TranslationServer()


@app.route('/models', methods=['GET'])
def get_models():
    global translation_server
    out = translation_server.list_models()

    return jsonify(out)


@app.route('/clone_model/<int:model_id>', methods=['POST'])
def clone_model(model_id):
    global translation_server
    out = {}
    data = request.get_json(force=True)

    timeout = -1
    if 'timeout' in data:
        timeout = data['timeout']
        del data['timeout']

    opt = data.get('opt', None)
    try:
        model_id, load_time = translation_server.clone_model(
            model_id, opt, timeout)
    except ServerModelError as e:
        out['status'] = STATUS_ERROR
        out['error'] = str(e)
    else:
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


@app.route('/translate', methods=['POST'])
def translate():
    global translation_server
    data = request.get_json(force=True)
    out = {}

    inputs = data
    try:
        # NOTE Ubiqus adhoc model_id is in the inputs
        translation, times = translation_server.run_model(-9999, inputs)
        out['result'] = translation
        out['status'] = STATUS_OK
        out['time'] = times

        out = [[{"src": "dummy src", "tgt": _, "n_best": -
                 1, "pred_score": 99999}for _ in translation]]
    except ServerModelError as e:
        out['error'] = str(e)
        out['status'] = STATUS_ERROR

    return jsonify(out)


@app.route('/to_cpu/<int:model_id>', methods=['GET'])
def to_cpu(model_id):
    out = {'model_id': model_id}
    translation_server.models[model_id].to_cpu()

    out['status'] = STATUS_OK
    return jsonify(out)


@app.route('/to_gpu/<int:model_id>', methods=['GET'])
def to_gpu(model_id):
    out = {'model_id': model_id}
    translation_server.models[model_id].to_gpu()

    out['status'] = STATUS_OK
    return jsonify(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OpenNMT-py REST Server")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--config", "-c", type=str,
                        default="./available_models/conf.json")
    args = parser.parse_args()

    translation_server.start(args.config)
    app.run(debug=args.debug, host=args.ip, port=args.port, use_reloader=False)
