from logging.handlers import RotatingFileHandler
from flask import Flask, request, render_template, flash, redirect
from flask.ext.runner import Runner
from forms import InputForm
import time
import logging
import json
from flask_jsonpify import jsonify
import amrtotext

app = Flask(__name__)
app.config.from_object('config')

runner = Runner(app)

@app.route('/amrtotext', methods=['GET', 'POST'])
def main():
    form = InputForm()
    amr = request.args.get('amr')
    model = request.args.get('model')
    print(model)
    sent = None
    if form.validate_on_submit():
        sent = amrtotext.run(form.input_amr.data, form.input_amr.model)

    elif amr is not None:
        sent = amrtotext.run(amr, model)
        if sent is not None:
            app.logger.info('Input: ' + amr)
            link = "http://bollin.inf.ed.ac.uk:9020/amrtotext?model=" + model + "&amr=" + "+".join([t for t in amr.split()])
            return jsonify({'sent': sent.replace("\n","<br/>").replace(" ","&nbsp;"), 'link': link}, sort_keys=True, indent=4, separators=(',', ': '))
        else:
            return jsonify({'sent': 'ERROR', 'link': ''}, sort_keys=True, indent=4, separators=(',', ': ')) 
        
if __name__ == '__main__':
    handler = RotatingFileHandler('amr.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG)
    runner.run()
