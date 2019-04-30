from flask import Flask, request, abort, jsonify
import string
import random
import os
# API import
from aws_translation import aws_translate
from text_summarization_api import summarize

app = Flask(__name__)

def random_string(size=4):
  chars = "abcdefghijklmnopqrstuvwxyz"
  return ''.join(random.choice(chars) for _ in range(size))

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/translate', methods=['POST'])
def translate():
    if not request.json or not 'text' in request.json:
        abort(400)

    input = request.json['text']
    input_language = request.json['in']
    output_language = request.json['out']

    # Translation
    output = aws_translate(input, input_language, output_language)

    data = {
        'action': "translated",
        'input': input_language,
        'output': output_language,
        'result': output
    }

    return jsonify(data), 201


@app.route('/summary', methods=['POST'])
def summary():
    if not request.json or not 'text' in request.json:
        abort(400)

    input = request.json['text']
    # Write input to file
    tmp_file = random_string()
    open_file = open(tmp_file, 'w')
    open_file.write(input)
    open_file.close()

    file_suffix = '_summarized'
    # To summarization
    print('In SUmmary')
    summarize(tmp_file, suffix = file_suffix)
  
    open_file = open(tmp_file + file_suffix , 'r')
    output = open_file.readlines()
    output = ''.join(output)
    open_file.close()

    # TODO: Delete file
    os.system("rm {}".format(tmp_file))
    os.system("rm {}".format(tmp_file+file_suffix))
    
    data = {
        'action': "summary",
        'result': output
    }

    return jsonify(data), 201


if __name__ == '__main__':
    app.run()