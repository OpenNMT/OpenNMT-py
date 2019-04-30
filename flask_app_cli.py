from flask import Flask, request, abort, jsonify, render_template
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
    return "Welcome to DL backend of Open Curriculum"


@app.route('/translate', methods=['POST'])
def translate():
    if not request.json or not 'text' in request.json:
        abort(400)

    input = request.json['text']
    input_language = request.json['src_lang']
    output_language = request.json['dst_lang']

    # Translation
    try:
      output = aws_translate(input, input_language, output_language)
    except:
        data = {
          'action': "translated",
          'src_lang': input_language,
          'dst_lang': output_language,
          'result': ""
        }
        return jsonify(data), 500

    data = {
        'action': "translated",
        'src_lang': input_language,
        'dst_lang': output_language,
        'result': output
    }

    return jsonify(data), 201


@app.route('/summary', methods=['POST'])
def summary():
    if not request.json or not 'text' in request.json:
        abort(400)

    input = request.json['text']
    try:
      # Write input to file
      tmp_file = random_string()
      open_file = open(tmp_file, 'w')
      open_file.write(input)
      open_file.close()

      file_suffix = '_summarized'
      # To summarization
      print('In Summarization')
      summarize(tmp_file, suffix = file_suffix)
    
      open_file = open(tmp_file + file_suffix , 'r')
      output = open_file.readlines()
      output = ''.join(output)
      open_file.close()

      # TODO: Delete file
      os.system("rm {}".format(tmp_file))
      os.system("rm {}".format(tmp_file+file_suffix))
      
    except:
      data = {
        'action': "summary",
        'result': ""
      }
      return jsonify(data), 500

    data = {
        'action': "summary",
        'result': output
    }

    return jsonify(data), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000 )
