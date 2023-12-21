import flask
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_fontawesome import FontAwesome
from flask import send_from_directory, Response
import os
# import utils
# import json
import time
import traceback

app = Flask(__name__, template_folder='./')
fa = FontAwesome(app)

app.config['VIDEO_FOLDER'] = 'static/videos'
app.config['AUDIO_FOLDER'] = 'static/audio'


SOURCES = ["vocals", "drums", "bass", "other"]
    

@app.route('/demo')
def demo():
    return render_template('index.html', show_result=0)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
