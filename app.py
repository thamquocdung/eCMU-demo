import flask
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_fontawesome import FontAwesome
from flask import send_from_directory, Response
import os
import utils
import json
import time
import traceback
from separator import Separator

app = Flask(__name__, template_folder='./templates')
fa = FontAwesome(app)

app.config['VIDEO_FOLDER'] = 'static/videos'
app.config['AUDIO_FOLDER'] = 'static/audio'


SOURCES = ["vocals", "drums", "bass", "other"]
separator = Separator()

def get_samples():
    song_infos = []
    sample_folder = "static/samples"

    song_names = sorted(os.listdir(sample_folder))
    song_names.remove('.DS_Store')
    for song_id, song_name in enumerate(song_names):
        info = {"song_name": song_name}
        info["mixture_audio"] = os.path.join(sample_folder, song_name, "mix.wav")
        info["mixture_spec"] = os.path.join(sample_folder, song_name, "spec_mix.png")
        info["src_audio_base"] = os.path.join(sample_folder, song_name)

        song_infos.append(info)
    return song_infos


@app.route('/')
def index():
    samples = get_samples()
    return render_template('index.html', samples=samples, source_names=SOURCES)

@app.route('/separate', methods=["GET", "POST"])
def separate():
    if flask.request.method == 'GET':
        return flask.redirect("/demo")

    url = request.form.get("url")
    time.sleep(5)
    
    try:
        video_filename, _ = utils._yt_dlp_download(url, quality=720, dst=app.config["VIDEO_FOLDER"])
        out_video, out_audio = utils.cut_video(video_filename, from_second=145, to_second=160)
        # audio_filename, status_a = utils.download_audio_file(url, dst=app.config["AUDIO_FOLDER"])
        video_filename = out_video # "static/videos/sample.mp4"
        filename = out_audio.split("/")[-1][:-4]
        separator.separate(out_audio)
        inner_html = render_template('result.html', video_src=out_video, filename=filename, show_result=1)
        return_data = {"innerHTML": inner_html, "filename": filename}
        return Response(json.dumps(return_data), mimetype=u'application/json')
    except:
        traceback.print_exc()
        return Response("Error: Can't download your video. Please try another one!", mimetype=u'application/json', status=401)
        
   

    

@app.route('/demo')
def demo():
    return render_template('demo.html', show_result=0)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
