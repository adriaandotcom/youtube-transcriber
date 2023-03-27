from flask import Flask, request, jsonify, render_template
from tempfile import mkdtemp
from yt_dlp import YoutubeDL
from yt_dlp.postprocessor import PostProcessor
import os
import whisper
import torch
import pickle
import re
import time

# Set your model name, device, and file path
DEFAULT_MODEL_NAME = "tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

MODEL_FILE_PATH = "/app/data/whisper_{}_model.pkl"


def validate_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        return DEFAULT_MODEL_NAME
    return model_name


def load_model(model_name, device):
    model_file_path = MODEL_FILE_PATH.format(model_name)
    if os.path.exists(model_file_path):
        print(f"Loading the {model_name} model from disk...")
        model = load_model_from_disk(model_file_path, DEVICE)
    else:
        print(f"Downloading the {model_name} model...")
        model = whisper.load_model(model_name, device=DEVICE)
        print(f"Saving the {model_name} model to disk...")
        save_model_to_disk(model, model_file_path)
    return model


class FilenameCollectorPP(PostProcessor):
    def __init__(self):
        super(FilenameCollectorPP, self).__init__(None)
        self.filenames = []

    def run(self, information):
        self.filenames.append(information["filepath"])
        return [], information


def save_model_to_disk(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)


def load_model_from_disk(file_path, device):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model.to(device)


app = Flask(__name__,
            static_folder="/app/static",
            template_folder="/app/src/templates"
            )


def download_youtube_audio(url: str):
    destinationDirectory = mkdtemp()

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'ffmpeg_location': '/usr/bin/',
        'playlist_items': '1',
        'paths': {
            'home': destinationDirectory
        }
    }
    filename_collector = FilenameCollectorPP()

    with YoutubeDL(ydl_opts) as ydl:
        ydl.add_post_processor(filename_collector)
        ydl.download([url])

    if len(filename_collector.filenames) <= 0:
        raise Exception("Cannot download " + url)

    result = filename_collector.filenames[0]

    return result


@app.route("/", methods=["GET"])
def home():

    # Log user agent
    print(request.headers.get("User-Agent"))

    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe_youtube_audio():
    start_time = time.time()

    youtube_url_or_id = request.form.get("youtube_url_or_id")
    model_name = request.form.get("model")

    if not youtube_url_or_id:
        return jsonify({"error": "YouTube URL or ID is required"}), 400

    youtube_id = validate_youtube_url_or_id(youtube_url_or_id)
    model_name = validate_model(model_name)

    if not youtube_id:
        return jsonify({"error": "Invalid YouTube URL or ID"}), 400

    youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
    model = load_model(model_name, DEVICE)

    file = download_youtube_audio(youtube_url)
    result = model.transcribe(file)
    transcription_result = result["text"]

    end_time = time.time()
    response_time = round(end_time - start_time, 2)

    return jsonify({
        "text": transcription_result,
        "youtube_id": youtube_id,
        "youtube_url": youtube_url,
        "response_time": response_time,
        "model": model_name
    })


def validate_youtube_url_or_id(youtube_url_or_id):
    youtube_id_pattern = r"(?:http(?:s)?://)?(?:www\.)?(?:(?:youtube.com/watch\?v=)|(?:youtu.be/))([a-zA-Z0-9_-]{11})|([a-zA-Z0-9_-]{11})"
    match = re.match(youtube_id_pattern, youtube_url_or_id)

    if match:
        if match.group(1):
            return match.group(1)
        else:
            return match.group(2)
    else:
        return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
