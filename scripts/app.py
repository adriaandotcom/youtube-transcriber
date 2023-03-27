from flask import Flask, request, jsonify
from tempfile import mkdtemp
from yt_dlp import YoutubeDL
from yt_dlp.postprocessor import PostProcessor
from urllib.parse import urlparse, parse_qs
import os
import whisper
import torch
import pickle
import re
import time

torch.cuda.is_available()

# Set your model name, device, and file path
MODEL_NAME = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_FILE_PATH = "whisper_base_model.pkl"

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

if os.path.exists(MODEL_FILE_PATH):
    # Load the model from the file
    print("Loading the model from disk...")
    model = load_model_from_disk(MODEL_FILE_PATH, DEVICE)
else:
    # Download the model, save it to disk, and load it
    print("Downloading the model...")
    model = whisper.load_model(MODEL_NAME, device=DEVICE)
    print("Saving the model to disk...")
    save_model_to_disk(model, MODEL_FILE_PATH)

app = Flask(__name__)

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

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


@app.route("/transcribe", methods=["POST"])
def transcribe_youtube_audio():
    start_time = time.time()

    youtube_url_or_id = request.form.get("youtube_url_or_id")

    if not youtube_url_or_id:
        return jsonify({"error": "YouTube URL or ID is required"}), 400

    youtube_id = validate_youtube_url_or_id(youtube_url_or_id)

    if not youtube_id:
        return jsonify({"error": "Invalid YouTube URL or ID"}), 400

    youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"

    file = download_youtube_audio(youtube_url)
    transcription_result = transcribe_audio(file)

    end_time = time.time()
    response_time = round(end_time - start_time, 2)

    return jsonify({
        "text": transcription_result,
        "youtube_id": youtube_id,
        "youtube_url": youtube_url,
        "response_time": response_time
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
    app.run(host="0.0.0.0", port=8080)
