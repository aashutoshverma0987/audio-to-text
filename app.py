from flask import Flask, render_template, request
import whisper
import os
from transformers import pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# uploads folder create
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Models
print("Loading Whisper Model...")
model = whisper.load_model("base")

print("Loading Sentiment Model...")
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def index():

    transcription = ""
    sentiment = ""

    if request.method == "POST":

        if "audio" not in request.files:
            return render_template("index.html")

        audio_file = request.files["audio"]

        if audio_file.filename == "":
            return render_template("index.html")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], audio_file.filename)

        audio_file.save(filepath)

        # Speech to Text
        result = model.transcribe(filepath)
        transcription = result["text"]

        # Sentiment Analysis
        sentiment_result = sentiment_pipeline(transcription)
        sentiment = sentiment_result[0]["label"]

    return render_template(
        "index.html",
        transcription=transcription,
        sentiment=sentiment
    )


if __name__ == "__main__":
    app.run()
