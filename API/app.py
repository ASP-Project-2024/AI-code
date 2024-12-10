from flask import Flask, request, jsonify
from summarization_openai import analyze_audio
from pathlib import Path

app = Flask(__name__)

UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


@app.route("/api/process_audio", methods=["POST"])
def process_audio():
    try:
        if "audio_file" not in request.files:
            return jsonify({"error": {"No audio files provided"}}), 400
        audio_file = request.files["audio_file"]
        file_path = UPLOAD_FOLDER / audio_file.filename
        audio_file.save(file_path)

        # Run the summarization pipeline
        summary = analyze_audio(str(file_path))

        # Prepare response data
        response_data = {
            "summary": summary.summary,
            "key_points": summary.key_points,
            "topics_discussed": summary.topics_discussed,
            "duration": summary.duration,
            "transcript": summary.transcript,
            "confidence_score": summary.confidence_score,
            "speaker_count": summary.speaker_count,
        }

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Super simple route to tell when the api is listening
@app.route("/api/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True)

