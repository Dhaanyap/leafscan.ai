from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os, io, base64, urllib.request, urllib.parse

app = Flask(__name__, static_folder='.')
CORS(app)

MODEL_PATH     = "plant_model.h5"
CLASS_IDX_PATH = "class_indices.json"
IMG_SIZE       = (224, 224)
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBbeZRaAzJk9FZPbSOzIVR4x7HHnz3MwoU")
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"

print("Loading plant model...")
if os.path.exists(MODEL_PATH):
    plant_model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_IDX_PATH) as f:
        idx_to_class = {v: k for k, v in json.load(f).items()}
    print(f"Model loaded. {len(idx_to_class)} classes.")
else:
    plant_model = None
    idx_to_class = {}
    print("WARNING: plant_model.h5 not found.")


def identify_plant(arr):
    preds = plant_model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return idx_to_class[top_idx], round(float(preds[top_idx]) * 100, 1)


def detect_disease(img_bytes, plant_name):
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = f"""You are an expert plant pathologist.

This is a leaf of {plant_name}.

Look very carefully at this leaf image. Examine it for:
- Yellowing or pale patches
- Brown or black spots
- White powdery coating
- Wilting or curling
- Holes or lesions
- Rust colored patches
- Any abnormality

Then respond with ONLY this JSON (no markdown, no extra text):

If the leaf looks completely healthy:
{{"status":"healthy","disease_name":"Healthy","severity":"healthy","cause":"The {plant_name} leaf appears healthy with no visible disease symptoms.","remedies":["No treatment needed","Continue regular watering","Monitor weekly for early signs"],"prevention":["Water at base only","Ensure good airflow around plant","Avoid overwatering"],"care":["Water every 2 days","Fertilize monthly with compost","Remove dead leaves promptly"],"sun":["Provide 6-8 hours of direct sunlight daily"],"timeline":["No action needed","Monitor weekly","Reassess monthly"]}}

If the leaf shows disease signs:
{{"status":"diseased","disease_name":"EXACT disease name here","severity":"mild or moderate or severe","cause":"Exact cause of this disease","remedies":["Remedy 1","Remedy 2","Remedy 3 home remedy"],"prevention":["Prevention tip 1","Prevention tip 2","Prevention tip 3"],"care":["Care tip 1","Care tip 2"],"sun":["Sunlight requirement"],"timeline":["Day 1 action","Week 1 action","Week 2 action"]}}"""

    payload = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
            {"text": prompt}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024}
    }).encode("utf-8")

    req = urllib.request.Request(
        GEMINI_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"Gemini raw response: {text[:200]}")
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    if plant_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    file = request.files["image"]
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Stage 1 — identify plant (internal only)
        arr = np.expand_dims(np.array(img.resize(IMG_SIZE)) / 255.0, axis=0)
        plant_name, plant_conf = identify_plant(arr)
        print(f"Plant identified: {plant_name} ({plant_conf}%)")

        # Compress for Gemini
        img_io = io.BytesIO()
        img.resize((512, 512)).save(img_io, format="JPEG", quality=80)

        # Stage 2 — detect disease with Gemini
        print("Sending to Gemini...")
        disease = detect_disease(img_io.getvalue(), plant_name)

        if disease is None:
            return jsonify({
                "display": "Analysis Failed",
                "status": "unknown",
                "disease_name": "Could not analyze",
                "severity": "moderate",
                "cause": "Gemini could not analyze the image. Please try again.",
                "remedies": ["Retry with a clearer image"],
                "prevention": ["Use bright natural lighting"],
                "care": ["Ensure leaf fills the frame"],
                "sun": ["Take photo in daylight"],
                "timeline": ["Retry immediately"],
                "link": "https://plantix.net",
                "plant": plant_name,
                "plant_confidence": plant_conf
            })

        is_healthy = disease.get("status", "").lower() == "healthy"
        disease_name = disease.get("disease_name", "Unknown")

        # Display — focus on health status not plant name
        if is_healthy:
            display = f"✅ {plant_name} — Healthy"
        else:
            display = f"⚠️ {plant_name} — {disease_name}"

        print(f"Result: {display} | Severity: {disease.get('severity')}")

        return jsonify({
            "plant": plant_name,
            "plant_confidence": plant_conf,
            "display": display,
            "status": disease.get("status"),
            "disease_name": disease_name,
            "severity": disease.get("severity", "moderate"),
            "cause": disease.get("cause", ""),
            "remedies": disease.get("remedies", []),
            "prevention": disease.get("prevention", []),
            "care": disease.get("care", []),
            "sun": disease.get("sun", []),
            "timeline": disease.get("timeline", []),
            "link": f"https://plantix.net"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "running", "model": plant_model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)