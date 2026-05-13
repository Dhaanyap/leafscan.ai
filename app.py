from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import json, os, io, base64, urllib.request, urllib.parse

app = Flask(__name__, static_folder='.')
CORS(app)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBbeZRaAzJk9FZPbSOzIVR4x7HHnz3MwoU")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"

def analyze_leaf(img_bytes):
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = """You are an expert botanist and plant pathologist.

Analyze this leaf image carefully and provide:
1. What medicinal plant is this? (Tulsi, Neem, Aloe Vera, Mint, Turmeric, Ginger, Amla, Brahmi, Tamarind, Henna, Insulin Plant, Indian Borage, Betel, Castor, Lemongrass, or Unknown)
2. Is it healthy or diseased?
3. If diseased, exact disease name
4. Cause, remedies, prevention, care tips

Reply ONLY with this JSON, no markdown:
{
  "plant": "plant name",
  "status": "healthy or diseased",
  "disease_name": "disease name or Healthy",
  "severity": "healthy or mild or moderate or severe",
  "cause": "cause explanation",
  "remedies": ["remedy 1", "remedy 2", "remedy 3"],
  "prevention": ["tip 1", "tip 2", "tip 3"],
  "care": ["care tip 1", "care tip 2"],
  "sun": ["sunlight requirement"],
  "timeline": ["action 1", "action 2", "action 3"]
}"""

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
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_io = io.BytesIO()
        img.resize((512, 512)).save(img_io, format="JPEG", quality=80)

        data = analyze_leaf(img_io.getvalue())

        plant = data.get("plant", "Unknown")
        is_healthy = data.get("status", "") == "healthy"
        disease = data.get("disease_name", "Unknown")
        display = f"{plant} — Healthy" if is_healthy else f"{plant} — {disease}"

        return jsonify({
            "plant": plant,
            "plant_confidence": 95,
            "display": display,
            "status": data.get("status"),
            "disease_name": disease,
            "severity": data.get("severity", "moderate"),
            "cause": data.get("cause", ""),
            "remedies": data.get("remedies", []),
            "prevention": data.get("prevention", []),
            "care": data.get("care", []),
            "sun": data.get("sun", []),
            "timeline": data.get("timeline", []),
            "link": f"https://www.healthline.com/search?q1={urllib.parse.quote(plant)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)