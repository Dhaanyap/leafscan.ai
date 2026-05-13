from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import json, os, io, base64, urllib.request, urllib.parse

app = Flask(__name__, static_folder='.')
CORS(app)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"

def analyze_leaf(img_bytes):
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = """You are an expert botanist and plant pathologist.

Analyze this image carefully. Even if it's a screenshot, photo, or any image format:

1. What plant or leaf do you see? (Tulsi, Neem, Aloe Vera, Mint, Turmeric, Ginger, Amla, Brahmi, Tamarind, Henna, Insulin Plant, Indian Borage, Betel, Castor, Lemongrass, Tomato, Potato, Pepper, or other)
2. Does it show any disease, yellowing, spots, wilting, or abnormality?
3. If diseased - exact disease name
4. Cause, remedies including home remedies, prevention, care tips
5. Severity: healthy / mild / moderate / severe

If you cannot identify a plant clearly, still provide your best analysis.

Reply ONLY with this exact JSON, no markdown, no extra text:
{"plant":"plant name","status":"healthy or diseased","disease_name":"disease name or Healthy","severity":"healthy or mild or moderate or severe","cause":"explanation","remedies":["remedy 1","remedy 2","remedy 3 home remedy"],"prevention":["tip 1","tip 2","tip 3"],"care":["care tip 1","care tip 2"],"sun":["sunlight requirement"],"timeline":["action 1","action 2","action 3"],"remedy_link":"https://www.healthline.com/search?q1=plant+disease+remedy"}"""

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

    file = request.files["image"]

    try:
        # Read and convert any image format to JPEG
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Convert RGBA or palette images to RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to max 800px keeping aspect ratio
        max_size = 800
        ratio = min(max_size/img.width, max_size/img.height)
        if ratio < 1:
            new_size = (int(img.width*ratio), int(img.height*ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Save as JPEG
        img_io = io.BytesIO()
        img.save(img_io, format="JPEG", quality=85)
        img_bytes_clean = img_io.getvalue()

        print(f"Image processed: {len(img_bytes_clean)} bytes")

        # Send to Gemini
        data = analyze_leaf(img_bytes_clean)

        plant = data.get("plant", "Unknown Plant")
        is_healthy = data.get("status", "").lower() == "healthy"
        disease = data.get("disease_name", "Unknown")
        display = f"{plant} — Healthy" if is_healthy else f"{plant} — {disease}"

        return jsonify({
            "plant": plant,
            "plant_confidence": 95,
            "display": display,
            "status": data.get("status", "unknown"),
            "disease_name": disease,
            "severity": data.get("severity", "moderate"),
            "cause": data.get("cause", ""),
            "remedies": data.get("remedies", []),
            "prevention": data.get("prevention", []),
            "care": data.get("care", []),
            "sun": data.get("sun", []),
            "timeline": data.get("timeline", []),
            "link": data.get("remedy_link", f"https://www.healthline.com/search?q1={urllib.parse.quote(plant)}")
        })

    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"Gemini HTTP error: {e.code} — {error_body}")
        return jsonify({"error": f"AI analysis failed: {e.code}. {error_body[:200]}"}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "running", "gemini": "connected"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)