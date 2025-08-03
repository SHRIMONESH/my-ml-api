from flask import Flask, request, jsonify
import joblib
import traceback

app = Flask(__name__)

# Load the model safely
try:
    model = joblib.load("fraud_pipeline_2.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    traceback.print_exc()
    model = None  # So we can handle it later

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        print("📨 Received data:", data)

        # Validate expected keys
        required_keys = ["behavior_score", "device_match", "mnrl_flag", "ip_risk_score", "reputation_score"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400

        X = [[
            float(data["behavior_score"]),
            int(data["device_match"]),
            int(data["mnrl_flag"]),
            float(data["ip_risk_score"]),
            float(data["reputation_score"])
        ]]

        prob = model.predict_proba(X)[0][1]

        verdict = "green"
        if prob > 0.75:
            verdict = "red"
        elif prob > 0.4:
            verdict = "yellow"

        return jsonify({ "risk_score": float(prob), "verdict": verdict })

    except Exception as e:
        print("❌ Exception in /predict route:")
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
