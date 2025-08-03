from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  # Optional: allows cross-origin requests

app = Flask(__name__)
CORS(app)  # Enable CORS

model = joblib.load("fraud_pipeline_2.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X = [[
        data["behavior_score"],
        data["device_match"],
        data["mnrl_flag"],
        data["ip_risk_score"],
        data["reputation_score"]
    ]]
    prob = model.predict_proba(X)[0][1]
    
    verdict = "green"
    if prob > 0.75:
        verdict = "red"
    elif prob > 0.4:
        verdict = "yellow"

    return jsonify({ "risk_score": float(prob), "verdict": verdict })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
