from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess
import pandas as pd

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

status = {
    "running": False,
    "done": False,
    "error": False,
    "log": []
}

# User → store mapping.
# In real project this comes from DB; here stores 1-10 are assigned per user.
USERS = {
    "admin":   {"password": "admin123",  "role": "admin",  "store_id": None},
    "viewer":  {"password": "viewer123", "role": "viewer", "store_id": 1},
    "store2":  {"password": "store2pw",  "role": "viewer", "store_id": 2},
    "store5":  {"password": "store5pw",  "role": "viewer", "store_id": 5},
    "store10": {"password": "store10pw", "role": "viewer", "store_id": 10},
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _graph_exists(filename):
    return os.path.exists(os.path.join(GRAPH_DIR, filename))


def _load_csv(filename):
    path = os.path.join(GRAPH_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ─── STATIC ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


# ─── AUTH ─────────────────────────────────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    user = USERS.get(username)
    if user and user["password"] == password:
        return jsonify({
            "ok": True,
            "username": username,
            "role": user["role"],
            "store_id": user["store_id"]
        })

    return jsonify({"ok": False, "message": "Invalid username or password"})


# ─── UPLOAD (admin only) ──────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    train = request.files.get("train")
    store = request.files.get("store")

    if not train or not store:
        return jsonify({"ok": False, "message": "train.csv and store.csv are required"})

    train.save(os.path.join(BASE_DIR, "train.csv"))
    store.save(os.path.join(BASE_DIR, "store.csv"))

    return jsonify({"ok": True, "message": "Files uploaded successfully"})


# ─── ANALYSIS (admin only) ────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    global status

    status = {
        "running": True,
        "done": False,
        "error": False,
        "log": ["Analysis started"]
    }

    try:
        result = subprocess.run(
            ["python", "analyse.py"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            status["log"].append("Analysis completed successfully")
            status["running"] = False
            status["done"] = True
            return jsonify({"ok": True})
        else:
            status["log"].append(result.stderr)
            status["running"] = False
            status["error"] = True
            return jsonify({"ok": False, "message": result.stderr})

    except Exception as e:
        status["log"].append(str(e))
        status["running"] = False
        status["error"] = True
        return jsonify({"ok": False, "message": str(e)})


@app.route("/api/status")
def get_status():
    return jsonify(status)


# ─── ADMIN: MODEL METRICS ─────────────────────────────────────────────────────

@app.route("/api/metrics")
def get_metrics():
    df = _load_csv("model_results.csv")
    if df is None:
        return jsonify({})

    result = {}
    for _, row in df.iterrows():
        metric = row["Metric"]
        value = round(float(row["Value"]), 2)
        result[metric] = value

    return jsonify(result)


# ─── ADMIN: KPIs ──────────────────────────────────────────────────────────────

@app.route("/api/kpis")
def get_kpis():
    df = _load_csv("kpi_results.csv")
    if df is None:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


# ─── ADMIN: INSIGHTS ──────────────────────────────────────────────────────────

@app.route("/api/insights")
def get_insights():
    df = _load_csv("ai_insights.csv")
    if df is None:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


# ─── ADMIN: INVENTORY ────────────────────────────────────────────────────────

@app.route("/api/inventory")
def get_inventory():
    df = _load_csv("inventory_results.csv")
    if df is None:
        return jsonify([])
    return jsonify(df.head(20).to_dict(orient="records"))


# ─── ADMIN: COMPETITION ───────────────────────────────────────────────────────

@app.route("/api/competition")
def get_competition():
    df = _load_csv("competition_analysis.csv")
    if df is None:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


# ─── ADMIN: STORE TYPES ───────────────────────────────────────────────────────

@app.route("/api/store_types")
def get_store_types():
    df = _load_csv("store_type_stats.csv")
    if df is None:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


# ─── ADMIN: STORE RANKING ─────────────────────────────────────────────────────

@app.route("/api/store_ranking")
def get_store_ranking():
    df = _load_csv("store_ranking.csv")
    if df is None:
        return jsonify([])
    return jsonify(df.head(30).to_dict(orient="records"))


# ─── STORE: SUMMARY ───────────────────────────────────────────────────────────

@app.route("/api/store/<int:store_id>/summary")
def store_summary(store_id):
    df = _load_csv("store_summaries.csv")
    if df is None:
        return jsonify({})
    row = df[df["Store"] == store_id]
    return jsonify({} if row.empty else row.iloc[0].to_dict())


@app.route("/api/store/<int:store_id>/monthly")
def store_monthly(store_id):
    df = _load_csv("store_monthly.csv")
    if df is None:
        return jsonify([])
    return jsonify(df[df["Store"] == store_id].to_dict(orient="records"))


@app.route("/api/store/<int:store_id>/promo")
def store_promo(store_id):
    df = _load_csv("store_promo.csv")
    if df is None:
        return jsonify({})
    row = df[df["Store"] == store_id]
    return jsonify({} if row.empty else row.iloc[0].to_dict())


@app.route("/api/store/<int:store_id>/inventory")
def store_inventory(store_id):
    df = _load_csv("inventory_results.csv")
    if df is None:
        return jsonify({})
    row = df[df["Store"] == store_id]
    return jsonify({} if row.empty else row.iloc[0].to_dict())


@app.route("/api/store/<int:store_id>/insights")
def store_insights(store_id):
    df = _load_csv("store_insights.csv")
    if df is None:
        return jsonify([])
    return jsonify(df[df["Store"] == store_id].to_dict(orient="records"))


@app.route("/api/store/<int:store_id>/forecast")
def store_forecast(store_id):
    df = _load_csv("store_forecast.csv")
    if df is None:
        return jsonify([])
    return jsonify(df[df["Store"] == store_id].to_dict(orient="records"))


@app.route("/api/store/<int:store_id>/dow")
def store_dow(store_id):
    df = _load_csv("store_dow.csv")
    if df is None:
        return jsonify([])
    return jsonify(df[df["Store"] == store_id].to_dict(orient="records"))


# ─── GRAPHS ───────────────────────────────────────────────────────────────────

@app.route("/api/graphs/<filename>")
def get_graph(filename):
    return send_from_directory(GRAPH_DIR, filename)


if __name__ == "__main__":
    app.run(port=5050, debug=True)