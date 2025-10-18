from flask import Flask, render_template, request, jsonify
from chat2 import llm, setup_retrieval_qa
import os
import requests
import sqlite3
from database import create_tables
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Milvus Config
MILVUS_URI = "https://in03-c3450588c0a2321.serverless.aws-eu-central-1.cloud.zilliz.com"
MILVUS_TOKEN = "5d587a55df90f60547f33af66bf12f2f6a46ea97dce29b3ad5067bd30e9c097daf648c22b860d4bca4fa8ce85540434beee6cbef"
COLLECTION_NAME = "vayazh"

app = Flask(__name__)
create_tables()

# ✅ Connect to existing Milvus collection
def get_milvus_vector_store():
    try:
        model_path = "sentence-transformers/all-mpnet-base-v2"  # ✅ ensure same model (768 dims)
        _ = SentenceTransformer(model_path)

        embedding_function = HuggingFaceEmbeddings(model_name=model_path)

        print("✅ Connecting to Milvus collection:", COLLECTION_NAME)
        db = Milvus(
            embedding_function,
            collection_name=COLLECTION_NAME,
            connection_args={
                "uri": MILVUS_URI,
                "token": MILVUS_TOKEN
            },
            text_field="text",
            vector_field="embedding"
        )
        return db
    except Exception as e:
        print("❌ Error connecting to Milvus:", str(e))
        raise e

# ✅ Get DB connection
db = get_milvus_vector_store()
chain = setup_retrieval_qa(db)

API_KEY = "d91a58b4ef77f5f11498e31e4ad2d756"  # Replace with your OpenWeatherMap API Key

# ---------------- Farmer DB Functions ----------------
def store_farmer_to_db(data):
    conn = sqlite3.connect("farmers.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO farmer_details (location, landSize, soilType, irrigationMethod, waterSource)
        VALUES (?, ?, ?, ?, ?)
    """, (data["location"], data["landSize"], data["soilType"], data["irrigationMethod"], data["waterSource"]))
    conn.commit()
    conn.close()

def get_farmer_details():
    conn = sqlite3.connect("farmers.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM farmer_details ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "location": row[1],
            "landSize": row[2],
            "soilType": row[3],
            "irrigationMethod": row[4],
            "waterSource": row[5]
        }
    return {}

def store_chat_history(farmer_id, question, answer):
    conn = sqlite3.connect("farmers.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (farmer_id, question, answer)
        VALUES (?, ?, ?)
    """, (farmer_id, question, answer))
    conn.commit()
    conn.close()

# ---------------- Weather Functions ----------------
def get_weather_by_latlon(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()

def get_weather_by_city(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()

def parse_weather_response(data, location=""):
    if data.get("cod") != 200:
        return "❌ Error: " + data.get("message", "Unable to fetch weather."), None

    weather_data = {
        "description": data["weather"][0]["description"].capitalize(),
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"]
    }

    weather_info = (
        f"🌦️ **Weather in {location if location else 'your area'}**:\n"
        f"- Condition: {weather_data['description']}\n"
        f"- Temperature: {weather_data['temp']}°C\n"
        f"- Humidity: {weather_data['humidity']}%\n"
        f"- Wind Speed: {weather_data['wind_speed']} m/s"
    )
    return weather_info, weather_data

def get_forecast_by_latlon(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()

def get_forecast_by_city(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},IN&appid={API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()

def parse_forecast_response(data, location=""):
    if data.get("cod") != "200":
        return "❌ Error: " + data.get("message", "Unable to fetch forecast."), None

    forecast_summary = {}
    for item in data["list"]:
        date = item["dt_txt"].split()[0]
        temp = item["main"]["temp"]
        condition = item["weather"][0]["description"]

        if date not in forecast_summary:
            forecast_summary[date] = {"temps": [], "conditions": []}

        forecast_summary[date]["temps"].append(temp)
        forecast_summary[date]["conditions"].append(condition)

    summary_lines = []
    for i, (date, details) in enumerate(forecast_summary.items()):
        if i >= 3: break
        avg_temp = round(sum(details["temps"]) / len(details["temps"]), 1)
        most_common_condition = max(set(details["conditions"]), key=details["conditions"].count)
        summary_lines.append(f"📅 {date}: {avg_temp}°C, {most_common_condition.capitalize()}")

    forecast_text = "\n".join(summary_lines)
    return f"📈 **3-Day Forecast for {location if location else 'your area'}**\n{forecast_text}", forecast_summary

# ---------------- Personalized Prompt ----------------
def prepare_personalized_prompt(query, farmer_details, weather_data):
    context = []

    if farmer_details:
        context.append(
            f"Farm Details: Location: {farmer_details.get('location', 'unknown')}, "
            f"Land Size: {farmer_details.get('landSize', 'unknown')} acres, "
            f"Soil Type: {farmer_details.get('soilType', 'unknown')}, "
            f"Irrigation: {farmer_details.get('irrigationMethod', 'unknown')}, "
            f"Water Source: {farmer_details.get('waterSource', 'unknown')}."
        )

    if weather_data:
        context.append(
            f"Current Weather: {weather_data['description']}, "
            f"Temperature: {weather_data['temp']}°C, "
            f"Humidity: {weather_data['humidity']}%, "
            f"Wind Speed: {weather_data['wind_speed']} m/s."
        )

    return f"{' '.join(context)}\nQuery: {query}"

# ---------------- Flask Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['messageText'].strip()

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if query.lower() in greetings:
        farmer_details = get_farmer_details()
        location = farmer_details.get('location', '')
        return jsonify({
            "answer": f"Hello from {location}! How can I assist with your {farmer_details.get('soilType', 'a farm')} today?"
        })

    if query.lower() in ["who developed you?", "who created you?", "who made you?"]:
        return jsonify({"answer": "I was developed by Team Sapphire."})

    farmer_details = get_farmer_details()
    location = farmer_details.get("location", "")

    # Weather (fallback: lat/lon → city)
    weather_info, weather_data = ("Weather data not available.", None)
    if location:
        data = get_weather_by_city(location)
        weather_info, weather_data = parse_weather_response(data, location)

    personalized_query = prepare_personalized_prompt(query, farmer_details, weather_data)

    response = chain.invoke({"query": personalized_query})

    final_answer = response['result'] if response and response['result'].strip().lower() not in ["don't know.", "i don't know"] else \
        f"I'm here to help with agriculture-related questions for your {farmer_details.get('landSize', 'unknown')} acre farm in {farmer_details.get('location', 'unknown')}. Please ask me about farming, crops, soil, or related topics!"

    store_chat_history(farmer_details.get("id"), query, final_answer)

    return jsonify({"answer": final_answer})

@app.route('/store_farmer_details', methods=['POST'])
def store_farmer_details():
    try:
        data = request.json
        required_fields = ['location', 'landSize', 'soilType', 'irrigationMethod', 'waterSource']
        if not all(data.get(field) for field in required_fields):
            return jsonify({"message": "Missing required farm details."}), 400
        store_farmer_to_db(data)
        return jsonify({"message": "Farm details saved successfully."})
    except Exception as e:
        return jsonify({"message": f"Error saving farm details: {str(e)}"}), 500

@app.route('/chat_history')
def view_chat_history():
    conn = sqlite3.connect("farmers.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT 20
    """)
    rows = cursor.fetchall()
    conn.close()
    return jsonify({"chat_history": rows})

@app.route('/get_weather', methods=['POST'])
def fetch_weather():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    location = data.get("location", "").strip() or get_farmer_details().get("location", "")

    if lat and lon:
        data = get_weather_by_latlon(lat, lon)
        weather_response, weather_data = parse_weather_response(data)
    elif location:
        data = get_weather_by_city(location)
        weather_response, weather_data = parse_weather_response(data, location)
    else:
        return jsonify({"answer": "❌ Please provide a location or enable GPS."})

    return jsonify({"answer": weather_response, "weatherData": weather_data})

@app.route('/get_forecast', methods=['POST'])
def fetch_forecast():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    location = data.get("location", "").strip() or get_farmer_details().get("location", "")

    if lat and lon:
        data = get_forecast_by_latlon(lat, lon)
        forecast_response, forecast_data = parse_forecast_response(data)
    elif location:
        data = get_forecast_by_city(location)
        forecast_response, forecast_data = parse_forecast_response(data, location)
    else:
        return jsonify({"answer": "❌ Please provide a location or enable GPS."})

    return jsonify({"answer": forecast_response, "forecastData": forecast_data})

if __name__ == "__main__":
    app.run(debug=True)
