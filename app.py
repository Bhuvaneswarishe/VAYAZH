from flask import Flask, render_template, request, jsonify
from chat1 import fetch_website_content, extract_pdf_text, initialize_vector_store
from chat2 import llm, setup_retrieval_qa
import os
import requests
import sqlite3
from database import create_tables

app = Flask(__name__)
create_tables()

# Example URLs and PDF files
urls = ["https://mospi.gov.in/4-agricultural-statistics"]
pdf_files = ["Farming Schemes (1).pdf", "farmerbook (1).pdf"]

# Fetch content from websites
website_contents = [fetch_website_content(url) for url in urls]
pdf_texts = [extract_pdf_text(pdf_file) for pdf_file in pdf_files]

# Initialize the vector store
db = initialize_vector_store(website_contents + pdf_texts)
chain = setup_retrieval_qa(db)

API_KEY = "******"  # Replace with your OpenWeatherMap API Key

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

def get_weather(location):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data["cod"] != 200:
            return "❌ Error: " + data.get("message", "Unable to fetch weather."), None

        weather_data = {
            "description": data["weather"][0]["description"].capitalize(),
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }

        weather_info = (
            f"🌦️ **Weather in {location.capitalize()}**:\n"
            f"- Condition: {weather_data['description']}\n"
            f"- Temperature: {weather_data['temp']}°C\n"
            f"- Humidity: {weather_data['humidity']}%\n"
            f"- Wind Speed: {weather_data['wind_speed']} m/s"
        )

        return weather_info, weather_data

    except Exception as e:
        return "❌ Error fetching weather data.", None

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
    weather_info, weather_data = get_weather(location) if location else ("Weather data not available.", None)

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
    location = data.get("location", "").strip() or get_farmer_details().get("location", "")

    if not location:
        return jsonify({"answer": "❌ Please provide a location."})

    weather_response, weather_data = get_weather(location)

    if weather_data:
        temp = weather_data["temp"]
        humidity = weather_data["humidity"]

        advice = []
        if temp > 30:
            advice.append("🚜 **Farm Advice**: High temperatures detected. Consider increasing irrigation and shading crops.")
        elif temp < 10:
            advice.append("🚜 **Farm Advice**: Cold temperatures detected. Protect crops from frost and reduce irrigation.")

        if humidity > 80:
            advice.append("High humidity increases disease risk. Monitor for fungal infections and improve ventilation.")
        elif humidity < 30:
            advice.append("Low humidity may cause water stress. Consider mulching to retain soil moisture.")

        weather_response += "\n\n" + "\n".join(advice)

    return jsonify({"answer": weather_response, "weatherData": weather_data})

if __name__ == "__main__":
    app.run(debug=True)
