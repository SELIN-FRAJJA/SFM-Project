import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import requests
from geopy.geocoders import Nominatim
import pandas as pd
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define Flask app
flask_app = Flask(__name__)

@flask_app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "Flask server is running!"})

@flask_app.route("/analyze", methods=["POST"])
def analyze_symptoms():
    data = request.json
    if not data or "responses" not in data:
        return jsonify({"error": "Invalid input, 'responses' is required."}), 400

    responses = data["responses"]
    if len(responses) == 0:
        return jsonify({"error": "No responses provided."}), 400

    summary = "\n".join([f"Response {i+1}: {r}" for i, r in enumerate(responses)])
    analysis_prompt = f"""
    Based on the following symptoms, what kind of fever could this person have? 
    Provide a detailed diagnosis including:
    1. Specific diseases that could be causing the fever.
    2. Recommended remedies.
    3. General recommendations for the patient.

    Symptoms:
    {summary}
    """
    
    memory = ConversationBufferWindowMemory(k=10)
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768"
    )
    conversation = ConversationChain(llm=groq_chat, memory=memory)
    diagnosis = conversation(analysis_prompt)["response"]

    return jsonify({"diagnosis": diagnosis})

def run_flask():
    flask_app.run(host="0.0.0.0", port=5002)

def geocode_city(city_name):
    geolocator = Nominatim(user_agent="symptom_checker")
    try:
        location = geolocator.geocode(city_name, exactly_one=True, addressdetails=True)
        if location:
            return {
                "lat": location.latitude,
                "lon": location.longitude,
                "bbox": list(map(float, location.raw.get('boundingbox', [])))
            }
        return None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None

def get_hospitals(city_name):
    location_info = geocode_city(city_name)
    if not location_info:
        return None
    
    bbox = location_info.get('bbox', [])
    if len(bbox) < 4:
        return None
    
    south, north, west, east = bbox[0], bbox[1], bbox[2], bbox[3]
    
    overpass_query = f"""
    [out:json];
    node["amenity"="hospital"]({south},{west},{north},{east});
    out body;
    """
    
    try:
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={'data': overpass_query},
            timeout=10
        )
        if response.status_code == 200:
            hospitals = []
            for element in response.json().get('elements', []):
                tags = element.get('tags', {})
                name = tags.get('name', '').lower()
                healthcare = tags.get('healthcare', '').lower()

                # Skip veterinary and fertility-related hospitals
                exclude_keywords = ['veterinary', 'fertility', 'ivf', 'reproductive']
                if (any(kw in name for kw in exclude_keywords) 
                    or any(kw in healthcare for kw in exclude_keywords)):
                    continue

                address = tags.get('addr:full', 'Address not available')
                if 'lat' in element and 'lon' in element:
                    hospitals.append({
                        'name': tags.get('name', 'Unnamed Hospital'),
                        'address': address,
                        'lat': element['lat'],
                        'lon': element['lon']
                    })
            return hospitals[:10]
        return None
    except Exception as e:
        st.error(f"Overpass API error: {e}")
        return None

def generate_followup_questions(user_input):
    prompt = f"""
    The user has described the following symptoms:
    {user_input}

    Based on this, generate up to 8 follow-up questions to narrow down the type of fever they might be experiencing.
    Return the questions as a numbered list.
    """
    
    memory = ConversationBufferWindowMemory(k=10)
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768"
    )
    conversation = ConversationChain(llm=groq_chat, memory=memory)
    response = conversation(prompt)["response"]
    
    questions = [q.split('. ', 1)[1].strip() for q in response.split("\n") if q.strip().startswith(tuple(str(i) for i in range(1,9)))]
    return questions[:8]

def main():
    st.title("Symptom Checker - Fever Analysis Bot")

    st.sidebar.title("Select a Model")
    model = st.sidebar.selectbox(
        "Choose an LLM",
        ["mixtral-8x7b-32768", "llama2-70b-4096"]
    )

    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "diagnosis_complete" not in st.session_state:
        st.session_state.diagnosis_complete = False
    if "followup_questions" not in st.session_state:
        st.session_state.followup_questions = []
    if "initial_symptoms_provided" not in st.session_state:
        st.session_state.initial_symptoms_provided = False
    if "location" not in st.session_state:
        st.session_state.location = None
    if "location_provided" not in st.session_state:
        st.session_state.location_provided = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["message"])

    if not st.session_state.location_provided:
        if "city_prompt_displayed" not in st.session_state:
            with st.chat_message("assistant"):
                st.write("Enter your current city loaction...")
            st.session_state.chat_history.append({"role": "assistant", "message": "Enter your city."})
            st.session_state.city_prompt_displayed = True

        user_input = st.chat_input("Type your city name here...")
        if user_input:
            st.session_state.location = user_input
            st.session_state.location_provided = True
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.chat_history.append({"role": "user", "message": user_input})

            with st.chat_message("assistant"):
                st.write("What are the symptoms you are experiencing?")
            st.session_state.chat_history.append({"role": "assistant", "message": "What are the symptoms you are experiencing?"})
            st.session_state.symptom_question_asked = True 
            st.rerun()
        return

    if st.session_state.diagnosis_complete:
        st.write("Diagnosis complete. Restart the bot for a new analysis.")
        if st.session_state.location:
            st.subheader(f"Hospitals near {st.session_state.location}")
            hospitals = get_hospitals(st.session_state.location)
            if hospitals:
                for idx, hospital in enumerate(hospitals, 1):
                    st.write(f"{idx}. **{hospital['name']}**  \n{hospital['address']}")
                df = pd.DataFrame(hospitals)
                st.map(df[['lat', 'lon']])
            else:
                st.write("No hospitals found in the specified area.")
        return

    user_input = st.chat_input("Type your answer here...")
    if user_input:
        st.session_state.responses.append(user_input)
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        if not st.session_state.initial_symptoms_provided:
            st.session_state.followup_questions = generate_followup_questions(user_input)
            st.session_state.initial_symptoms_provided = True
            st.session_state.current_question_index = 0
        else:
            st.session_state.current_question_index += 1

        if st.session_state.current_question_index >= len(st.session_state.followup_questions):
            with st.spinner("Analyzing your responses..."):
                max_retries = 5
                for _ in range(max_retries):
                    try:
                        response = requests.get("http://localhost:5002/health")
                        if response.status_code == 200:
                            break
                    except:
                        time.sleep(1)
                else:
                    st.error("Flask server is not responding.")
                    return

                flask_url = "http://localhost:5002/analyze"
                response = requests.post(flask_url, json={"responses": st.session_state.responses})
                if response.status_code == 200:
                    diagnosis = response.json().get("diagnosis", "Could not generate a diagnosis.")
                else:
                    diagnosis = "Error in diagnosis analysis."

            with st.chat_message("assistant"):
                st.write("Based on your symptoms, here is the analysis:")
                st.write(diagnosis)
            st.session_state.chat_history.append({
                "role": "assistant",
                "message": f"Based on your symptoms, here is the analysis: {diagnosis}"
            })
            st.session_state.diagnosis_complete = True
            st.rerun()
        else:
            st.rerun()

    if (st.session_state.location_provided 
        and not st.session_state.diagnosis_complete 
        and st.session_state.initial_symptoms_provided
        and st.session_state.current_question_index < len(st.session_state.followup_questions)):
        
        current_question = st.session_state.followup_questions[st.session_state.current_question_index]
        with st.chat_message("assistant"):
            st.write(current_question)
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": current_question
        })

if __name__ == "__main__":
    thread = Thread(target=run_flask)
    thread.daemon = True
    thread.start()
    main()