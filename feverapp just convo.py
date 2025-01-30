import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define Flask app
flask_app = Flask(__name__)

# Flask route for health check
@flask_app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "Flask server is running!"})

# Flask route for diagnosis analysis
@flask_app.route("/analyze", methods=["POST"])
def analyze_symptoms():
    data = request.json
    if not data or "responses" not in data:
        return jsonify({"error": "Invalid input, 'responses' is required."}), 400

    responses = data["responses"]
    if len(responses) == 0:
        return jsonify({"error": "No responses provided."}), 400

    # Combine responses for analysis
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
    
    # Use LangChain for analysis
    memory = ConversationBufferWindowMemory(k=10)
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768"  # Default model
    )
    conversation = ConversationChain(llm=groq_chat, memory=memory)
    diagnosis = conversation(analysis_prompt)["response"]

    return jsonify({"diagnosis": diagnosis})

# Run Flask server in a separate thread
def run_flask():
    flask_app.run(host="0.0.0.0", port=5002)

# Function to generate follow-up questions based on user input
def generate_followup_questions(user_input):
    prompt = f"""
    The user has described the following symptoms:
    {user_input}

    Based on this, generate up to 8 follow-up questions to narrow down the type of fever they might be experiencing.
    Return the questions as a list.
    """
    
    # Use LangChain to generate questions
    memory = ConversationBufferWindowMemory(k=10)
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768"  # Default model
    )
    conversation = ConversationChain(llm=groq_chat, memory=memory)
    response = conversation(prompt)["response"]
    
    # Extract questions from the response
    questions = [q.strip() for q in response.split("\n") if q.strip()]
    return questions[:8]  # Limit to 8 questions

# Main Streamlit app
def main():
    st.title("Symptom Checker - Fever Analysis Bot")

    # Sidebar for model selection
    st.sidebar.title("Select a Model")
    model = st.sidebar.selectbox(
        "Choose an LLM",
        ["mixtral-8x7b-32768", "llama2-70b-4096"]
    )

    # Conversation memory
    memory = ConversationBufferWindowMemory(k=10)

    # Initialize LangChain Groq model
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    # Initialize session state for tracking
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

    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["message"])

    # Diagnosis complete
    if st.session_state.diagnosis_complete:
        st.write("Diagnosis complete. Restart the bot for a new analysis.")
        return

    # Input for user response
    user_input = st.chat_input("Type your answer here...")
    if user_input:
        # Save the user's response
        st.session_state.responses.append(user_input)

        # Add the user response to the chat history
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_input
        })

        # If initial symptoms are not provided, generate follow-up questions
        if not st.session_state.initial_symptoms_provided:
            st.session_state.followup_questions = generate_followup_questions(user_input)
            st.session_state.initial_symptoms_provided = True
            st.session_state.current_question_index = 0
        else:
            # Move to the next question
            st.session_state.current_question_index += 1

        # If all follow-up questions are answered, send responses to Flask for analysis
        # If all follow-up questions are answered, send responses to Flask for analysis
        if st.session_state.current_question_index == len(st.session_state.followup_questions):
            with st.spinner("Analyzing your responses..."):
                responses = st.session_state.responses
                flask_url = "http://localhost:5002/analyze"
                response = requests.post(flask_url, json={"responses": responses})
                diagnosis = response.json().get("diagnosis", "Could not generate a diagnosis.")

            # Show the diagnosis
            with st.chat_message("assistant"):
                st.write("Based on your symptoms, here is the analysis:")
                st.write(diagnosis)

            # Add the diagnosis to the chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "message": f"Based on your symptoms, here is the analysis: {diagnosis}"
            })

            # Mark the diagnosis as complete
            st.session_state.diagnosis_complete = True

    # Handle the current question
    if not st.session_state.initial_symptoms_provided:
        # Ask for initial symptoms
        with st.chat_message("assistant"):
            st.write("What are the symptoms you are experiencing?")

        # Add the question to the chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": "What are the symptoms you are experiencing?"
        })
    elif st.session_state.current_question_index < len(st.session_state.followup_questions):
        current_question = st.session_state.followup_questions[st.session_state.current_question_index]

        # Show the question
        with st.chat_message("assistant"):
            st.write(current_question)

        # Add the question to the chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": current_question
        })

# Run the Flask server and Streamlit app
if __name__ == "__main__":
    thread = Thread(target=run_flask)
    thread.daemon = True
    thread.start()
    main()