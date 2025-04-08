import os
import re
import json
import time
import streamlit as st
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
DATA_FILE = "candidates.json"
SESSION_TIMEOUT = 300  # 5 minutes for technical questions
MAX_RETRIES = 3

# Initialize session state structure
def init_session_state():
    return {
        "messages": [],
        "current_step": 0,
        "candidate_info": {
            "name": "",
            "email": "",
            "phone": "",
            "years_exp": "",
            "desired_position": "",
            "location": "",
            "tech_stack": [],
            "technical_responses": {},
            "timings": {},
            "sentiment": []
        },
        "tech_questions": [],
        "active_question_index": 0,
        "question_start_time": None,
        "session_start": time.time()
    }

if "session" not in st.session_state:
    st.session_state.session = init_session_state()

# Validation functions
def validate_email(email):
    return re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email)

def validate_phone(phone):
    return re.match(r"^\+?[1-9]\d{1,14}$", phone)

def validate_experience(years):
    try:
        return 0 <= float(years) <= 50
    except ValueError:
        return False

# Sentiment analysis (enhanced dummy implementation)
def analyze_sentiment(text):
    positive = ["great", "excellent", "love", "happy", "thanks"]
    negative = ["hate", "bad", "terrible", "frustrating", "annoying"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive if word in text_lower)
    neg_count = sum(1 for word in negative if word in text_lower)
    
    if pos_count > neg_count: return "positive"
    if neg_count > pos_count: return "negative"
    return "neutral"

# Data handling
def save_candidate_data():
    try:
        data = st.session_state.session["candidate_info"].copy()
        # Redact sensitive information
        data["email"] = "[REDACTED]"
        data["phone"] = "[REDACTED]"
        
        with open(DATA_FILE, "a") as f:
            json.dump(data, f)
            f.write("\n")
    except Exception as e:
        st.error(f"Failed to save data: {str(e)}")

# Question generation with retries
def generate_technical_questions(tech_stack, experience):
    questions = []
    for tech in tech_stack:
        prompt = f"""<s>[INST] Generate 3 technical interview questions about {tech} 
        suitable for a candidate with {experience} years of experience. 
        Format as a numbered list without markdown.[/INST]"""
        
        for _ in range(MAX_RETRIES):
            try:
                response = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 300,
                            "temperature": 0.7,
                            "return_full_text": False
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    questions.append({
                        "technology": tech,
                        "questions": response.json()[0]['generated_text'],
                        "answers": [],
                        "response_times": [],
                        "start_time": None
                    })
                    break
                elif response.status_code == 503:
                    time.sleep(15)  # Wait for model to load
                else:
                    st.error(f"API Error: {response.text}")
                    break
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
                time.sleep(5)
    
    return questions

# UI Components
def render_technical_question_interface():
    col1, col2 = st.columns([3, 1])
    with col1:
        answer = st.text_area(
            "**Write your answer (code supported):**",
            height=200,
            key=f"answer_{st.session_state.session['active_question_index']}",
            help="Write your answer here. You can include code snippets using backticks."
        )
    with col2:
        elapsed = time.time() - st.session_state.session["question_start_time"]
        remaining = max(SESSION_TIMEOUT - elapsed, 0)
        
        st.metric("Time Remaining", f"{int(remaining // 60)}m {int(remaining % 60)}s")
        st.progress(min(elapsed / SESSION_TIMEOUT, 1.0))
        
        if remaining <= 0:
            st.warning("Time's up! Moving to next question.")
            return None
    return answer

def render_sidebar():
    st.sidebar.header("Session Overview")
    st.sidebar.subheader("Progress")
    
    # General progress
    current_step = st.session_state.session["current_step"]
    total_steps = len(steps) + len(st.session_state.session["tech_questions"])
    progress = (current_step / total_steps) if total_steps > 0 else 0
    st.sidebar.progress(progress)
    
    # Timing information
    session_duration = time.time() - st.session_state.session["session_start"]
    st.sidebar.metric("Session Duration", 
                     f"{int(session_duration // 60)}m {int(session_duration % 60)}s")
    
    # Data controls
    if st.sidebar.button("Restart Session"):
        st.session_state.session = init_session_state()
        st.rerun()
    
    st.sidebar.download_button(
        "Export Session Data",
        data=json.dumps(st.session_state.session["candidate_info"], indent=2),
        file_name="interview_data.json",
        mime="application/json"
    )

# Conversation steps
steps = [
    {"prompt": "👋 Hello! I'm TalentScout Hiring Assistant. Let's start with your full name.", 
     "key": "name", "validator": lambda x: len(x) >= 3},
    {"prompt": "📧 What's your email address?", 
     "key": "email", "validator": validate_email},
    {"prompt": "📱 Please share your phone number (international format):", 
     "key": "phone", "validator": validate_phone},
    {"prompt": "⏳ How many years of professional experience do you have?", 
     "key": "years_exp", "validator": validate_experience},
    {"prompt": "💼 What position are you applying for?", 
     "key": "desired_position", "validator": lambda x: len(x) >= 2},
    {"prompt": "📍 Where are you currently located?", 
     "key": "location", "validator": lambda x: len(x) >= 2},
    {"prompt": "🛠️ List your technical expertise (comma-separated):\nExamples: Python, React, AWS", 
     "key": "tech_stack", "validator": lambda x: len(x) >= 1},
]

# Main UI
st.set_page_config(page_title="TalentScout AI Interviewer", page_icon="🤖")
st.title("TalentScout AI Interview Assistant 🤖")
render_sidebar()

# Display chat history
for msg in st.session_state.session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Conversation flow management
current_step = st.session_state.session["current_step"]

# Handle technical questions
if st.session_state.session["tech_questions"]:
    q_idx = st.session_state.session["active_question_index"]
    tech_q = st.session_state.session["tech_questions"][q_idx]
    
    if not tech_q["start_time"]:
        st.session_state.session["tech_questions"][q_idx]["start_time"] = time.time()
        st.session_state.session["question_start_time"] = time.time()
        st.session_state.session["messages"].append({
            "role": "assistant",
            "content": f"**{tech_q['technology']} Questions:**\n{tech_q['questions']}",
            "timestamp": time.time()
        })
        st.rerun()
    
    answer = render_technical_question_interface()
    if answer:
        # Record answer and timing
        tech_q["answers"].append(answer)
        response_time = time.time() - tech_q["start_time"]
        tech_q["response_times"].append(response_time)
        
        # Record sentiment
        sentiment = analyze_sentiment(answer)
        st.session_state.session["candidate_info"]["sentiment"].append({
            "question": tech_q['technology'],
            "answer": answer,
            "sentiment": sentiment
        })
        
        # Move to next question
        st.session_state.session["active_question_index"] += 1
        if st.session_state.session["active_question_index"] >= len(st.session_state.session["tech_questions"]):
            st.session_state.session["messages"].append({
                "role": "assistant",
                "content": "✅ Thank you! We've received all your answers.",
                "timestamp": time.time()
            })
            save_candidate_data()
            time.sleep(2)
            st.session_state.session = init_session_state()
            st.rerun()
        else:
            st.session_state.session["question_start_time"] = time.time()
            st.rerun()

# Handle normal conversation flow
elif current_step < len(steps):
    step = steps[current_step]
    
    # Show current question if not already displayed
    if not any(m["content"] == step["prompt"] for m in st.session_state.session["messages"]):
        st.session_state.session["messages"].append({
            "role": "assistant",
            "content": step["prompt"],
            "timestamp": time.time()
        })
        st.rerun()

# Input handling
if user_input := st.chat_input("Type your response..."):
    # Handle exit commands
    if user_input.lower() in ["exit", "quit", "end"]:
        st.session_state.session["messages"].append({"role": "user", "content": user_input})
        st.session_state.session["messages"].append({
            "role": "assistant",
            "content": "👋 Thank you for your time! Your progress has been saved."
        })
        save_candidate_data()
        st.session_state.session = init_session_state()
        st.rerun()
    
    # Handle technical answers (already handled above)
    elif st.session_state.session["tech_questions"]:
        pass
    
    # Handle normal conversation steps
    else:
        current_step = st.session_state.session["current_step"]
        step = steps[current_step]
        
        # Validate input
        if not step["validator"](user_input):
            error_msg = {
                "name": "Please enter a valid name (min 3 characters)",
                "email": "Invalid email format (example@domain.com)",
                "phone": "Invalid phone number (use international format)",
                "years_exp": "Please enter a valid number (0-50)",
                "desired_position": "Please enter a valid position title",
                "location": "Please enter a valid location",
                "tech_stack": "Please enter at least one technology"
            }.get(step["key"], "Invalid input")
            st.error(error_msg)
        else:
            # Store valid response
            st.session_state.session["candidate_info"][step["key"]] = user_input
            st.session_state.session["messages"].append({
                "role": "user", 
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Record timing
            if current_step > 0:
                prev_step = steps[current_step-1]
                response_time = time.time() - prev_step["timestamp"]
                st.session_state.session["candidate_info"]["timings"][prev_step["key"]] = response_time
            
            # Move to next step
            st.session_state.session["current_step"] += 1
            
            # Generate questions when all info collected
            if st.session_state.session["current_step"] == len(steps):
                tech_stack = [t.strip() for t in 
                    st.session_state.session["candidate_info"]["tech_stack"].split(",")]
                years_exp = st.session_state.session["candidate_info"]["years_exp"]
                
                with st.spinner("🔍 Generating technical questions..."):
                    st.session_state.session["tech_questions"] = generate_technical_questions(
                        tech_stack, years_exp
                    )
                    st.session_state.session["active_question_index"] = 0
                    st.session_state.session["question_start_time"] = time.time()
            
            st.rerun()

# Session timeout handling
if (time.time() - st.session_state.session["session_start"]) > 3600:  # 1 hour timeout
    st.session_state.session["messages"].append({
        "role": "assistant",
        "content": "⏳ Session timed out due to inactivity. Please start a new session."
    })
    save_candidate_data()
    st.session_state.session = init_session_state()
    st.rerun()