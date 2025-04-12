import os
import re
import json
import time
import html
import streamlit as st
import requests
from dotenv import load_dotenv
from functools import wraps

# Load environment variables
load_dotenv()

# Configuration constants
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
DATA_FILE = "candidates.json"
MAX_RETRIES = 3
MAX_INVALID_ATTEMPTS = 2
INITIAL_BACKOFF = 5  # seconds

# Position categories and attributes remain as before
POSITION_CATEGORIES = {
    "Technical": [
        "Software Engineer", "Data Scientist", "DevOps Engineer",
        "Frontend Developer", "Backend Developer", "Full Stack Developer"
    ],
    "Non-Technical": [
        "CEO", "Manager", "HR Executive",
        "Security Guard", "Cleaner", "Watchman",
        "Receptionist", "Driver", "Office Assistant"
    ]
}

POSITION_ATTRIBUTES = {
    "Software Engineer": {
        "skills": ["Python", "Java", "C++", "Algorithms", "System Design"],
        "question_type": "technical"
    },
    "Data Scientist": {
        "skills": ["Python", "SQL", "Machine Learning", "Statistics", "Pandas"],
        "question_type": "technical"
    },
    "CEO": {
        "skills": ["Leadership", "Strategy", "Decision Making"],
        "question_type": "behavioral"
    },
    "Security Guard": {
        "skills": ["Vigilance", "Emergency Response", "Observation"],
        "question_type": "situational"
    },
    "Cleaner": {
        "skills": ["Attention to Detail", "Time Management", "Chemical Safety"],
        "question_type": "procedural"
    }
}

# --- Caching Decorator for API Responses ---
def cache_api_call(func):
    cache = {}
    @wraps(func)
    def wrapper(prompt):
        if prompt in cache:
            return cache[prompt]
        result = func(prompt)
        cache[prompt] = result
        return result
    return wrapper

@cache_api_call
def call_hf_api(prompt):
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
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
                data = response.json()
                if data and isinstance(data, list) and "generated_text" in data[0]:
                    return data[0]['generated_text']
                else:
                    st.warning("Received empty API response. Retrying...")
            elif response.status_code == 503:
                st.warning("Service unavailable. Waiting for model to load...")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
            time.sleep(backoff)
            backoff *= 2  # exponential backoff
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
            time.sleep(backoff)
            backoff *= 2
    return ""  # Fallback: return empty string after MAX_RETRIES

# --- Session State Initialization ---
def init_session_state():
    return {
        "messages": [],
        "current_step": 0,
        "invalid_attempts": {},
        "candidate_info": {
            "name": "",
            "email": "",
            "phone": "",
            "years_exp": "",
            "desired_position": "",
            "position_type": "",  # Optional, can be inferred
            "location": "",
            "skills": [],
            "responses": {},
            "sentiment": []
        },
        "questions": [],
        "active_question_index": 0
    }

if "session" not in st.session_state:
    st.session_state.session = init_session_state()

# --- Validation Functions ---
def validate_email(email):
    return re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email)

def validate_phone(phone):
    return re.match(r"^\+?[1-9]\d{1,14}$", phone)

def validate_experience(years):
    try:
        return 0 <= float(years) <= 50
    except ValueError:
        return False

# --- Generate Interview Questions ---
def generate_questions(position, experience, skills):
    questions = []
    position_attrs = POSITION_ATTRIBUTES.get(position, {})  # Allow custom position if not present
    question_type = position_attrs.get("question_type", "general")
    position_skills = position_attrs.get("skills", [])
    
    combined_skills = list(set(skills + position_skills))
    experience_level = "junior" if float(experience) < 3 else "mid-level" if float(experience) < 5 else "senior"
    
    if question_type == "technical":
        prompt = (f"Generate 3 detailed technical interview questions for a {experience_level} {position} role. "
                  f"Focus on these skills: {', '.join(combined_skills)}. "
                  "Present each question as a numbered list.")
    elif question_type == "behavioral":
        prompt = (f"Generate 3 detailed behavioral interview questions for a {position} role. "
                  "Emphasize leadership and management skills. Present the questions as a numbered list.")
    elif question_type == "situational":
        prompt = (f"Generate 3 detailed situational interview questions for a {position} role. "
                  "Focus on real-world scenarios the candidate may face. Present as a numbered list.")
    else:
        prompt = (f"Generate 3 detailed interview questions for a {experience_level} {position} role. "
                  f"Consider these skills: {', '.join(combined_skills)}. Present each question as a numbered list.")
    
    # Call API and handle empty response
    result = call_hf_api(prompt)
    if result.strip() == "":
        st.error("Unable to generate interview questions after multiple attempts. Please try again later.")
    else:
        questions.append({
            "position": position,
            "experience_level": experience_level,
            "question_type": question_type,
            "questions": result,
            "answers": []
        })
    return questions

# --- UI Components ---
def render_question_interface():
    answer = st.text_area(
        "**Write your answer (Character Count: {len_answer})**".format(
            len_answer=len(st.session_state.get("user_answer", ""))
        ),
        height=200,
        key=f"answer_{st.session_state.session['active_question_index']}",
        help="Provide your detailed answer here. Type 'skip' to skip this question."
    )
    return answer

def render_sidebar():
    st.sidebar.header("Session Overview")
    st.sidebar.subheader("Progress")
    
    # Improved restart confirmation flow
    if st.sidebar.button("Restart Session"):
        with st.sidebar:
            st.warning("Are you sure you want to restart?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm Restart"):
                    with st.spinner("Starting new session..."):
                        st.session_state.session = init_session_state()
                        time.sleep(0.5)  # Give visual feedback
                        st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    pass  # Do nothing, just close the confirmation
    
    # Display progress based on candidate info & questions
    current_step = st.session_state.session["current_step"]
    total_steps = len(steps) + len(st.session_state.session["questions"])
    progress = (current_step / total_steps) if total_steps > 0 else 0
    st.sidebar.progress(progress)
    
    st.sidebar.download_button(
        "Export Session Data",
        data=json.dumps(st.session_state.session["candidate_info"], indent=2),
        file_name="interview_data.json",
        mime="application/json"
    )

# --- Conversation Steps (Candidate Info Collection) ---
steps = [
    {"prompt": "👋 Hello! I'm TalentScout Hiring Assistant. Let's start with your full name.", 
     "key": "name", "validator": lambda x: len(x.strip()) >= 3},
    {"prompt": "📧 What's your email address?", 
     "key": "email", "validator": validate_email},
    {"prompt": "📱 Please share your phone number (international format):", 
     "key": "phone", "validator": validate_phone},
    {"prompt": "⏳ How many years of professional experience do you have?", 
     "key": "years_exp", "validator": validate_experience},
    {"prompt": "💼 What position are you applying for? (You may enter a custom role)", 
     "key": "desired_position", "validator": lambda x: len(x.strip()) >= 2},
    {"prompt": "📍 Where are you currently located?", 
     "key": "location", "validator": lambda x: len(x.strip()) >= 2},
    {"prompt": "🛠️ List your relevant skills (comma-separated):", 
     "key": "skills", "validator": lambda x: len([s.strip() for s in x.split(",") if s.strip()]) >= 1},
]

# --- Main UI ---
st.set_page_config(page_title="TalentScout AI Interviewer", page_icon="🤖")
st.title("TalentScout AI Interview Assistant🤖")
render_sidebar()

# Display chat history
for msg in st.session_state.session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# --- Conversation Flow ---
current_step = st.session_state.session["current_step"]

# Technical Question Phase: one question at a time with progress indicator
if st.session_state.session["questions"]:
    q_idx = st.session_state.session["active_question_index"]
    # Check bounds for questions list
    if q_idx < len(st.session_state.session["questions"]):
        question = st.session_state.session["questions"][q_idx]
        if not question.get("asked", False):
            st.session_state.session["questions"][q_idx]["asked"] = True
            st.session_state.session["messages"].append({
                "role": "assistant",
                "content": f"**{question['position']} ({question['experience_level']}) Interview Questions:**\n{question['questions']}"
            })
            st.rerun()
        
        answer = render_question_interface()
        # Allow user to skip a question by entering "skip"
        if answer and answer.lower().strip() == "skip":
            st.session_state.session["messages"].append({
                "role": "assistant",
                "content": "⏭️ Question skipped."
            })
            st.session_state.session["active_question_index"] += 1
            st.rerun()
        elif answer and answer.lower().strip() != "":
            question["answers"].append(answer)
            st.session_state.session["messages"].append({
                "role": "user",
                "content": answer
            })
            st.session_state.session["active_question_index"] += 1
            # If last question answered, show summary
            if st.session_state.session["active_question_index"] >= len(st.session_state.session["questions"]):
                st.session_state.session["messages"].append({
                    "role": "assistant",
                    "content": "✅ Thank you! Here is a summary of your answers:\n" +
                               "\n".join([f"Q{i+1}: {q['answers']}" for i, q in enumerate(st.session_state.session["questions"])])
                })
                # Optionally, save candidate data here
                st.session_state.session = init_session_state()
                st.rerun()
            else:
                st.rerun()
    else:
        st.error("No questions available. Please try again later.")
        
# Candidate Info Collection Phase
elif current_step < len(steps):
    step = steps[current_step]
    if not any(m["content"] == step["prompt"] for m in st.session_state.session["messages"]):
        st.session_state.session["messages"].append({
            "role": "assistant",
            "content": step["prompt"]
        })
        st.rerun()

# --- Input Handling ---
if user_input := st.chat_input("Type your response..."):
    if user_input.lower() in ["exit", "quit", "end"]:
        st.session_state.session["messages"].append({
            "role": "user", 
            "content": user_input
        })
        st.session_state.session["messages"].append({
            "role": "assistant",
            "content": "👋 Thank you for your time! Your progress has been saved."
        })
        st.session_state.session = init_session_state()
        st.rerun()
    elif st.session_state.session["questions"]:
        # Question-answering phase is handled above
        pass
    else:
        current_step = st.session_state.session["current_step"]
        step = steps[current_step]
        # Sanitize the input (strip and html-escape)
        sanitized_input = html.escape(user_input.strip())
        if not step["validator"](sanitized_input):
            st.session_state.session["invalid_attempts"].setdefault(step["key"], 0)
            st.session_state.session["invalid_attempts"][step["key"]] += 1
            if st.session_state.session["invalid_attempts"][step["key"]] > MAX_INVALID_ATTEMPTS:
                st.error(f"Too many invalid attempts for {step['key']}. Exiting session for security.")
                st.session_state.session = init_session_state()
                st.rerun()
            else:
                error_msg = {
                    "name": "Please enter a valid name (min 3 characters).",
                    "email": "Invalid email format (example@domain.com).",
                    "phone": "Invalid phone number (use international format).",
                    "years_exp": "Please enter a valid number (0-50).",
                    "desired_position": "Please enter a valid position title.",
                    "location": "Please enter a valid location.",
                    "skills": "Please enter at least one skill."
                }.get(step["key"], "Invalid input.")
                st.error(error_msg)
        else:
            st.session_state.session["candidate_info"][step["key"]] = sanitized_input
            st.session_state.session["messages"].append({
                "role": "user",
                "content": sanitized_input
            })
            st.session_state.session["current_step"] += 1
            # When all candidate info is collected, generate interview questions
            if st.session_state.session["current_step"] == len(steps):
                skills = [s.strip() for s in st.session_state.session["candidate_info"]["skills"].split(",") if s.strip()]
                years_exp = st.session_state.session["candidate_info"]["years_exp"]
                position = st.session_state.session["candidate_info"]["desired_position"]
                with st.spinner("🔍 Generating interview questions..."):
                    st.session_state.session["questions"] = generate_questions(
                        position, years_exp, skills
                    )
                    st.session_state.session["active_question_index"] = 0
            st.rerun()
