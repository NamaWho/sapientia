import openai
import json
import os
import dotenv
import streamlit as st
import requests

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.Client()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Constants for file paths
STUDENT_HISTORY_FILE = "student_history.json"

# Load student history
def load_student_history():
    if os.path.exists(STUDENT_HISTORY_FILE):
        with open(STUDENT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return {}

# Save student history
def save_student_history(history):
    with open(STUDENT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Query OpenAI API
def query_openai(prompt, temperature=0.7, max_tokens=500):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# Extract question from dataset
def get_question(dataset, level):
    return [q for q in dataset if q["livello"] == level]

# Evaluate student response
def evaluate_response(student_response, correct_answer):
    prompt = f"""Valuta questa risposta: '{student_response}' rispetto alla risposta corretta: '{correct_answer}'.\n"""
    return query_openai(prompt)

# Suggest external resources
def suggest_resources(concept, level):
    prompt = f"""Suggerisci video educativi su YouTube per il concetto '{concept}', specifici per il livello '{level}'.\nIncludi titolo, link e descrizione."""
    return query_openai(prompt)

# Generate practical examples
def generate_practical_example(concept, level):
    prompt = f"""Crea un esempio pratico per spiegare il concetto di '{concept}' a uno studente di livello '{level}'.\n"""
    return query_openai(prompt)

def search_youtube(concept, level, max_results=5):
    base_url = "https://www.googleapis.com/youtube/v3/search"
    query = f"{concept} tutorial livello {level}"
 
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY,
        "order": "relevance"
    }
 
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        videos = response.json().get("items", [])
        results = []
        for video in videos:
            title = video["snippet"]["title"]
            description = video["snippet"]["description"]
            video_id = video["id"]["videoId"]
            link = f"https://www.youtube.com/watch?v={video_id}"
            results.append({"title": title, "description": description, "link": link})
        return results
    else:
        print("Errore nella richiesta YouTube API:", response.status_code)
        return []

# Main Streamlit app
def main():
    st.title("Piattaforma Interattiva di Apprendimento")

    # Load dataset
    dataset_path = "Domande_e_Risposte.json"
    if not os.path.exists(dataset_path):
        st.error("Il file del dataset non esiste.")
        return

    with open(dataset_path, "r") as file:
        dataset = json.load(file)

    # Student ID input
    student_id = st.text_input("Inserisci il tuo ID studente:")
    if not student_id:
        st.warning("Inserisci il tuo ID studente per continuare.")
        return

    # Load student history
    history = load_student_history()
    if student_id not in history:
        history[student_id] = {"level": "base", "progress": []}

    student_data = history[student_id]
    level = student_data["level"]

    # Display question
    questions = get_question(dataset, level)
    if not questions:
        st.info("Nessuna domanda disponibile per il livello attuale.")
        return

    current_question = questions[0]
    st.subheader(f"Domanda ({level.capitalize()}):")
    st.write(current_question["domanda"])

    if "opzioni" in current_question:
        st.write("Opzioni:")
        for key, option in current_question["opzioni"].items():
            st.write(f"{key}: {option}")

    # Student response input
    student_response = st.text_area("La tua risposta:")

    understood = None

    if st.button("Invia risposta"):
        if not student_response:
            st.warning("Inserisci una risposta prima di inviare.")
        else:
            feedback = evaluate_response(student_response, current_question["risposta"])
            st.success("Feedback ricevuto:")
            st.write(feedback)

            if st.radio("Hai capito il concetto:", ["Sì", "No"]) == "No":
                understood = "No"
            
                st.write("Ecco alcune risorse aggiuntive:")
                resources = search_youtube(current_question["domanda"], level)
                for resource in resources:
                    st.write(f"Titolo: {resource['title']}")
                    st.write(f"Descrizione: {resource['description']}")
                    st.write(f"Link: {resource['link']}")

                st.write("Ecco un esempio pratico:")
                practical_example = generate_practical_example(current_question['domanda'], level)
                st.write(practical_example)
                st.button("Salva progresso")
            else:
                understood = "Sì"

            # Update student progress
            student_data["progress"].append({
                "domanda": current_question["domanda"],
                "risposta": student_response,
                "feedback": feedback,
                "understood": understood == "Sì"
            })

            # Update level if understood
            if understood == "Sì":
                if level == "base":
                    student_data["level"] = "intermedio"
                elif level == "intermedio":
                    student_data["level"] = "avanzato"

            save_student_history(history)
            st.success("Progresso salvato con successo!")

if __name__ == "__main__":
    main()
