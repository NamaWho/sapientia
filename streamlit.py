import streamlit as st
import openai
import json, os, re, time, random, warnings
import dotenv
import requests
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np


warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Caricamento delle variabili d'ambiente
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Costanti per i file di dataset e storico
STUDENT_HISTORY_FILE = "student_history.json"
DATASET_PATH = "qa.json"

####################################
# Data Persistence
####################################

def load_student_history():
    if os.path.exists(STUDENT_HISTORY_FILE):
        with open(STUDENT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return {}

def save_student_history(history):
    with open(STUDENT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

####################################
# OpenAI Helpers
####################################

def query_openai(messages, temperature=0.7, max_tokens=500):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    text = response["choices"][0]["message"]["content"].strip()
    return text

def evaluate_response(student_response, correct_answer):
    prompt = (
        f"Valuta la seguente risposta rispetto alla risposta corretta:\n\n"
        f"Risposta dello studente: '{student_response}'\n"
        f"Risposta corretta: '{correct_answer}'\n\n"
        f"Fornisci un breve feedback su cosa c'è di giusto o sbagliato, e come migliorare."
    )
    messages = [
        {"role": "system", "content": "Sei un tutor che valuta risposte e fornisce correzioni in modo chiaro e conciso."},
        {"role": "user", "content": prompt}
    ]
    return query_openai(messages)

def generate_followup_mcq(question, correct_answer):
    prompt = f"""Crea tre domande a scelta multipla (in italiano) per verificare la comprensione del concetto relativo a questa domanda e risposta:

Domanda: {question}
Risposta corretta: {correct_answer}

Ogni domanda deve avere 4 opzioni con UNA sola risposta corretta, e indica chiaramente la risposta corretta in un formato strutturato JSON del tipo:
[{{"domanda":"...","opzioni":{{"A":"...","B":"...","C":"...","D":"..."}},"corretta":"A"}}, ...]"""
    messages = [
        {"role": "system", "content": "Sei un tutor che genera quiz a scelta multipla in formato JSON."},
        {"role": "user", "content": prompt}
    ]
    response = query_openai(messages)
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    json_str = match.group(1) if match else response
    try:
        mcqs = json.loads(json_str)
        if isinstance(mcqs, list):
            return mcqs
    except Exception as e:
        st.error("Errore nel parsing delle MCQ generate.")
    return []

def check_mcq_answers(mcq_set, user_answers):
    if len(mcq_set) != len(user_answers):
        return False
    for i, mcq in enumerate(mcq_set):
        if user_answers[i].upper() != mcq.get("corretta", "").upper():
            return False
    return True

def generate_yt_query(question, answer, level):
    prompt = (
        f"Estrai 3 parole chiave da questo contesto:\n"
        f"- Domanda: '{question}'\n"
        f"- Risposta: '{answer}'\n"
        f"L'output deve essere una stringa di parole chiave separate da spazio."
    )
    messages = [
        {"role": "system", "content": "Sei un tutor che estrae parole chiave per una query di ricerca su YouTube."},
        {"role": "user", "content": prompt}
    ]
    return query_openai(messages)

def search_youtube(concept, max_results=3):
    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": concept,
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
            description = video["snippet"].get("description", "")
            video_id = video["id"].get("videoId", "")
            link = f"https://www.youtube.com/watch?v={video_id}"
            results.append({"title": title, "description": description, "link": link})
        return results
    else:
        st.error(f"Richiesta YouTube API fallita: {response.status_code}")
        return []

def generate_practical_example(concept, level):
    prompt = f"Crea un esempio pratico per spiegare il concetto di '{concept}' a uno studente di livello '{level}'."
    messages = [
        {"role": "system", "content": "Sei un tutor che fornisce esempi pratici in modo chiaro e comprensibile."},
        {"role": "user", "content": prompt}
    ]
    return query_openai(messages)

####################################
# Funzioni per la registrazione e trascrizione audio
####################################

# Registra l'audio utilizzando sounddevice (durata in secondi)
def record_audio_sd(duration=10, sample_rate=32000):
    st.info("Registrazione audio in corso... Parla ora!")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return recording, sample_rate

# Salva l'audio registrato in un file WAV
def save_audio_sd(recording, sample_rate, filename="temp_recording.wav"):
    sf.write(filename, recording, sample_rate)
    return filename

# Trascrive l'audio salvato utilizzando il modello Whisper
def transcribe_audio_file(audio_file, model="base"):
    try:
        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        st.error("Errore durante il caricamento o la trascrizione del modello Whisper.\n"
                 "Questo potrebbe indicare che il file del modello è corrotto. Prova a cancellare la cache del modello e riprovare.\n"
                 "Errore: " + str(e))
        return ""


####################################
# Funzioni di supporto per il dataset
####################################

def get_questions_by_level(dataset, level):
    return [q for q in dataset if q["livello"] == level]

def get_review_question(dataset, student_id, history):
    student_data = history.get(student_id, {})
    progress = student_data.get("progress", [])
    review_candidates = [attempt["domanda"] for attempt in progress if not attempt.get("understood", False)]
    if not review_candidates:
        return None
    review_question_text = random.choice(review_candidates)
    for q in dataset:
        if q["domanda"] == review_question_text:
            return q
    return None

####################################
# Interfaccia Grafica con Streamlit
####################################

# Modalità Studio (testuale)
def run_study_mode(dataset, student_id, history):
    student_data = history.get(student_id, {"level": "base", "current_index": 0, "progress": []})
    level = student_data.get("level", "base")
    level_questions = get_questions_by_level(dataset, level)
    if not level_questions:
        st.error(f"Nessuna domanda disponibile per il livello '{level}'.")
        return
    current_index = student_data.get("current_index", 0)
    if current_index >= len(level_questions):
        st.success("Hai completato tutte le domande per il livello attuale!")
        return

    current_question = level_questions[current_index]
    st.header(f"Domanda #{current_index + 1} - Livello: {level}")
    st.markdown(f"**Domanda:** {current_question['domanda']}")
    if "opzioni" in current_question and isinstance(current_question["opzioni"], dict):
        st.markdown("**Opzioni disponibili:**")
        for key, option in current_question["opzioni"].items():
            st.write(f"{key}: {option}")

    if "mcq_set" in st.session_state and st.session_state["mcq_set"] is not None:
        mcq_set = st.session_state["mcq_set"]
        st.markdown("### Domande a Scelta Multipla (MCQ)")
        user_answers = []
        for idx, mcq in enumerate(mcq_set):
            st.markdown(f"**MCQ {idx + 1}:** {mcq['domanda']}")
            options = mcq.get("opzioni", {})
            answer = st.radio(f"Seleziona la risposta per MCQ {idx + 1}:", list(options.keys()), key=f"mcq_{idx}")
            user_answers.append(answer)
            for letter, text in options.items():
                st.write(f"{letter}: {text}")
        if st.button("Invia risposte MCQ"):
            if check_mcq_answers(mcq_set, user_answers):
                st.success("Complimenti! Hai risposto correttamente a tutte le MCQ.")
                if "attempt_index" in st.session_state:
                    idx = st.session_state["attempt_index"]
                    if idx < len(student_data["progress"]):
                        student_data["progress"][idx]["understood"] = True
                student_data["current_index"] = current_index + 1
                history[student_id] = student_data
                save_student_history(history)
                st.session_state["mcq_set"] = None
                st.session_state["attempt_index"] = None
            else:
                st.error("Alcune risposte non sono corrette. Riprova.")
                st.markdown("### Risorse aggiuntive:")
                yt_query = generate_yt_query(current_question["domanda"], current_question["risposta"], level)
                videos = search_youtube(yt_query)
                if videos:
                    st.markdown("**Video consigliati su YouTube:**")
                    for vid in videos:
                        st.write(f"**{vid['title']}**")
                        st.write(vid['link'])
                else:
                    st.write("Nessun video trovato.")
                practical_example = generate_practical_example(current_question["domanda"], level)
                st.markdown("**Esempio pratico:**")
                st.write(practical_example)
    else:
        student_response = st.text_area("Inserisci la tua risposta:", "")
        if st.button("Invia risposta"):
            if not student_response.strip():
                st.warning("Inserisci una risposta valida.")
            else:
                feedback = evaluate_response(student_response, current_question["risposta"])
                st.markdown("### Feedback:")
                st.write(feedback)
                attempt = {
                    "domanda": current_question["domanda"],
                    "risposta_studente": student_response,
                    "feedback": feedback,
                    "understood": False
                }
                student_data["progress"].append(attempt)
                st.session_state["attempt_index"] = len(student_data["progress"]) - 1
                mcq_set = generate_followup_mcq(current_question["domanda"], current_question["risposta"])
                if not mcq_set:
                    st.error("Non è stato possibile generare domande a scelta multipla. Procedi alla prossima domanda.")
                    student_data["current_index"] = current_index + 1
                    history[student_id] = student_data
                    save_student_history(history)
                else:
                    st.session_state["mcq_set"] = mcq_set

# Modalità Ripasso (registrazione audio con sounddevice)
def run_review_mode(dataset, student_id, history):
    st.header("Modalità Ripasso")
    question = get_review_question(dataset, student_id, history)
    if not question:
        st.info("Non ci sono domande da ripassare al momento.")
        return

    st.markdown(f"**Domanda da ripassare:** {question['domanda']}")
    st.write("Utilizza il pulsante per registrare la tua risposta audio.")

    # Pulsante per registrare l'audio (durata fissa, es. 10 secondi)
    if st.button("🎙️ Registra Audio per 10 secondi"):
        recording, sample_rate = record_audio_sd(duration=10, sample_rate=32000)
        st.success("Registrazione terminata!")
        # Salva il file audio
        audio_file = save_audio_sd(recording, sample_rate)
        st.audio(audio_file, format="audio/wav")
        # Trascrizione dell'audio
        with st.spinner("Trascrizione in corso..."):
            transcribed_response = transcribe_audio_file(audio_file)
        st.markdown("### Trascrizione:")
        st.write(transcribed_response)
        # Valuta la risposta trascritta
        feedback = evaluate_response(transcribed_response, question["risposta"])
        st.markdown("### Feedback:")
        st.write(feedback)
        # Aggiorna lo storico dello studente
        student_data = history.get(student_id, {"progress": []})
        updated = False
        for attempt in student_data.get("progress", []):
            if attempt["domanda"] == question["domanda"]:
                attempt["review_attempt"] = {
                    "risposta": transcribed_response,
                    "feedback": feedback,
                    "timestamp": time.time()
                }
                updated = True
                break
        if not updated:
            student_data["progress"].append({
                "domanda": question["domanda"],
                "review_attempt": {
                    "risposta": transcribed_response,
                    "feedback": feedback,
                    "timestamp": time.time()
                }
            })
        history[student_id] = student_data
        save_student_history(history)
        st.success("✅ Revisione completata!")

####################################
# Funzione principale
####################################

def main():
    st.title("Sapientia")
    if not os.path.exists(DATASET_PATH):
        st.error("Il file del dataset non esiste. Assicurati di avere qa.json.")
        return
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    student_id = st.text_input("Inserisci il tuo ID studente", key="student_id")
    if student_id.strip() == "":
        st.warning("Inserisci il tuo ID studente per continuare.")
        return

    history = load_student_history()
    if student_id not in history:
        history[student_id] = {"level": "base", "current_index": 0, "progress": []}
        save_student_history(history)

    mode = st.sidebar.radio("Scegli la modalità:", ("Modalità Studio", "Modalità Ripasso"))
    if mode == "Modalità Studio":
        run_study_mode(dataset, student_id, history)
    elif mode == "Modalità Ripasso":
        run_review_mode(dataset, student_id, history)

if __name__ == "__main__":
    main()
