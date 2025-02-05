#!/usr/bin/env python3

"""
CLI Interattiva con effetti speciali (colori, emoji, testo lampeggiante)
"""

import openai
import json
import os
import dotenv
import requests
import sys
import re
import time
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import warnings 
from rich.console import Console
from rich.markdown import Markdown

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Per i colori e gli effetti ANSI
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
except ImportError:
    print("[AVVISO] 'colorama' non installato. Installalo con 'pip install colorama' per i colori.")
    class Fore:
        RED = ''
        GREEN = ''
        YELLOW = ''
        CYAN = ''
        MAGENTA = ''
        RESET = ''
    class Back:
        RESET = ''
    class Style:
        BRIGHT = ''
        RESET_ALL = ''

# Per testo lampeggiante (non tutti i terminali lo supportano)
BLINK = "\033[5m"
RESET = "\033[0m"

# Se vuoi aggiungere altre emoji, puoi modificarle qui
EMOJI_BOOK = "📚"
EMOJI_STAR = "✨"
EMOJI_ROBOT = "🤖"
EMOJI_CLAP = "👏"
EMOJI_FIRE = "🔥"
EMOJI_COOL = "😎"

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Prova a creare un client OpenAI se disponibile, altrimenti usa ChatCompletion
try:
    client = openai.Client()
except AttributeError:
    client = None

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Percorsi file
STUDENT_HISTORY_FILE = "student_history.json"
DATASET_PATH = "qa.json"

############################
# Data Persistence
############################

def load_student_history():
    """Carica lo storico degli studenti dal file JSON."""
    if os.path.exists(STUDENT_HISTORY_FILE):
        with open(STUDENT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return {}

def save_student_history(history):
    """Salva lo storico degli studenti su file JSON."""
    with open(STUDENT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

############################
# OpenAI Helpers
############################

def display_gpt_response(response):
    console = Console()
    markdown = Markdown(response)
    return console.print(markdown)

def query_openai(messages, temperature=0.7, max_tokens=500, stream=True):
    """Esegue una query al modello GPT-4 (o mini) con i parametri specificati."""
    if client:
        # Se la tua versione OpenAI supporta 'client.chat.completions.create'
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
    else:
        # Altrimenti, usa la chiamata standard "openai.ChatCompletion.create"
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens, 
            stream=True
        )
    
    # Stampare lo stream man mano che arriva
    full_response = ""
    for chunk in response:
        if hasattr(chunk, 'choices'):
            for choice in chunk.choices:
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    content = choice.delta.content
                    if content:
                        full_response += content
                        if stream:
                            print(content, end="", flush=True)

    print()  # Nuova riga alla fine della risposta
    return full_response  # Se vuoi usarlo altrove
    # return display_gpt_response(response["choices"][0]["message"]["content"].strip())

############################
# Core Functions
############################

def get_questions_by_level(dataset, level):
    """Ritorna le domande del dataset filtrate per livello."""
    return [q for q in dataset if q["livello"] == level]

def evaluate_response(student_response, correct_answer):
    """Valuta la risposta dello studente rispetto alla risposta corretta tramite GPT."""
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
    """
    Genera tre domande a scelta multipla (MCQ) basate sul concetto trattato,
    con una sola risposta corretta, in formato JSON.
    """
    prompt = f"""Crea tre domande a scelta multipla (in italiano) per verificare la comprensione del concetto relativo a questa domanda e risposta:\n\nDomanda: {question}\nRisposta corretta: {correct_answer}\n\nOgni domanda deve avere 4 opzioni con UNA sola risposta corretta, e indica chiaramente la risposta corretta in un formato strutturato JSON del tipo:\n[{{\"domanda\":\"...\",\"opzioni\":{{\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\"}},\"corretta\":\"A\"}}, ...]"""
    messages = [
        {"role": "system", "content": "Sei un tutor che genera quiz a scelta multipla in formato JSON."},
        {"role": "user", "content": prompt}
    ]
    response = query_openai(messages, stream=False)

    # Pulizia dell'eventuale codice tra ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response

    try:
        mcqs = json.loads(json_str)
        if isinstance(mcqs, list):
            return mcqs
        else:
            return []
    except:
        return []

def check_mcq_answers(mcq_set, user_answers):
    """
    Verifica se l'utente ha risposto correttamente a tutte e tre le MCQ.
    mcq_set: lista di domande MCQ generate
    user_answers: lista di risposte A/B/C/D dell'utente
    """
    if len(mcq_set) != len(user_answers):
        return False
    for i, mcq in enumerate(mcq_set):
        if user_answers[i].upper() != mcq.get("corretta", "").upper():
            return False
    return True

def generate_yt_query(question, answer, level):
    """Usa LLM per generare una query di ricerca YouTube basata sulla domanda e sulla risposta."""
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
    return query_openai(messages, stream=False)

def search_youtube(concept, max_results=3):
    """Effettua una ricerca su YouTube con le parole chiave specificate e restituisce i risultati."""
    base_url = "https://www.googleapis.com/youtube/v3/search"
    query = f"{concept}"

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
            description = video["snippet"].get("description", "")
            video_id = video["id"].get("videoId", "")
            link = f"https://www.youtube.com/watch?v={video_id}"
            results.append({"title": title, "description": description, "link": link})
        return results
    else:
        print(f"{Fore.RED}[ERRORE]{Fore.RESET} Richiesta YouTube API fallita: {response.status_code}")
        return []

def generate_practical_example(concept, level):
    """Genera un esempio pratico per spiegare il concetto di 'concept' a uno studente di livello 'level'."""
    prompt = f"Crea un esempio pratico per spiegare il concetto di '{concept}' a uno studente di livello '{level}'."
    messages = [
        {"role": "system", "content": "Sei un tutor che fornisce esempi pratici in modo chiaro e comprensibile."},
        {"role": "user", "content": prompt}
    ]
    return query_openai(messages)

# Nuove funzioni per il ripasso
def record_audio(duration=15, sample_rate=32000):
    """Registra audio dal microfono per la durata specificata."""
    print(f"{Fore.YELLOW}Registrazione in corso...{Fore.RESET}")
    recording = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1)
    sd.wait()
    return recording, sample_rate

def save_audio(recording, sample_rate, filename="temp_recording.wav"):
    """Salva la registrazione audio su file."""
    sf.write(filename, recording, sample_rate)
    return filename

def transcribe_audio(audio_file, model="base"):
    """Trascrive l'audio usando Whisper."""
    whisper_model = whisper.load_model(model)
    result = whisper_model.transcribe(audio_file)
    return result["text"]

def get_review_question(dataset, student_id, history):
    """Seleziona una domanda per il ripasso basandosi sullo storico dello studente."""
    student_data = history[student_id]
    progress = student_data.get("progress", [])
    
    # Filtra le domande che necessitano ripasso (understood=False o non recenti)
    review_candidates = []
    for attempt in progress:
        if not attempt.get("understood", False):
            review_candidates.append(attempt["domanda"])
    
    if not review_candidates:
        return None
    
    # Seleziona una domanda casuale tra quelle che necessitano ripasso
    import random
    review_question = random.choice(review_candidates)
    
    # Trova la domanda completa nel dataset
    for q in dataset:
        if q["domanda"] == review_question:
            return q
    
    return None

############################
# Effetti Speciali
############################

def fancy_intro():
    print(Fore.MAGENTA + Style.BRIGHT + "============================================")
    print(
        EMOJI_BOOK + "  " + BLINK + "BENVENUTO IN SAPIENTIA " + RESET + "  " + EMOJI_BOOK
    )
    print("============================================" + Fore.RESET + Style.RESET_ALL + "\n")
    # Piccola animazione
    spinner_chars = ["|", "/", "-", "\\"]
    print("Caricamento in corso ", end="")
    for _ in range(8):
        for char in spinner_chars:
            print(f"\b{Fore.YELLOW}{char}{Fore.RESET}", end="", flush=True)
            time.sleep(0.1)
    print("\b ")
    print(f"{EMOJI_STAR} Setup completato! {EMOJI_STAR}\n")


def fancy_level_banner(level, question_index):
    print(Fore.CYAN + Style.BRIGHT + "------------------------------------------------")
    print(f"{EMOJI_FIRE} LIVELLO: {level.upper()} | DOMANDA # {question_index} {EMOJI_FIRE}")
    print("------------------------------------------------" + Fore.RESET + Style.RESET_ALL)


def study_mode(dataset, student_id, history):
    student_data = history[student_id]
    level = student_data["level"]
    current_index = student_data.get("current_index", 0)

    # Filtra le domande per livello
    level_questions = get_questions_by_level(dataset, level)

    if not level_questions:
        print(f"[INFO] Nessuna domanda disponibile per il livello '{level}'.")
        print("Esco dal programma.")
        sys.exit(0)

    while True:
        # Se sforiamo il numero di domande disponibili
        if current_index >= len(level_questions):
            print(Fore.YELLOW + Style.BRIGHT + "Hai completato TUTTE le domande disponibili per il livello attuale!" + Fore.RESET)
            print(f"Complimenti {EMOJI_CLAP}, non ci sono altre domande da svolgere.")
            sys.exit(0)

        # Otteniamo la domanda corrente
        current_question = level_questions[current_index]

        # Banner di livello
        fancy_level_banner(level, current_index + 1)

        print(Fore.MAGENTA + "Domanda:" + Fore.RESET, current_question['domanda'], "\n")

        # Se nel dataset ci sono opzioni preimpostate
        if "opzioni" in current_question and isinstance(current_question["opzioni"], dict):
            print(Fore.BLUE + "Opzioni disponibili:" + Fore.RESET)
            for key, option in current_question["opzioni"].items():
                print(f"  {key}: {option}")
            print("")

        # Richiesta risposta studente
        print(Fore.GREEN + Style.BRIGHT + "Scrivi la tua risposta qui sotto (conferma con Invio):" + Fore.RESET + Style.RESET_ALL)
        student_response = input(EMOJI_ROBOT + " >> ").strip()

        # Se lo studente non inserisce nulla
        if not student_response:
            print("[ATTENZIONE] Non hai inserito alcuna risposta. Riavvia il programma per riprovare.")
            sys.exit(0)

        # Valuta la risposta
        print("\n" + Fore.YELLOW + Style.BRIGHT + "=== FEEDBACK SULLA TUA RISPOSTA ===" + Fore.RESET + Style.RESET_ALL)
        feedback = evaluate_response(student_response, current_question["risposta"])

        # Salviamo il tentativo
        attempt = {
            "domanda": current_question["domanda"],
            "risposta_studente": student_response,
            "feedback": feedback,
            "understood": False
        }
        student_data["progress"].append(attempt)
        save_student_history(history)

        # Generiamo 3 MCQ
        mcq_set = generate_followup_mcq(
            current_question["domanda"],
            current_question["risposta"]
        )

        if not mcq_set:
            print("\n" + Fore.RED + "[ATTENZIONE]" + Fore.RESET + " Non è stato possibile generare domande a scelta multipla.")
            print("Procedo alla domanda successiva.")
            current_index += 1
            student_data["current_index"] = current_index
            save_student_history(history)
            continue
        else:
            print("\nOra rispondi alle seguenti " + Fore.CYAN + "DOMANDE A SCELTA MULTIPLA (MCQ)" + Fore.RESET + ".")

        # Ciclo di tentativi sulle MCQ
        while True:
            user_answers = []

            for idx, mcq in enumerate(mcq_set):
                print("\n" + Fore.MAGENTA + Style.BRIGHT + f"MCQ {idx + 1}:" + Fore.RESET + Style.RESET_ALL, mcq['domanda'])
                for letter, text in mcq.get("opzioni", {}).items():
                    print(f"   {letter}: {text}")

                answer = ""
                while answer.upper() not in ["A", "B", "C", "D"]:
                    answer = input(EMOJI_ROBOT + " La tua risposta (A/B/C/D): ").strip()
                    if answer.upper() not in ["A", "B", "C", "D"]:
                        print(Fore.RED + "[ERRORE]" + Fore.RESET + " Devi inserire una delle quattro opzioni: A, B, C o D.")
                user_answers.append(answer.upper())

            if check_mcq_answers(mcq_set, user_answers):
                print("\n" + Fore.GREEN + Style.BRIGHT + "COMPLIMENTI!" + Fore.RESET + Style.RESET_ALL + " Hai risposto correttamente a tutte le domande a scelta multipla!")
                if student_data["progress"]:
                    student_data["progress"][-1]["understood"] = True

                student_data["current_index"] += 1
                save_student_history(history)

                if student_data["current_index"] >= len(level_questions):
                    print(Fore.YELLOW + Style.BRIGHT + "Hai completato tutte le domande disponibili per il livello attuale!" + Fore.RESET)
                    print(f"{EMOJI_COOL} Ottimo lavoro!")
                else:
                    print("\nVuoi continuare con la prossima domanda? (s/n)")
                    choice = input(EMOJI_ROBOT + " >> ").strip().lower()
                    if choice != "s":
                        print(f"\n{EMOJI_STAR} Grazie per aver partecipato! {EMOJI_STAR}")
                        return
                    else:
                        print("\nProcedo con la prossima domanda...")
                        current_index += 1
                break
            else:
                print("\n" + Fore.RED + "Alcune risposte non sono corrette." + Fore.RESET + f" {EMOJI_STAR}Riproviamo!{EMOJI_STAR}\n")

                # Risorse aggiuntive
                print("Ecco delle risorse aggiuntive per aiutarti:\n")

                # YouTube
                query = generate_yt_query(current_question["domanda"], current_question['risposta'], level)
                videos = search_youtube(query)

                print(Fore.YELLOW + Style.BRIGHT + "=== Video consigliati su YouTube ===\n" + Fore.RESET + Style.RESET_ALL)
                if not videos:
                    print("(Nessun video trovato o errore nella richiesta.)\n")
                else:
                    for vid in videos:
                        print(f"Titolo: {vid['title']}")
                        print(f"Descrizione: {vid['description']}")
                        print(f"Link: {vid['link']}\n")

                print(Fore.YELLOW + Style.BRIGHT + "=== Esempio pratico ===" + Fore.RESET + Style.RESET_ALL)
                generate_practical_example(current_question["domanda"], level)


                print("\n[RIPROVA] Rivedi le tue risposte e riprova a rispondere alle MCQ.")
                input("Premi Invio per continuare...")


def review_mode(dataset, student_id, history):
    """Gestisce la modalità di ripasso."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== MODALITÀ RIPASSO ==={Style.RESET_ALL}")
    
    while True:
        question = get_review_question(dataset, student_id, history)
        if not question:
            print(f"\n{Fore.YELLOW}Non ci sono domande da ripassare al momento.{Fore.RESET}")
            return
        
        print(f"\n{Fore.MAGENTA}Domanda da ripassare:{Fore.RESET} {question['domanda']}\n")
        input(f"{Fore.GREEN}Premi Invio quando sei pronto a registrare la tua risposta...{Fore.RESET}")
        
        # Registra la risposta audio
        recording, sample_rate = record_audio()
        audio_file = save_audio(recording, sample_rate)
        
        # Trascrivi la risposta
        print(f"\n{Fore.YELLOW}Elaborazione della risposta...{Fore.RESET}")
        transcribed_response = transcribe_audio(audio_file)
        
        print(f"\n{Fore.CYAN}La tua risposta trascritta:{Fore.RESET}\n{transcribed_response}\n")
        
        # Valuta la risposta
        print(f"\n{Fore.YELLOW}=== FEEDBACK SULLA TUA RISPOSTA ==={Fore.RESET}")
        feedback = evaluate_response(transcribed_response, question["risposta"])
        
        # Aggiorna lo storico
        for idx, attempt in enumerate(history[student_id]["progress"]):
            if attempt["domanda"] == question["domanda"]:
                history[student_id]["progress"][idx]["review_attempt"] = {
                    "risposta": transcribed_response,
                    "feedback": feedback,
                    "timestamp": time.time()
                }
                break
        
        save_student_history(history)
        
        # Chiedi se continuare
        choice = input(f"\n{EMOJI_ROBOT} Vuoi ripassare un'altra domanda? (s/n): ").strip().lower()
        if choice != 's':
            break


############################
# CLI "Main" Function
############################
def main():
    fancy_intro()

    # Verifichiamo che esista il file dataset
    if not os.path.exists(DATASET_PATH):
        print(f"{Fore.RED}[ERRORE]{Fore.RESET} Il file del dataset non esiste. Assicurati di avere qa.json.")
        sys.exit(1)

    # Carichiamo il dataset
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    # Input ID studente
    student_id = ""
    while not student_id.strip():
        student_id = input(Fore.GREEN + Style.BRIGHT + "Inserisci il tuo ID studente (obbligatorio): " + Fore.RESET + Style.RESET_ALL).strip()

    # Carica lo storico
    history = load_student_history()
    if student_id not in history:
        # Se è la prima volta che questo studente accede, creiamo una struttura di base
        history[student_id] = {
            "level": "base",         # partiamo dal livello base
            "current_index": 0,      # indice domanda corrente
            "progress": []
        }

    # Menu principale
    while True:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}=== MENU PRINCIPALE ==={Style.RESET_ALL}")
        print("1. Modalità Studio")
        print("2. Modalità Ripasso")
        print("3. Esci")
        
        choice = input(f"\n{EMOJI_ROBOT} Seleziona un'opzione (1-3): ").strip()
        
        if choice == "1":
            study_mode(dataset, student_id, history)  # Rinomina il vecchio main loop in study_mode()
        elif choice == "2":
            review_mode(dataset, student_id, history)
        elif choice == "3":
            print(f"\nGrazie per aver usato la piattaforma didattica CLI! {EMOJI_COOL}")
            sys.exit(0)
        else:
            print(f"{Fore.RED}Opzione non valida. Riprova.{Fore.RESET}")

if __name__ == "__main__":
    main()
