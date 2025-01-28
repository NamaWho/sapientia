import openai
import json
import os
import dotenv
import requests
import sys
import re

# Carica le variabili d'ambiente dal file .env
dotenv.load_dotenv()

# Imposta la chiave API di OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# In alcuni ambienti recenti openai.Client() non è più necessario;
# puoi usare direttamente openai.ChatCompletion per le chiamate.
# Se la tua versione di libreria OpenAI supporta `openai.Client`, lascialo pure:
try:
    client = openai.Client()
except AttributeError:
    # Se dà errore, lo sostituiamo con la chiamata diretta
    client = None

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Costanti per i percorsi dei file
STUDENT_HISTORY_FILE = "student_history.json"
DATASET_PATH = "Domande_e_Risposte.json"

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

def query_openai(messages, temperature=0.7, max_tokens=500):
    """Esegue una query al modello GPT-4 (o mini) con i parametri specificati."""
    if client:
        # Se la tua versione OpenAI supporta 'client.chat.completions.create'
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    else:
        # Altrimenti, usa la chiamata standard "openai.ChatCompletion.create"
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"].strip()

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
    response = query_openai(messages)
    
    # Prima di fare il parse, rimuoviamo eventuali backtick e testo superfluo
    # usando una regex che cerca il contenuto tra ```json e ```
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # In assenza di backtick, proviamo comunque a parsare la risposta
        json_str = response

    # Prova a convertire la risposta "ripulita" in JSON
    try:
        mcqs = json.loads(json_str)

        if isinstance(mcqs, list):
            return mcqs
        else:
            return []
    except:
        # Se non riesce a fare parse, ritorna lista vuota
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
    return query_openai(messages)

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
        print(f"Errore nella richiesta YouTube API: {response.status_code}")
        return []

def generate_practical_example(concept, level):
    """Genera un esempio pratico per spiegare il concetto di 'concept' a uno studente di livello 'level'."""
    prompt = f"Crea un esempio pratico per spiegare il concetto di '{concept}' a uno studente di livello '{level}'."
    messages = [
        {"role": "system", "content": "Sei un tutor che fornisce esempi pratici in modo chiaro e comprensibile."},
        {"role": "user", "content": prompt}
    ]
    return query_openai(messages)

############################
# CLI "Main" Function
############################

def main():
    print("============================================")
    print("  BENVENUTO NELLA PIATTAFORMA DIDATTICA CLI  ")
    print("============================================\n")

    # Verifichiamo che esista il file dataset
    if not os.path.exists(DATASET_PATH):
        print("[ERRORE] Il file del dataset non esiste. Assicurati di avere Domande_e_Risposte.json nella cartella.")
        sys.exit(1)

    # Carichiamo il dataset
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    # Input ID studente
    student_id = ""
    while not student_id.strip():
        student_id = input("Inserisci il tuo ID studente (obbligatorio): ").strip()

    # Carica lo storico degli studenti
    history = load_student_history()
    if student_id not in history:
        # Se è la prima volta che questo studente accede, creiamo una struttura di base
        history[student_id] = {
            "level": "base",        # partiamo dal livello base
            "current_index": 0,     # indice della domanda corrente in quel livello
            "progress": []          # lista di tentativi (domanda, risposta, feedback, ecc.)
        }

    student_data = history[student_id]
    level = student_data["level"]
    current_index = student_data.get("current_index", 0)

    # Filtra le domande per livello
    level_questions = get_questions_by_level(dataset, level)

    if not level_questions:
        print(f"[INFO] Nessuna domanda disponibile per il livello '{level}'.")
        print("Esci dal programma.")
        sys.exit(0)

    while True:

        # Se sforiamo il numero di domande disponibili
        if current_index >= len(level_questions):
            print("Hai completato TUTTE le domande disponibili per il livello attuale!")
            print("Complimenti, non ci sono altre domande da svolgere.")
            sys.exit(0)

        # Otteniamo la domanda corrente
        current_question = level_questions[current_index]

        # Stampa la domanda
        print("\n------------------------------------------------")
        print(f"LIVELLO: {level.upper()} | DOMANDA #{current_index + 1}")
        print("------------------------------------------------")
        print(f"Domanda: {current_question['domanda']}\n")

        # Se nel dataset ci sono opzioni preimpostate, le mostriamo
        if "opzioni" in current_question and isinstance(current_question["opzioni"], dict):
            print("Opzioni disponibili:")
            for key, option in current_question["opzioni"].items():
                print(f"  {key}: {option}")
            print("")

        # Richiesta risposta studente
        print("Scrivi la tua risposta qui sotto (conferma con Invio):")
        student_response = input(">> ").strip()

        # Se lo studente non inserisce nulla
        if not student_response:
            print("[ATTENZIONE] Non hai inserito alcuna risposta. Rilancia il programma per riprovare.")
            sys.exit(0)

        # Valuta la risposta e mostra feedback
        feedback = evaluate_response(student_response, current_question["risposta"])
        print("\n=== FEEDBACK SULLA TUA RISPOSTA ===")
        print(feedback)

        # Salviamo il tentativo nel "progress"
        attempt = {
            "domanda": current_question["domanda"],
            "risposta_studente": student_response,
            "feedback": feedback,
            "understood": False
        }
        student_data["progress"].append(attempt)
        save_student_history(history)

        # Generiamo 3 domande a scelta multipla per testare la comprensione
        mcq_set = generate_followup_mcq(
            current_question["domanda"],
            current_question["risposta"]
        )

        if not mcq_set:
            print("\n[ATTENZIONE] Non è stato possibile generare domande a scelta multipla.\n")
            print("Procedo alla domanda successiva.")
            current_index += 1
            student_data["current_index"] = current_index
            save_student_history(history)
            continue
        else:
            print("\nOra rispondi alle seguenti DOMANDE A SCELTA MULTIPLA (MCQ).")

        # Ciclo di tentativi sulle MCQ
        while True:
            user_answers = []

            # Mostra le MCQ e chiedi la risposta utente (A/B/C/D)
            for idx, mcq in enumerate(mcq_set):
                print("\n--------------------------------------------")
                print(f"MCQ {idx + 1}: {mcq['domanda']}")
                for letter, text in mcq.get("opzioni", {}).items():
                    print(f"   {letter}: {text}")
                print("--------------------------------------------")
                answer = ""
                while answer.upper() not in ["A", "B", "C", "D"]:
                    answer = input("La tua risposta (A/B/C/D): ").strip()
                    if answer.upper() not in ["A", "B", "C", "D"]:
                        print("[ERRORE] Devi inserire una delle quattro opzioni: A, B, C o D.")
                user_answers.append(answer.upper())

            # Verifichiamo se l'utente ha risposto correttamente a tutte
            if check_mcq_answers(mcq_set, user_answers):
                print("\nCOMPLIMENTI! Hai risposto correttamente a tutte le domande a scelta multipla!")
                # Aggiorniamo il tentativo come "understood"
                if student_data["progress"]:
                    student_data["progress"][-1]["understood"] = True

                # Passiamo alla prossima domanda
                student_data["current_index"] += 1
                save_student_history(history)

                # Se non ci sono più domande, concludiamo
                if student_data["current_index"] >= len(level_questions):
                    print("Hai completato tutte le domande disponibili per il livello attuale! Ottimo lavoro!")
                else:
                    # Domanda all'utente se vuole continuare
                    print("\nVuoi continuare con la prossima domanda? (s/n)")
                    choice = input(">> ").strip().lower()
                    if choice != "s":
                        print("\nGrazie per aver usato la piattaforma didattica CLI.")
                        sys.exit(0)
                    else:
                        print("Passiamo alla prossima domanda.\n")
                        current_index += 1
                break
            else:
                print("\nAlcune risposte non sono corrette. Ecco delle risorse aggiuntive per aiutarti:\n")

                # Generiamo query per YouTube
                query = generate_yt_query(current_question["domanda"], current_question['risposta'], level)
                videos = search_youtube(query)

                # Mostriamo i video di YouTube
                print("=== Video consigliati su YouTube ===\n")
                if not videos:
                    print("(Nessun video trovato o errore nella richiesta.)\n")
                else:
                    for vid in videos:
                        print(f"Titolo: {vid['title']}")
                        print(f"Descrizione: {vid['description']}")
                        print(f"Link: {vid['link']}\n")

                # Esempio pratico
                print("=== Esempio pratico ===")
                example = generate_practical_example(current_question["domanda"], level)
                print(example)

                print("\n[RIPROVA] Rivedi le tue risposte e riprova a rispondere alle MCQ.\n")
                input("Premi Invio per continuare...")

if __name__ == "__main__":
    main()
