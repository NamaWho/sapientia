# Sapientia - Interactive CLI with AI-powered Virtual Tutoring

## Description
Sapientia is a proof-of-concept project designed to develop a virtual tutor capable of assisting students in their learning journey. This interactive CLI application utilizes AI-powered tools such as GPT-4 for response evaluation, Whisper for voice transcription, and YouTube integration for supplemental learning resources.

## Key Features
- **Colorful CLI Interface** using `colorama` and `rich`
- **Emoji-enhanced interaction** for improved user engagement
- **Blinking text support** (compatible with select terminals)
- **Integration with OpenAI GPT-4** for response evaluation and question generation
- **Voice recording and transcription** using `sounddevice` and `whisper`
- **Adaptive multiple-choice question (MCQ) generation**
- **Automated YouTube search** for educational content
- **Student history tracking** for personalized learning experiences

## Requirements
- Python 3.x
- OpenAI API key (`OPENAI_API_KEY`)
- YouTube API key (`YOUTUBE_API_KEY`)
- Required Python libraries:
  ```sh
  pip install openai dotenv requests colorama rich sounddevice soundfile numpy whisper
  ```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/NamaWho/sapientia.git
   cd sapientia
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create a `.env` file with API keys:
   ```sh
   echo "OPENAI_API_KEY=your_openai_api_key" >> .env
   echo "YOUTUBE_API_KEY=your_youtube_api_key" >> .env
   ```

## Usage
Run the program with:
```sh
python main.py
```
Follow on-screen instructions to select a mode:
- **Study Mode** (AI-assisted questions and feedback)
- **Review Mode** (Voice-based answers and transcription with Whisper)

## Main Files
- `main.py`: Core CLI application
- `student_history.json`: Stores students' learning progress
- `qa.json`: Dataset of questions and correct answers

## Contribution
We welcome contributions! Feel free to open a pull request or report issues on GitHub.

## License
MIT License

