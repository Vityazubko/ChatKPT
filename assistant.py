import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import json
from urllib import request
from urllib.error import HTTPError, URLError


SYSTEM_PROMPT = (
    "Ти дружній голосовий асистент. Відповідай українською, коротко і по суті."
)


@dataclass
class ChatConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 40


class VoiceChatAI:
    def __init__(self, config: ChatConfig, enable_voice: bool = True):
        self.config = config
        self.enable_voice = enable_voice
        self.history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.engine = None
        self.recognizer = None

        if self.enable_voice:
            import pyttsx3
            import speech_recognition as sr

            self.engine = pyttsx3.init()
            self.recognizer = sr.Recognizer()

    def speak(self, text: str) -> None:
        if not self.engine:
            return
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self) -> str:
        if not self.recognizer:
            raise RuntimeError("Voice mode is disabled")

        import speech_recognition as sr

        with sr.Microphone() as source:
            print("🎙️ Слухаю... (скажи фразу)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio, language="uk-UA")
            print(f"👤 Ви: {text}")
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

    def build_messages(self, user_text: str) -> List[Dict[str, str]]:
        return self.history + [{"role": "user", "content": user_text}]

    def ask_model(self, user_text: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": self.build_messages(user_text),
            "temperature": 0.7,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        req = request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError("request failed") from exc

        return data["choices"][0]["message"]["content"].strip()

    def reply(self, user_text: str) -> str:
        if not user_text.strip():
            return "Я не розчув. Спробуй ще раз."

        try:
            answer = self.ask_model(user_text)
        except Exception:
            return "Вибач, сталася помилка підключення до моделі."

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        return answer


def load_config() -> ChatConfig:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ Додай OPENAI_API_KEY в змінні середовища.")
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return ChatConfig(api_key=api_key, model=model, base_url=base_url)


def main() -> None:
    config = load_config()
    assistant = VoiceChatAI(config, enable_voice=True)

    print("🤖 Голосовий ШІ запущено. Скажи 'стоп' або 'вихід', щоб завершити.")

    while True:
        user_text = assistant.listen()
        if user_text.lower().strip() in {"стоп", "вихід", "stop", "exit"}:
            bye = "До зустрічі!"
            print(f"🤖 ШІ: {bye}")
            assistant.speak(bye)
            break

        answer = assistant.reply(user_text)
        print(f"🤖 ШІ: {answer}")
        assistant.speak(answer)


if __name__ == "__main__":
    main()
