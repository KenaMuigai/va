from ASR import listen_once
from llm import LLM
from TTS import text_to_speech_stream

def main():
    print("🎙️ Voice Assistant Started (Ctrl+C to stop)")

    llm = LLM()

    while True:
        try:
            print("\nListening...")
            user_text = listen_once()
            print("🧑 You:", user_text)

            response = llm.generate(user_text)
            print("🤖 Assistant:", response)

            text_to_speech_stream(response)

        except KeyboardInterrupt:
            print("\n🛑 Assistant stopped.")
            break

        except Exception as e:
            print("⚠️ Error:", e)


if __name__ == "__main__":
    main()
