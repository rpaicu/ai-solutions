import os
import argparse
import time
from pathlib import Path
from openai import OpenAI

# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:4000')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.rpa.icu')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://gpu01:4000')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://gpu02:13000')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'https://t.me/evilfreelancer')
DEFAULT_MODEL = 'fish-speech-1.5'


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Fish Speech TTS Generator')
    parser.add_argument('text', type=str, help='Input text or path to .txt file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output audio file path')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--reference-audio', type=str, help='Path to reference audio file')
    parser.add_argument('--seed', type=int, default=42, help='Seed for generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature (0.0-1.0)')
    parser.add_argument('--top-p', type=float, default=0.7, help='Top-p sampling (0.0-1.0)')
    parser.add_argument('--voice', type=str, default='english-nice', help='Voice preset')

    args = parser.parse_args()

    # Read input text
    if os.path.isfile(args.text) and args.text.endswith('.txt'):
        with open(args.text, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        input_text = args.text

    # Prepare reference audio
    reference_audio = None
    if args.reference_audio:
        if not os.path.isfile(args.reference_audio):
            raise FileNotFoundError(f"Reference audio file not found: {args.reference_audio}")

        reference_audio = open(args.reference_audio, 'rb')

    # Initialize client
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    # Generate speech
    try:
        response = client.audio.speech.create(
            model=args.model,
            input=input_text,
            voice=args.voice,
            extra_body={
                "temperature": args.temperature,
                "top_p":       args.top_p,
                "seed":        args.seed,
            }
        )
    finally:
        if reference_audio:
            reference_audio.close()

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(response.content)

    print(f"Audio saved to: {output_path.absolute()}")
    print(f"Execution time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    main()
