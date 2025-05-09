import os
import argparse
import time

from openai import OpenAI

# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:4000')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.rpa.icu')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://gpu01:12000')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'https://t.me/evilfreelancer')
DEFAULT_MODEL = 'large-v3-turbo'
DEFAULT_LANGUAGE = "ru"


def main():
    # Замер времени начала выполнения
    start_time = time.time()

    # Read the file from the file path by arguments of argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--format', type=str, choices=['text', 'json', 'srt', 'verbose_json'], default='srt')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--language', type=str, default=DEFAULT_LANGUAGE)
    args = parser.parse_args()

    # Read the file from the file path
    input_file = open(args.file, "rb")

    # Initialize client
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    # Run the transcription with time measurement
    transcription = client.audio.transcriptions.create(
        language=args.language,
        file=input_file,
        model=args.model,
        response_format=args.format,
    )

    # Save the transcription to the file
    if args.format == 'text':
        extension = 'json'
    elif args.format == 'srt':
        extension = 'json'
    elif args.format == 'vtt':
        extension = 'json'
    elif args.format == 'json' or args.format == 'verbose_json':
        extension = 'json'
    else:
        raise ValueError('Invalid format')

    # Remove extension from file name
    output_file = f'{os.path.splitext(args.file)[0]}.{extension}'
    with open(output_file, 'w', encoding='utf-8') as f:
        if args.format in ['json', 'verbose_json']:
            f.write(transcription.model_dump_json(indent=2))
        else:
            f.write(transcription)
        print(f'Saved transcription to {output_file}')

    # Print time elapsed for transcription
    total_elapsed = time.time() - start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds")


if __name__ == '__main__':
    main()
