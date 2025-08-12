import os
import argparse
import time
import json

from openai import OpenAI

# Read base URL and API key from environment or fallback to default values
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.rpa.icu')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://lb01:12000')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'https://t.me/evilfreelancer')
DEFAULT_MODEL = 'large-v3-turbo'
DEFAULT_LANGUAGE = "ru"


def main():
    # Measure execution start time
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--format', type=str, choices=["json", "text", "srt", "verbose_json", "vtt"], default='json')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--language', type=str, default=DEFAULT_LANGUAGE)
    parser.add_argument('--raw', action='store_true', help='Save raw JSON response from API')
    args = parser.parse_args()

    # Open input file in binary mode
    input_file = open(args.file, "rb")

    # Initialize OpenAI client
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    # Request transcription from OpenAI API
    transcription = client.audio.transcriptions.create(
        language=args.language,
        file=input_file,
        model=args.model,
        response_format=args.format,
    )

    # Determine output file path and extension
    output_path = os.path.splitext(args.file)[0]

    # If --raw is set, save the full JSON response regardless of the format
    if args.raw:
        extension = 'json'
        output_file = f'{output_path}.{extension}'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
    else:
        # Handle standard formatting
        if args.format in ['json', 'verbose_json']:
            extension = 'json'
            content = transcription
        elif args.format == 'srt':
            extension = 'srt'
            content = json.loads(transcription)['text']
        elif args.format == 'text':
            extension = 'txt'
            content = json.loads(transcription)['text']
        else:
            raise ValueError(f"Unsupported format: {args.format}")

        output_file = f'{output_path}.{extension}'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

    # Print result file path
    print(f'Saved transcription to {output_file}')

    # Print total execution time
    total_elapsed = time.time() - start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds")


if __name__ == '__main__':
    main()
