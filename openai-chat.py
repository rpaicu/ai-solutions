import os
import argparse
from openai import OpenAI

# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:4000')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.rpa.icu')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://gpu01:4000')
# OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://gpu02:13000')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'https://t.me/evilfreelancer')
DEFAULT_MODEL = 'deepseek-r1:8b'


def main():
    parser = argparse.ArgumentParser(description='DeepSeek Chat Completion')
    parser.add_argument('message', type=str, help='Input message for the model')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{
                "role": "user",
                "content": args.message
            }],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            seed=args.seed,
            stream=False
        )

        print("\nResponse:")
        print(response.choices[0].message.content)
        print(f"\nUsage: {response.usage}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
