import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

client = AsyncOpenAI(base_url=os.getenv('BASE'), api_key=os.getenv("API"))

SYSTEM_PROMPT = """You generate realistic mobile notification bodies only.
Each is 5–25 words, no titles, sender names, or extra text.
Match the tone, urgency, and topic in the user's prompt.
Output only a numbered list, no explanations.
"""

# Process a single prompt file
async def process_prompt(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1.5
        )
        data = response.choices[0].message.content

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(data)

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

async def main():
    tasks = []

    for i in range(1, 21):
        file_path = f'prompts/low/{i}.txt'
        output_path = f'prompts/low/output/{i}.txt'
        tasks.append(process_prompt(file_path, output_path))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
