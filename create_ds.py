import pandas as pd
import asyncio
import os
import re

contents = []
columns = ["content", "class"]

async def process(folder):
    to_return = []
    for file in os.listdir(f'./prompts/{folder}/output/'):
        f =  open(file=f'./prompts/{folder}/output/{file}')
        data = f.read().splitlines()
        pattern = r'^\d+\.\s*'
        text = [re.sub(pattern, '', text) for text in data]
        for line in text:
            to_return.append(line)
    return to_return

async def main():
    prompt_dir = os.listdir('./prompts')

    tasks = []
    for folder in prompt_dir:
        tasks.append(asyncio.gather(process(folder)))

    results = await asyncio.gather(*tasks)

    for i, content  in enumerate(results):
        for line in content[0]:
            data = [line, prompt_dir[i]]
            contents.append(data)

    df = pd.DataFrame(data=contents, columns=columns)
    df.to_csv('data.csv', index=False)


asyncio.run(main())
