import json
import random
from datetime import datetime
from huggingface_hub import InferenceClient, AsyncInferenceClient
from ft_datasets.unified_dataset import get_template, make_background, default_rules
import asyncio
import time

client_dpo = AsyncInferenceClient(model="http://localhost:8303", timeout=100)

def process_citation_text(citation_text):
    cites = citation_text.split(',')
    return f"({','.join(cites)})"

class BufferedProcessor:

    def __init__(self, original_generator, buffer_length=50):
        self.buffer_length = buffer_length
        self.buffer = []
        self.original_generator = original_generator
        self.in_citation = False
        self.citation_text = ''

    def endwith(self, target):
        if len(self.buffer)<len(target):
            return False
        return ''.join(self.buffer[-len(target):])==target
        
    def process(self):
        for token in self.original_generator:
            for c in token:
                if not self.in_citation:
                    self.buffer.append(c)
                    if self.endwith('<cite>'):
                        self.in_citation = True
                        self.buffer = self.buffer[:-len('<cite>')]
                else:
                    self.citation_text+=c
                    stop_pos = self.citation_text.find("</cite>")
                    if stop_pos>=0:
                        for cc in process_citation_text(self.citation_text[:stop_pos]):
                            self.buffer.append(cc)
                        self.in_citation = False
                        self.citation_text = ""
            if len(self.buffer)>self.buffer_length:
                yield self.buffer.pop(0)
        while self.buffer:
            yield self.buffer.pop(0)

rules = (
    "1. Please cite your source of information, formatted as an article number enclosed in <cite></cite>, for example, <cite>1</cite>. If citing multiple sources at once, separate the numbers with a comma, like <cite>3,4</cite>."
)
data = []
with open('data/unified_valid_0104.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        if d['source'] in ['webglm', 'real_search']:
            data.append(d)
random.shuffle(data)

async def one_task(sem, prompt):
    async with sem:
        return await client_dpo.text_generation(prompt, max_new_tokens=512, stream=False, seed=42,
                                                    do_sample=True, temperature=0.7, top_p=0.95, top_k=40,
                                                    repetition_penalty=1.1,
                                                    return_full_text=True)


async def run_test():
    start = time.perf_counter()
    tasks = []
    sem = asyncio.Semaphore(1)
    for d in data:
        background = make_background(d['refs'])
        if 'date' in data and data['date'] is not None:
            date = data['date']
        else:
            date = datetime.now().strftime("%m-%d-%Y")
        input_prompt = get_template().format(
                background=background, 
                query=d['query'], 
                date=date,
                rules=rules,
                user_location=d.get('user_loc', 'America'))
        print(input_prompt)
        break
        tasks.append(asyncio.create_task(one_task(sem, "Who are you?")))
        if len(tasks)==50:
            break
    print('total test samples:', len(tasks))

    results = await asyncio.gather(*tasks)
    print(f"finish in {time.perf_counter()-start}s")
    
if __name__ == "__main__":
    asyncio.run(run_test())
        