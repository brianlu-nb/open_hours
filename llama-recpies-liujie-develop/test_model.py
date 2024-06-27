import json
import random
from datetime import datetime
from huggingface_hub import InferenceClient
from ft_datasets.unified_dataset import get_template, make_background, default_rules

client = InferenceClient(model="http://localhost:8081", timeout=100)
client_dpo = InferenceClient(model="http://localhost:8300", timeout=100)

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



print('total test samples:', len(data))
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
    print('='*10)
    print(d['query'])
    # print('-'*10,' old ', '-'*10)
    # in_citation = False
    # citation_text = ""
    # for c in BufferedProcessor(client.text_generation(input_prompt.strip(), max_new_tokens=512, stream=True, seed=42,
    #                                             do_sample=True, temperature=0.7, top_p=0.95, top_k=40,
    #                                             repetition_penalty=1.1,
    #                                             return_full_text=True)).process():
    #     print(c, end='')
    # print()
    print('-'*10,' dpo ', '-'*10)
    in_citation = False
    citation_text = ""
    for c in BufferedProcessor(client_dpo.text_generation(input_prompt.strip(), max_new_tokens=512, stream=True, seed=42,
                                                do_sample=True, temperature=0.7, top_p=0.95, top_k=40,
                                                repetition_penalty=1.1,
                                                return_full_text=True)).process():
        print(c, end='')

    print()
    c = input("press enter to continue")
    