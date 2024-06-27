import json 
import gradio as gr
import os
import requests
from huggingface_hub import InferenceClient
import traceback
# hf_token = os.getenv('HF_TOKEN')
client = InferenceClient(model="http://127.0.0.1:8090")

# api_url_nostream = os.getenv('API_URL_NOSTREAM')
headers = {
    'Content-Type': 'application/json',
}

default_system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

css = """.toast-wrap { display: none !important } """

SINGLE_TURN_TEMPLATE = (
    "You are a smart AI assistant. Please answer the following question from a user:\n"
    "{question}\n\n"
    "Below are information you find about the question(some of them may be irrelevant!):\n"
    "{background}\n\n"
    "and the dialog history between you and the user:\n"
    "{history}\n\n"
    "If the information you found does not include the necessary information to answer the question, tell the user that you can't find relevant information and suggest a few related questions about the entity. "
    "When recommend related questions, use a natural tone. "
    "Use \"information I found\" instead of \"informatino provided\"."
    "In this case, the entity info will not be shown to the user. As a result, DO NOT mention or cite the found information!!\n"
    "Please cite the local entity information you use ONLY IF it contains the answer to the question. When you cite, make sure to use in-text and in-words IEEE citations (just a number in square brackets).\n"
    "Reply:\n"
)

system_template = (
        "Below is some background information: \n"
        "{system}"
        "Answer questinos based on the given info, and provide the reference No. in the end of you answer."
        "If the question cannot be answered with the information given, please output a search query beginning with 'query:'"
    )

def yelp_entity_formatter(entity):
    def get_first_n_words(raw_text, n=50):
        split_text = raw_text.split()
        if len(split_text)>n:
            return ' '.join(split_text[:n])+'...'
        return raw_text
    
    def clean_desc_text(raw_desc):
        return "\n".join([x for x in raw_desc.splitlines() if len(x.strip())>0])
    
    default_keys = ['name','cuisine','price_range','address','rating','telephone','website']
    ret = ''
    for k in default_keys:
        if k in entity and entity[k] is not None:
            ret += f"{k.replace('_',' ').capitalize()}: {entity[k]}\n"
    if 'description' in entity and entity['description'] is not None:
        if 'description_text' in entity['description']:
            ret += f"Desc:\n{get_first_n_words(clean_desc_text(entity['description']['description_text']), 80)}\n"
        if 'description_list' in entity['description']:
            ret += f"Amentities:\n{', '.join(entity['description']['description_list'][:8])}\n"
    if 'hours' in entity and entity['hours'] is not None:
        ret+= 'Hours:\n'
        for k, v in entity['hours'].items():
            ret += f"{k}: {v}\n"
    if 'categories' in entity and entity['categories'] is not None:
        ret += f"Categories:\n{', '.join(entity['categories'])}\n"
    if 'popular_dishes' in entity and entity['popular_dishes'] is not None and len(entity['popular_dishes'])>0:
        ret += f"Popular dishes:\n{', '.join(x['name'] for x in entity['popular_dishes'])}\n"
    if 'reviews' in entity and isinstance(entity['reviews'], list):
        ret += "Reviews:\n"
        for r in entity['reviews'][:3]:
            ret += get_first_n_words(f"{r['date']}, {r['comment_text']}", n=50)+"\n"
    return ret

def transform_entity(entities):
    try:
        background = json.loads(entities)
    except:
        traceback.print_exc()
        background = []
        gr.Warning("Entity format error!")
    new_bg = []
    for e in background:
        new_bg.append(yelp_entity_formatter(e))
    return json.dumps(new_bg, indent=2)

def predict(message, chatbot, refs, entity_as_system, max_tokens, temperature, top_p, top_k, repetition_penalty):
    try:
        refs = json.loads(refs)
    except:
        refs = None
    background = ""
    if isinstance(refs, list) and len(refs)>0:
        for idx, e in enumerate(refs):
            background += f"[{idx+1}]\n{e}\n"
    else:
        gr.Warning("Empty reference")
    
    if len(background)>0:
        # use qa format
        if len(chatbot)>0:
            history = ""
            for interaction in chatbot:
                history += f"User: {str(interaction[0])}\nYou: {str(interaction[1]).replace(' </s>','')}\n"
        else:
            history = "EMPTY"
        input_prompt = SINGLE_TURN_TEMPLATE.format(
            question=message,
            background=background,
            history=history
        )
    else:
        # use chat format
        input_prompt = ""
        for interaction in chatbot:
            input_prompt += str(interaction[0]) + " [/INST] " + str(interaction[1]).replace(" </s>","") + " </s><s> [INST] "
            # input_prompt += str(interaction[0]) + " " + str(interaction[1]) + " <s> "
        input_prompt = input_prompt + str(message) + " [/INST] "

    print(input_prompt)
    partial_message = ""
    # print(entity_as_system, max_tokens, temperature, top_p, top_k, repetition_penalty) # strange
    for token in client.text_generation(input_prompt, max_new_tokens=max_tokens, stream=True, seed=42,
                                        do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                                        repetition_penalty=repetition_penalty,
                                        return_full_text=False):
        partial_message = partial_message + token
        yield partial_message

# Gradio Demo 
with gr.Blocks() as demo:

    with gr.Tab("Streaming"):
        gr.Markdown("""
                    # KBQA & Chat
                    References shold be `list of strings`.
                    """)
        with gr.Row():
            with gr.Column(scale=4):
                entities = gr.Code(label="Entities", language="json")
            with gr.Column(scale=4):
                refs = gr.Code(label="References", language="json")
        with gr.Row(scale=1):
            trans_btn = gr.Button("Transform Entity to Reference")
            clear_btn = gr.Button("Clear Reference")
        with gr.Accordion("Show Model Parameters", open=False):
            with gr.Row():
                with gr.Column():
                    entity_as_system = gr.Checkbox(False, label="Entity as system prompt")
                    max_tokens = gr.Slider(20, 512, label="Max Tokens", step=16, value=500)
                    temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=0.8)
                    top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.95)
                    top_k = gr.Slider(0, 100, label="Top K", step=1, value=40)
                    repetition_penalty = gr.Slider(0.0, 2.0, label="Repetition Penalty", step=0.1, value=1.1)
        gr.ChatInterface(predict, css=css, additional_inputs=[refs, entity_as_system, max_tokens, temperature, top_p, top_k, repetition_penalty])             
        trans_btn.click(transform_entity, inputs=[entities], outputs=[refs])
        clear_btn.click(lambda x: '', inputs=[refs], outputs=refs)
demo.queue(concurrency_count=75).launch(debug=True, server_name="0.0.0.0")