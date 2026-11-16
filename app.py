"""Optional Gradio demo. Importing this file neither downloads nor starts a model."""
from functools import lru_cache
import json
import os
import threading
from gpt2_portfolio import load_bundle,generate_reply

MODEL_IDS={
    'SFT — assistant prompt':os.environ.get('SFT_MODEL_ID','ytyiury/gpt2-style-124m-sft'),
    'Base — raw text continuation':os.environ.get('BASE_MODEL_ID','ytyiury/gpt2-style-124m-base'),
}
_lock=threading.Lock()

@lru_cache(maxsize=2)
def get_model(model_id):
    return load_bundle(model_id,device=os.environ.get('MODEL_DEVICE','auto'))

def respond(variant,prompt,max_new_tokens=120,temperature=.7,seed=42):
    if variant not in MODEL_IDS:raise ValueError('Choose one of the configured checkpoint releases')
    with _lock:
        bundle=get_model(MODEL_IDS[variant])
        reply,_,info=generate_reply(bundle,prompt,max_new_tokens=int(max_new_tokens),
                                    temperature=float(temperature),seed=int(seed))
    return reply,json.dumps(info,indent=2)

def create_demo():
    import gradio as gr
    return gr.Interface(fn=respond,inputs=[
        gr.Dropdown(list(MODEL_IDS),value=next(iter(MODEL_IDS)),label='Checkpoint'),
        gr.Textbox(lines=4,label='Question for SFT, or text to continue for Base'),
        gr.Slider(1,256,value=120,step=1,label='Maximum new tokens'),
        gr.Slider(0,1.5,value=.7,step=.05,label='Temperature (0 = greedy)'),
        gr.Number(value=42,precision=0,label='Seed')],
        outputs=[gr.Textbox(label='Generated text'),gr.Textbox(label='Generation details')],
        title='GPT-2-style 124M: Base and SFT',
        description='Custom dense Transformer with learned absolute positions. Base: post-LayerNorm; SFT: pre-LayerNorm as in the supplied notebooks. '
                    'Small research baseline: generated facts may be inaccurate.',
        examples=[[next(iter(MODEL_IDS)),'Explain gravity in simple terms.',120,.7,42]],
        cache_examples=False)

if __name__=='__main__':
    create_demo().queue(default_concurrency_limit=1).launch()
