"""Use a local exported folder or a published Hugging Face model ID."""
import argparse
import json
from gpt2_portfolio import load_bundle,generate_reply

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='ytyiury/gpt2-style-124m-sft')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--revision',help='Optional HF commit hash for an immutable release')
    parser.add_argument('--prompt')
    parser.add_argument('--chat',action='store_true')
    parser.add_argument('--max-new-tokens',type=int,default=120)
    parser.add_argument('--temperature',type=float,default=.7)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    bundle=load_bundle(args.model,device=args.device,revision=args.revision)
    def answer(prompt,history):
        reply,kept,info=generate_reply(bundle,prompt,history,max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature,seed=args.seed)
        print('\nModel:',reply)
        print(json.dumps(info))
        return kept+[(prompt,reply)] if reply and bundle.stage=='sft' else []
    if args.chat:
        history=[]
        print('Stage:',bundle.stage,'| /reset /exit')
        while True:
            try:prompt=input('\nYou: ').strip()
            except (EOFError,KeyboardInterrupt):break
            if prompt.lower() in ('/exit','/quit'):break
            if prompt.lower()=='/reset':history=[];continue
            if prompt:
                try:history=answer(prompt,history)
                except (ValueError,RuntimeError) as exc:print('Error:',exc)
    elif args.prompt:
        answer(args.prompt,[])
    else:parser.error('Provide --prompt TEXT or --chat')

if __name__=='__main__':main()
