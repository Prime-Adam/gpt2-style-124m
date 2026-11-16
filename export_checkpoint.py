"""Convert one notebook checkpoint into a compact, identified HF inference bundle."""
import argparse
import json
from pathlib import Path
from gpt2_portfolio.exporting import export_checkpoint
from gpt2_portfolio.model import ModelConfig

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--stage',required=True,choices=['base','sft'])
    parser.add_argument('--norm-order',required=True,choices=['pre','post'],help='Read the actual training code: not inferable from tensor shapes')
    parser.add_argument('--output',required=True)
    parser.add_argument('--record')
    parser.add_argument('--tokenizer',default='openai-community/gpt2',help='Original GPT-2 tokenizer repo or saved tokenizer directory')
    args=parser.parse_args()
    path=export_checkpoint(args.checkpoint,args.output,stage=args.stage,norm_order=args.norm_order,
                           tokenizer_source=args.tokenizer,record_path=args.record)
    print('Exported:',path)
    print(json.dumps(json.loads((Path(path)/'config.json').read_text()),indent=2))

if __name__=='__main__':main()
