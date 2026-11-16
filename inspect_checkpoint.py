"""Example: python inspect_checkpoint.py /path/to/best.pt --sha256"""
import argparse
import json
from gpt2_portfolio.checkpoints import inspect_checkpoint

def main():
    parser=argparse.ArgumentParser(description='Identify a legacy checkpoint without dumping weights/optimizer tensors')
    parser.add_argument('checkpoint')
    parser.add_argument('--record',help='Matching continued-pretraining best_record.json, if available')
    parser.add_argument('--sha256',action='store_true',help='Hash the complete source file (can take time on Drive)')
    args=parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint,args.record,args.sha256),indent=2,default=str))

if __name__=='__main__':main()
