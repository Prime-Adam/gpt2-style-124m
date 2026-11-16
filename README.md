# GPT-2-style 124M: custom pretraining and instruction fine-tuning

A custom dense decoder-only Transformer project by [Prime-Adam](https://github.com/Prime-Adam).
The project includes the architecture, historical Colab training notebooks, checkpoint
identification/export tools and portable CPU/CUDA inference.

**Release status:** this repository starter contains code and documentation. Trained
weights must be exported from the author's own checkpoints and uploaded before the
model/demo links below become usable. No trained checkpoint is bundled with this code.

| Artifact | Intended location |
|---|---|
| Code and methodology | [Prime-Adam/gpt2-style-124m](https://github.com/Prime-Adam/gpt2-style-124m) |
| Base text-continuation weights | [ytyiury/gpt2-style-124m-base](https://huggingface.co/ytyiury/gpt2-style-124m-base) |
| SFT assistant weights | [ytyiury/gpt2-style-124m-sft](https://huggingface.co/ytyiury/gpt2-style-124m-sft) |
| Inference Colab | [Open notebook](https://colab.research.google.com/github/Prime-Adam/gpt2-style-124m/blob/main/notebooks/inference_demo.ipynb) |
| Optional interactive Space | [ytyiury/gpt2-style-124m-demo](https://huggingface.co/spaces/ytyiury/gpt2-style-124m-demo) |

Start with [START_HERE.md](START_HERE.md) for account-specific publishing steps.

## Architecture

| Setting | Value |
|---|---|
| Unique learned parameters | 124,439,808 |
| Decoder blocks | 12 dense Transformer blocks |
| Hidden / FFN dimensions | 768 / 3072 |
| Attention heads | 12 × 64 dimensions; standard multi-head attention |
| Positional encoding | Learned absolute position embeddings |
| Total context | 1,024 tokens including prompt and response |
| Vocabulary / tokenizer | Original GPT-2 byte-level BPE, 50,257 tokens |
| End-of-text token | `<\|endoftext\|>`, ID 50,256 |
| MLP | GELU with tanh approximation |
| Normalization | LayerNorm, epsilon 1e-5; stage-specific placement below |
| Weight tying | Token embedding and output head share weights |
| Training dropout | 0.1 |

This is a GPT-2-style custom implementation. Reusing GPT-2's tokenizer does not imply
use of OpenAI's pretrained model weights. The legacy parameter names and forward
equations are preserved by `gpt2_portfolio/model.py`.

**Important source-code difference:** `gpt2.ipynb` uses **post-LayerNorm** blocks;
`gpt2_small_smol_smoltalk_sft_colab.ipynb` uses **pre-LayerNorm** blocks. These have
identical parameter names/shapes, but compute different functions. A successful
`load_state_dict(strict=True)` cannot detect that difference. This project therefore
stores `norm_order` explicitly in each exported release and tests both variants.
Check the actual training definition associated with a checkpoint before export.
The supplied notebooks do not establish why the normalization layout changed.

## Data and training methodology

The pretraining notebook prepares [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
configuration `sample-10BT`, with the original GPT-2 tokenizer and EOS delimiters.
Its saved additional-data snapshot contains 750,049,345 earlier training tokens and
1,250,052,856 additional tokens: 2,000,102,201 distinct positions in those merged files.
That is a **dataset snapshot count**, not a verified count of training tokens consumed
by an unknown release checkpoint. The continued-training code requests four passes
over the additional data and uses a separate checkpoint directory.

The SFT notebook streams [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk).
It assigns whole conversations to a deterministic approximately 98%/2% train/validation
split, then creates one training example for each assistant turn. Prompt tokens are
masked out of the loss; assistant tokens and EOS are supervised. Replies are capped
at 768 tokens during preparation; histories are fitted into the 1024-token context.

Saved preparation outputs report **1,010,041 training examples** and **20,774 validation
examples**. These are assistant-turn examples, not unique source conversations.
The SFT run requests one epoch, batch size 8, LR 2e-5, AdamW weight decay 0.01 and
warmup 200 updates (the later override in the notebook). Its saved output reports
126,299 updates completed. Actual checkpoint metadata should be published alongside
the selected release to confirm its identity and lineage.

### Design tradeoffs

| Observed design | Practical consequence |
|---|---|
| Small dense model | Provides an inspectable baseline with fewer serving dependencies than the later MoE project. |
| Learned absolute positions | Simple positional mechanism with a fixed trained 1024-position range. |
| Existing GPT-2 tokenizer | Keeps a stable byte-level BPE and avoids another tokenizer-training experiment. |
| Tied input/output weights | Reduces unique parameter storage; exports store the shared matrix once. |
| Assistant-only SFT loss | Directs fine-tuning updates toward answers rather than copying user prompts. |
| Separate base/SFT artifacts | Makes stage identity and prompting requirements explicit. |

These are consequences of the implementation. Personal motivations and unrecorded
experiments should be added by the author rather than inferred from notebook code.

## Reported results and limitations

| Source notebook output | Reported value | Interpretation |
|---|---:|---|
| Pretraining train loss / perplexity | 3.2683 / 26.27 | Sampled raw-token next-token evaluation |
| Pretraining validation loss / perplexity | 3.3342 / 28.05 | Sampled raw-token evaluation; 1,638,400 validation targets |
| Best SFT validation loss | 1.6416 | Assistant-only loss on separate SFT data |

These values are transcribed from saved notebook outputs. The actual trained weights
were not supplied with the notebook audit, so these results have **not** been
reproduced or attached to a verified public checkpoint. Pretraining and SFT losses
use different datasets and target masks and must not be interpreted as a direct
before/after quality improvement. No benchmark score is claimed.

Examples in the notebooks include factual hallucinations, repetitive text and invented
personal identities. This is a research/portfolio baseline with a short context and
limited factual reliability. See [docs/notebook_audit.md](docs/notebook_audit.md).

## Run inference

After the corresponding verified model bundle is published:

```bash
pip install -r requirements.txt
python inference.py --model ytyiury/gpt2-style-124m-sft --chat
python inference.py --model ytyiury/gpt2-style-124m-base --prompt "Once upon a time"
```

CPU is supported. CUDA uses BF16 where available and FP16 otherwise. The custom
inference path provides a KV cache, bounded generation, top-k/top-p sampling and a
mild repetition penalty. SFT history preserves recent complete exchanges and rejects
an oversized current question; base checkpoints receive raw text.

The exact SFT headers are separately tokenized pieces: `### User:\n`, content,
`\n\n`, and `### Assistant:\n`. Earlier assistant turns end with EOS. No new special
role tokens or invented chat template are added. The original notebooks' formatting
is reproduced by `gpt2_portfolio/tokenization.py`.

For an optional Gradio demo:

```bash
pip install -r requirements-demo.txt
python app.py
```

The model is loaded on first generation and reused. The demo labels base continuation
and SFT assistant behavior separately. See the optional Space instructions in the guide.

## Identify and export the author's checkpoints

```bash
python inspect_checkpoint.py /path/to/best.pt --sha256
python export_checkpoint.py /path/to/base_best.pt --stage base --norm-order post --output exports/base
python export_checkpoint.py /path/to/sft_best.pt --stage sft --norm-order pre --output exports/sft
```

For continued-pretraining bare `best.pt`, pass its matching `best_record.json` using
`--record`. The sidecar checksum is verified before its step/loss metadata is trusted.
Normalization order and head count cannot be identified from tensor shapes alone.
See the preparation Colab for the known original Drive paths and uploads from Colab.

Each inference bundle includes `model.safetensors`, explicit configuration, tokenizer,
source checkpoint SHA256, stage metadata, checksums and an automatically generated
Hugging Face model card. Optimizer/RNG data is omitted. Export preserves FP32 weights
and promotes FP16/BF16 exactly to FP32 rather than quantizing.

## Repository contents

- `gpt2_portfolio/`: exact architecture variants, inspection/export and inference code.
- `notebooks/prepare_hf_models.ipynb`: identify/export the two checkpoints in Colab.
- `notebooks/inference_demo.ipynb`: load published weights and run inference without Drive.
- `historical_training/`: output-cleared copies of the supplied training notebooks.
- `tests/`: CPU model-parity, checkpoint, tokenization, export/reload and demo checks.
- `docs/`: architecture caveats, dataset provenance and validation notes.

## License and attribution

A code/weight license has not been selected by this starter. Choose appropriate
licenses before expecting third parties to reuse a release. Dataset licenses and
attribution are separate; links to the original sources are in [docs/data_sources.md](docs/data_sources.md).
