# Start here — publishing your first model project

Your accounts: **GitHub `Prime-Adam`**, **Hugging Face `ytyiury`**.
Use **one GitHub repository** for this architecture/project, with **two Hugging Face
model repositories** for its base and SFT checkpoints.

## 1. Create the GitHub repository

1. Sign in as `Prime-Adam` and open [github.com/new](https://github.com/new).
2. Repository name: **gpt2-style-124m**.
3. Description: **Custom 124M dense Transformer: pretraining, instruction fine-tuning and inference**.
4. Review the files locally, then choose **Public** for this code repository. The Colab
   installer uses its public GitHub URL. The Hugging Face weight repositories can remain
   private drafts while you validate them.
5. Leave automatic README/.gitignore/license initialization off: this package provides
   README and .gitignore. You can add your chosen license later.
6. Click **Create repository**.
7. Extract this starter ZIP on your PC. Open its `gpt2-style-124m` folder.
8. On GitHub choose **uploading an existing file**, or **Add file → Upload files**.
   Drag the files/folders INSIDE the extracted project folder into the upload area.
   Keep the folder hierarchy. Do not upload the ZIP as the project itself.
9. Commit with message **Add GPT-2-style architecture and release tools**.

You should see `README.md`, `app.py`, `gpt2_portfolio/` and `notebooks/` at the
repository root. Your URL is:

**https://github.com/Prime-Adam/gpt2-style-124m**

The README is already written from your notebooks. Review the architecture difference
and limitations before changing its release status. It does not claim that unavailable
weights or notebook metrics have been verified.

GitHub's [repository quickstart](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories)
describes the creation interface. The browser upload limit is 25MiB per file, while
ordinary Git blocks files over 100MiB; the starter contains only small source files.
See [GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## 2. Identify the two actual checkpoint files

Your notebooks point to these likely candidates in Drive:

| Stage | Relative to `MyDrive/GPT2_Project/` |
|---|---|
| Continued-pretraining base | `checkpoints_continued_2B/best.pt` |
| Matching base sidecar | `checkpoints_continued_2B/best_record.json` |
| SFT best | `sft_smoltalk_gpt2_v1/checkpoints_from_2B/best.pt` |
| SFT latest/recovery | `sft_smoltalk_gpt2_v1/checkpoints_from_2B/latest.pt` |

Open [the preparation notebook in Colab](https://colab.research.google.com/github/Prime-Adam/gpt2-style-124m/blob/main/notebooks/prepare_hf_models.ipynb).
After step 1 the link will resolve. Run its cells in order. It installs this repository's
package, mounts Drive and prints checkpoint metadata before exporting.

**Architecture check:** the supplied base notebook uses post-LayerNorm, the supplied
SFT notebook uses pre-LayerNorm. Keep `BASE_NORM_ORDER='post'` and
`SFT_NORM_ORDER='pre'` only if those are the definitions actually used for the selected
files. Edit them if you trained using another code version. Tensor shapes cannot answer
this question, and no unsafe automatic conversion is performed.

Prefer `best.pt` for a chosen validation checkpoint. `latest.pt` is the recovery state
and can include large optimizer data. The exporter can read either supported format,
but publishes only weights/configuration/tokenizer/provenance. It never modifies the
source checkpoints. These exports are for inference, not exact optimizer-state resume.

## 3. Create two Hugging Face model repositories

Sign in as `ytyiury`, open [huggingface.co/new](https://huggingface.co/new), and create:

- **ytyiury/gpt2-style-124m-base** — choose a Model repository.
- **ytyiury/gpt2-style-124m-sft** — choose a Model repository.

A private draft lets you validate before sharing publicly. Choose the release visibility
yourself. The [Hugging Face upload guide](https://huggingface.co/docs/hub/models-uploading)
describes model repositories, uploaded files and model cards.

Do not put `.pt` weights in the ordinary GitHub repository. Each FP32 inference release
is about 498MB of unique parameter data, plus tokenizer/metadata. Exporting both in Colab
avoids copying large files through your laptop.

## 4. Export and upload directly from Colab

In the preparation notebook:

1. Check the two checkpoint paths printed by the inspection cell.
2. Run export. It creates local `base` and `sft` inference folders with verified checksums.
3. Check the parent SHA256 comparison. A mismatch means the selected base file is not
   confirmed as the exact file used to initialize that SFT run; choose the matching
   original file or document the difference.
4. Add a Hugging Face **Write** token to Colab Secrets named **HF_TOKEN**, with notebook
   access enabled. Never paste it into a public notebook cell or README.
5. Set `UPLOAD_MODELS=True` in the last cell and run it. This uploads the verified
   inference folders to the two model repositories you created. It checks that the token
   belongs to `ytyiury`. No weights are uploaded until you run this cell yourself.

Each Hugging Face repository should contain:

```text
README.md                 # generated model card
config.json               # dimensions, stage and explicit norm order
model.safetensors         # unique weights, with tied head reconstructed by loader
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
provenance.json            # source checkpoint SHA256 and available training metadata
bundle.json                # per-file checksums
```

Additional tokenizer metadata may be present. Keep these files together. This custom
architecture uses the project's loader; do not advertise generic
`AutoModelForCausalLM.from_pretrained(...)` support without a separate integration.

## 5. Test the public inference demo

Open [inference_demo.ipynb](https://colab.research.google.com/github/Prime-Adam/gpt2-style-124m/blob/main/notebooks/inference_demo.ipynb)
in a fresh Colab runtime. It does not mount your Drive or run training.

1. Run installation.
2. Select `STAGE='sft'` and load. For a private model repository, add an HF read token
   to Colab Secrets as `HF_TOKEN`, enable notebook access and set `USE_PRIVATE_MODEL=True`.
   Public releases need no author credentials.
3. Run a question, then the simple interactive chat cell.
4. Select `STAGE='base'`, rerun loading and try raw text continuation.

The model requires no API provider. CPU works; CUDA speeds it up. The total prompt
and generated reply must stay within the learned 1024-token context. The generation
settings do not make a small model's factual answers dependable.

## 6. Optional: create a live Hugging Face Space

The Colab is a sufficient initial demo. For a live browser app, create
**ytyiury/gpt2-style-124m-demo** at [huggingface.co/new-space](https://huggingface.co/new-space)
and choose **Gradio**. Check the currently available plan/hardware options first;
this guide makes no claim that a persistent compute Space is free.

Upload/copy `app.py`, the `gpt2_portfolio/` package folder, and `requirements.txt` from
this project. In the Space's `requirements.txt`, append **gradio==6.30.0**. Its README
needs the YAML header shown in `docs/space_README.md`; set `app_file: app.py`.
The app loads the two published model IDs and lets visitors compare base continuation
and SFT responses. Public model repos are simplest; private ones require an appropriate
HF_TOKEN in the Space's Secrets, not in source code.

See [Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio) for deployment.

## 7. Finish the portfolio entry

After verifying the actual released checkpoints:

- Update README release status and model-card license/attribution.
- Confirm model links and demo links work in a fresh session.
- Add evaluation results tied to each model revision/hash. Keep the historical saved
  notebook values clearly marked until reproduced.
- Add a short paragraph in your own words explaining your motivations, what failed and
  what you learned. Avoid claiming that the normalization change was intentional unless
  you can confirm it.
- Keep one project repo for this model family. Use another small portfolio overview
  repository to link your different projects later.
