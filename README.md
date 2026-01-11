# SentCodeLM

Sentence-step autoregressive planner + non-autoregressive renderer. The planner predicts a discrete sentence plan (RVQ codes) and a continuous residual; the renderer emits byte strings in parallel and is trained with sentence-level reward (no token-level LM loss, no sentence bank).

## Setup

Requires Python 3.10+ and `uv`.

```
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Data format

Create `data/raw.txt` with one document per line.

## Path B: Build raw.txt from Hugging Face

```
python bin/00_list_hf_configs.py --dataset epfml/FineWeb2-HQ
python bin/00_build_raw_from_hf.py \
  --out data/raw.txt \
  --target_lines 300000 \
  --weights comma=0.6,fineweb_edu=0.3,fineweb2_hq=0.1 \
  --fineweb_edu_name sample-10BT \
  --fineweb2_hq_config en
```

## Pipeline

```
python 01_make_sentences.py --raw data/raw.txt --out_dir data
python 02_train_sentence_encoder.py --data_dir data --out_dir out
python 03_embed_sentences.py --data_dir data --out_dir data --encoder out/enc.pt
python 04_fit_rvq.py --emb data/sent_emb.npy --out_dir out
python 04b_encode_codes.py --emb data/sent_emb.npy --rvq out/rvq.pt --out_dir data
python 05_build_datasets.py --data_dir data --out_dir data
python 06_train_planner.py --train_data data/packed_train.pt --val_data data/packed_val.pt --rvq out/rvq.pt --out_dir out
python 07_train_renderer_reward.py --train_data data/packed_train.pt --encoder out/enc.pt --rvq out/rvq.pt --out_dir out
python 08_generate.py --prompt \"Write two concise sentences about the ocean.\" --steps 2
```

## Interactive

```
python 09_chat.py --steps 1
```

Optional HTTP server:

```
python 10_server.py --port 8000
```

POST JSON to `/generate` with `prompt`, optional `steps`, `sample`, and `temperature`.

## Notes

- The encoder outputs a compact sentence embedding (`d_emb`, default 128) used for RVQ and contrastive losses. Internal transformer width stays at `d_model=512`.
- The renderer uses only sentence-level reward (cosine in encoder space + length/repetition penalties). No autoregressive decoding or teacher-forced token loss is used.
