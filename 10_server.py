import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from torch.nn import functional as F

from tools.data import BYTE_EOS, BYTE_PAD, BYTE_VOCAB_SIZE, bytes_to_text
from tools.encoder import ByteEncoder
from tools.planner import Planner
from tools.renderer import Renderer
from tools.rvq import load_rvq
from utils import split_sentences


def sample_lengths(len_logits):
    logits = len_logits[:, 1:]
    probs = F.softmax(logits, dim=-1)
    lengths = torch.multinomial(probs, num_samples=1).squeeze(1) + 1
    return lengths


def greedy_lengths(len_logits):
    logits = len_logits[:, 1:]
    return logits.argmax(dim=-1) + 1


def sample_tokens(logits, temperature: float):
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
    tokens = tokens.view(probs.size(0), probs.size(1))
    return tokens


def greedy_tokens(logits):
    return logits.argmax(dim=-1)


def encoder_input_ids(text: str, max_len: int, device):
    data = text.encode("utf-8", errors="ignore")
    ids = list(data)[:max_len]
    if len(ids) < max_len:
        ids.append(BYTE_EOS)
    if len(ids) < max_len:
        ids += [BYTE_PAD] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long, device=device)


def render_sentence(renderer, codes, resid, sample: bool, temperature: float):
    with torch.no_grad():
        logits, len_logits = renderer(codes.unsqueeze(0), resid.unsqueeze(0), ctx=None)
        if sample:
            lengths = sample_lengths(len_logits)
            tokens = sample_tokens(logits, temperature)
        else:
            lengths = greedy_lengths(len_logits)
            tokens = greedy_tokens(logits)
        row = tokens[0, : lengths.item()].tolist()
        text = bytes_to_text([t for t in row if 0 <= t < 256])
    return text.strip()


def build_app(models):
    encoder, rvq, planner, renderer, device, max_len = models

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                self.send_response(404)
                self.end_headers()
                return
            content_len = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_len)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self.end_headers()
                return

            prompt = payload.get("prompt", "")
            steps = int(payload.get("steps", 1))
            sample = bool(payload.get("sample", False))
            temperature = float(payload.get("temperature", 1.0))

            sentences = split_sentences(prompt)
            if not sentences:
                out = {"text": ""}
                self.respond_json(out)
                return

            with torch.no_grad():
                codes_seq = []
                resid_seq = []
                for sent in sentences:
                    ids = encoder_input_ids(sent, max_len, device)
                    emb = encoder(ids)
                    codes, _, resid = rvq.encode(emb)
                    codes_seq.append(codes.squeeze(0))
                    resid_seq.append(resid.squeeze(0))

                for _ in range(steps):
                    codes_t = torch.stack(codes_seq).unsqueeze(0).to(device)
                    resid_t = torch.stack(resid_seq).unsqueeze(0).to(device)
                    lengths = torch.tensor([codes_t.size(1)], device=device)
                    code_logits, pred_resid, _ = planner(codes_t, resid_t, lengths=lengths)
                    next_codes = torch.stack([logits[0, -1].argmax(dim=-1) for logits in code_logits])
                    next_resid = pred_resid[0, -1]

                    text = render_sentence(renderer, next_codes, next_resid, sample, temperature)
                    sentences.append(text)
                    codes_seq.append(next_codes)
                    resid_seq.append(next_resid)

            out = {"text": " ".join(sentences)}
            self.respond_json(out)

        def respond_json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(args):
    device = torch.device(args.device)

    enc_ckpt = torch.load(args.encoder, map_location=device)
    encoder = ByteEncoder(**enc_ckpt["config"])
    encoder.load_state_dict(enc_ckpt["model"])
    encoder.to(device)
    encoder.eval()

    rvq = load_rvq(args.rvq, device=device)

    plan_ckpt = torch.load(args.planner, map_location=device)
    planner = Planner(**plan_ckpt["config"])
    planner.load_state_dict(plan_ckpt["model"])
    planner.to(device)
    planner.eval()

    rend_ckpt = torch.load(args.renderer, map_location=device)
    renderer = Renderer(**rend_ckpt["config"], vocab_size=BYTE_VOCAB_SIZE)
    renderer.load_state_dict(rend_ckpt["model"])
    renderer.to(device)
    renderer.eval()

    handler = build_app((encoder, rvq, planner, renderer, device, enc_ckpt["config"]["max_len"]))
    server = HTTPServer((args.host, args.port), handler)
    print(f"listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--encoder", default="out/enc.pt")
    ap.add_argument("--rvq", default="out/rvq.pt")
    ap.add_argument("--planner", default="out/planner.pt")
    ap.add_argument("--renderer", default="out/renderer.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    main(args)
