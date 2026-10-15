# d3-mini Windows native smoke test: README Quickstart (text / image / video) + timing.
# Run: .venv\Scripts\python.exe quickstart.py
import json
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel

ROUTE = {
    "type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {
        "returns": "Refunds, replacements and damaged deliveries",
        "billing": "Payments, invoices and charges",
        "technical": "Product setup and faults",
    },
}


def run(label, **kw):
    model.system_one(**kw)  # warm-up (first call compiles / allocates)
    t = time.perf_counter()
    result = model.system_one(**kw)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000
    print(f"\n== {label} ({ms:.1f} ms) ==")
    print(json.dumps(result["answers"], indent=2, ensure_ascii=False))
    return result


print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

t = time.perf_counter()
# Pinned: trust_remote_code runs the repo's .py files, so don't pull unreviewed updates silently.
REV = "6d8971f464d6f9d4c5af7d9a8d75f3aa03be4f59"
model = AutoModel.from_pretrained("vllm-sr/d3-mini", revision=REV, trust_remote_code=True)
print(f"load {time.perf_counter() - t:.1f}s")

run("text", state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    questions={
        "route": ROUTE,
        "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
        "urgency": {"type": "score", "instructions": "How urgent is this request?",
                    "criteria": ["Routine", "Soon", "Today"]},
    })

run("image", state="The customer says the blender arrived cracked and attached the receipt.",
    images=[hf_hub_download("vllm-sr/d3-mini", "assets/example-receipt.png")],
    questions={
        "route": ROUTE,
        "on_receipt": {"type": "noul", "instructions": "Does the receipt list the blender?"},
        "payment": {"type": "choice", "instructions": "How was the order paid?",
                    "criteria": {"card": None, "cash": None, "gift card": None}},
    })

run("video", state="A short clip from a test camera.",
    videos=[hf_hub_download("vllm-sr/d3-mini", "assets/example-video.mp4")],
    questions={
        "direction": {"type": "choice", "instructions": "Which way does the square move?",
                      "criteria": {"right": "From left to right", "left": "From right to left",
                                   "still": "It does not move"}},
        "color_change": {"type": "noul", "instructions": "Does the square change color?"},
    })

# Korean input check (webtoon-ish), not in the README
run("korean", state="결제했는데 회차가 안 열려요. 코인은 빠져나갔고 지금 바로 보고 싶어요.",
    questions={
        "route": {"type": "choice", "instructions": "어느 팀이 처리해야 하나요?",
                  "criteria": {"payment": "결제·코인·환불", "content": "작품·회차 오류", "account": "로그인·계정"}},
        "charged": {"type": "noul", "instructions": "코인이 차감되었나요?"},
    })

print(f"\npeak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print(json.dumps(model.info() if hasattr(model, "info") else {}, indent=2, default=str)[:3000])
