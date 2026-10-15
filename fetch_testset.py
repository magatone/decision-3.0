"""deepghs/anime_dbrating에서 등급별 N장만 골라 받아 data/에 저장한다.

zip 하나가 ~18GB라 통째로 받지 않고, HTTP Range 요청으로 목차와 필요한 파일만 읽는다.
explicit 등급은 법무 확인 전이라 일부러 제외했다.
"""
import csv, io, random, sys, urllib.request, zipfile
from pathlib import Path

BASE = "https://huggingface.co/datasets/deepghs/anime_dbrating/resolve/main/"
ZIPS = {"general": "general_341506.zip", "sensitive": "sensitive_341769.zip",
        "questionable": "questionable_320107.zip"}
N, SEED = 333, 42
OUT = Path(__file__).parent / "data"
BLOCK = 1 << 20


class HttpFile(io.RawIOBase):
    """zipfile이 원격 파일을 로컬처럼 seek/read 하도록 Range 요청으로 감싼다."""
    def __init__(self, url):
        self.url, self.pos, self.cache = url, 0, {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.url = r.url  # CDN 주소로 고정해 리다이렉트 반복을 피한다
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = [off, self.pos + off, self.size + off][whence]
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            end = min((i + 1) * BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={i * BLOCK}-{end}"})
            with urllib.request.urlopen(req) as r:
                self.cache[i] = r.read()
        return self.cache[i]

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        out = bytearray()
        while self.pos < end:
            b = self._block(self.pos // BLOCK)
            s = self.pos % BLOCK
            chunk = b[s:s + end - self.pos]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)


def main():
    rows = []
    for label, name in ZIPS.items():
        dest = OUT / label
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{label}] 목차 읽는 중...", flush=True)
        hf = HttpFile(BASE + name)
        zf = zipfile.ZipFile(hf)
        files = [i for i in zf.infolist() if not i.is_dir()]
        picks = random.Random(SEED).sample(files, N)
        for k, info in enumerate(picks, 1):
            fname = Path(info.filename).name
            path = dest / fname
            if not path.exists():
                path.write_bytes(zf.read(info))
                hf.cache.clear()  # 파일마다 비워 메모리가 쌓이지 않게
            rows.append((f"{label}/{fname}", label))
            if k % 50 == 0:
                print(f"  {k}/{N}", flush=True)
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)
    print(f"완료: {len(rows)}장, {OUT / 'labels.csv'}")


if __name__ == "__main__":
    sys.exit(main())
