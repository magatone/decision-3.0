# decision-3.0: d3-mini 멀티모달 판단 모델 로컬 검증

오픈소스 판단 모델 [`vllm-sr/d3-mini`](https://huggingface.co/vllm-sr/d3-mini)(Decision 3.0)를 로컬 GPU에서 실행했습니다.
그리고 **간단한 웹 GUI를 만들어 텍스트·이미지·영상 입력(멀티모달)에 대한 판단 성능을 검증**했습니다.

- 검증일: 2026-10-14 ~ 10-15
- 환경: Windows 11, RTX 5090 32GB
- 실행 방식: Windows 네이티브(WSL 없음)와 Docker(Linux 컨테이너) 두 가지 모두 확인

## 이 프로젝트에서 한 일

1. 오픈소스 모델 d3-mini를 Windows 환경에서 설치하고 실행했습니다. 가능한 실행 방식과 불가능한 실행 방식을 확인했습니다.
2. 브라우저용 테스트 GUI를 만들었습니다. 파이썬 표준 라이브러리 HTTP 서버에 HTML 한 장을 붙인 형태라, 추가로 설치할 웹 프레임워크가 없습니다.
3. GUI로 멀티모달 판단 테스트를 진행했습니다.
   - 텍스트 분류
   - 이미지 1장 판단
   - 이미지 A·B 비교
   - 여러 장 일괄 예/아니오 분류
   - 처리 속도
4. Docker로 패키징했습니다. 사내망 직접 접속과 Cloudflare 터널 공개 두 가지 방식을 모두 지원합니다.

## 활용한 오픈소스

이 저장소에는 **모델 가중치와 외부 데이터가 포함되어 있지 않습니다.** 모델은 실행할 때 Hugging Face에서 받거나, 로컬 캐시를 연결해서 사용합니다.

| 오픈소스 | 용도 | 라이선스 |
|---|---|---|
| [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini) | 판단 모델 본체 (Qwen3.5-4B 기반 파인튜닝, 4.54B 파라미터) | Apache-2.0 (모델 카드 기준) |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) 5.17.0 | 모델 로드·추론 | Apache-2.0 |
| [PyTorch](https://pytorch.org) 2.11.0 (CUDA 12.8) | GPU 연산 | BSD-3-Clause |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 0.5.2 | 고속 커널 (Docker/Linux에서만 사용) | MIT |
| accelerate, safetensors, huggingface_hub, einops, Pillow, opencv-python-headless | 모델 실행 보조, 이미지·영상 디코딩 | 각 프로젝트 라이선스 |
| [cloudflared](https://github.com/cloudflare/cloudflared) 2025.9.1 | 외부 공개용 터널 (선택) | Apache-2.0 |

모델 코드를 저장소에서 그대로 실행하기 때문에(`trust_remote_code`) revision을 `6d8971f464d6f9d4c5af7d9a8d75f3aa03be4f59`로 고정했습니다.

> 라이선스는 2026-10-15 기준 각 프로젝트 표기를 따른 것입니다. 외부에 배포하기 전에 원문을 다시 확인하세요.

## 모델 개요

- 텍스트를 생성하지 않습니다. 주어진 입력(텍스트, 이미지, 영상)에 대해 **질문별 확률만 한 번에 반환**하는 판단 모델입니다.
- 질문 유형은 세 가지입니다.
  - `choice`: 보기 중 하나 고르기
  - `noul`: 예/아니오 확률
  - `score`: 순서가 있는 점수
- 이미지는 장당 최대 1.6MP로 줄여서 처리합니다. 영상은 2fps로, 최대 32프레임까지 봅니다.
- 하지 못하는 것: 설명문 생성, 위치(좌표) 찾기, OCR.

## GUI로 진행한 검증

### GUI 기능

- **단일 테스트:** 입력 문장과 이미지·영상을 첨부하고, 질문은 JSON으로 편집해서 실행합니다. 실행은 버튼이나 Ctrl+Enter로 합니다.
  - 결과는 질문별 확률 막대로 표시되고, 처리 시간(초)도 함께 나옵니다.
  - 예제가 준비되어 있습니다: 고객문의 분류(한국어), 텍스트 예제, 이미지 1장 판단, 이미지 A vs B 비교.
- **일괄 판단:** 파일 여러 개나 폴더를 선택하고 예/아니오 질문 하나를 입력하면, 한 장씩 처리해서 Yes/No 상자에 썸네일로 분류합니다.
  - Yes 기준(%) 슬라이더로 다시 분류할 수 있습니다. 모델을 다시 실행하지 않습니다.
  - 썸네일을 클릭하면 원본 비율 팝업이 열립니다. ←/→로 같은 상자 안에서 이동하고, 상단에 판정 확신도가 표시됩니다.
  - 지원 형식은 png, jpg, webp입니다.

### 결과 요약

| 항목 | 결과 |
|---|---|
| 모델 카드 예제 3종 | 기대값과 모두 일치 |
| 한국어 고객문의 분류 | 동작 확인 ("결제했는데 회차가 안 열려요" → payment) |
| 질문 언어 비교 (문의 8건 × 질문 2개) | 영어 13/16, 한국어 13/16, 한국어 문의 + 영어 질문 13/16. 틀린 문항도 모두 같음 → 이 표본에서는 언어 차이 없음 |
| 이미지 A·B 순서 구분 | 순서를 바꿔 8/8 정답 (약 99.9%) |
| 일괄 예/아니오 분류 (30장, "영수증인가요?") | 영수증 6장을 정확히 Yes로 분리 |

검증 이미지는 모델 저장소에 포함된 예제 이미지(영수증, 배너, 차트)를 복사해서 사용했습니다.

### 처리 속도 (RTX 5090)

| 항목 | Windows 네이티브 | Docker (Linux) |
|---|---|---|
| 텍스트 1건 | 약 0.055~0.07초 | 약 0.044초 |
| 이미지 1장 | 약 0.3~0.5초 | 약 0.18초 (일괄 판단 평균) |
| 일괄 판단 30장 | 9.6초 | **6.9초** |
| 모델 로드 | 약 105초 (최초 다운로드 포함) | 약 70초 |

Docker가 더 빠른 이유는 Linux에서만 설치되는 고속 커널(`flash-linear-attention`)로 보입니다. 이 커널만 켜고 끄며 따로 비교하지는 않았습니다.

### 주의할 점 (검증 중 확인)

- **질문이 이미지와 맞지 않아도 모델은 답을 거부하지 않습니다.** 캐릭터가 없는 영수증 2장에 "같은 캐릭터인가요?"라고 물었더니 "예 97%"가 나왔습니다. 그래서 실제 도입 전에는 정답이 붙은 실제 데이터로 정확도와 기준선을 측정해야 합니다.
- **같은 GPU에 모델을 2개 띄우면 오히려 느려집니다.** 텍스트 처리량이 초당 16.9건에서 1.4건으로 떨어졌습니다. 그래서 모델은 1개만 띄우고 순서대로 처리합니다.
- 모델은 **이미지를 먼저, 영상을 나중에** 읽습니다. A·B에 이미지와 영상을 섞어 넣으면 순서가 바뀝니다.

## 실행 방법

### 1) Docker (권장)

NVIDIA GPU를 쓸 수 있는 Docker 환경(Windows는 Docker Desktop + WSL2)이 필요합니다.

```bat
docker compose up -d --build              :: 이 PC: http://127.0.0.1:7860 , 사내망: http://<이 PC IP>:7860
docker compose logs -f gui                :: "ready"가 뜨면 사용 가능 (모델 로드 약 70초)
docker compose --profile tunnel up -d     :: Cloudflare 터널까지 함께 실행 (.env 필요)
docker compose down                       :: 종료
```

- 첫 빌드는 약 10분 걸리고, 이미지 크기는 12.2GB입니다.
- 모델은 호스트의 Hugging Face 캐시(`%USERPROFILE%\.cache\huggingface\hub`)를 **읽기 전용으로 연결**해서 씁니다. 캐시에 모델이 없다면 먼저 아래 2)의 `quickstart.py`를 한 번 실행해서 받아 두세요. 컨테이너는 오프라인 모드로 실행됩니다.
- **사내망에서 접속하려면** Windows 방화벽에서 7860 인바운드를 허용해야 합니다. 관리자 PowerShell에서 실행하세요.

  ```powershell
  New-NetFirewallRule -DisplayName "d3 GUI 7860" -Direction Inbound -Protocol TCP -LocalPort 7860 -RemoteAddress LocalSubnet -Action Allow
  ```

- **Cloudflare 터널 설정:**
  1. Cloudflare Zero Trust에서 터널을 만들고, 토큰을 복사합니다.
  2. `.env.example`을 `.env`로 복사하고 `TUNNEL_TOKEN`에 토큰을 넣습니다.
  3. Public hostname의 service를 **`http://gui:7860`**으로 지정합니다. `localhost`가 아닙니다.
- **로그인(선택):** `.env`에 `GUI_AUTH=아이디:비밀번호`를 넣으면 브라우저 로그인 창이 뜹니다. 비워 두면 로그인 없이 열립니다.
- `gui.html`을 수정했다면 `docker compose up -d --build`로 다시 빌드해야 반영됩니다.

### 2) Windows 네이티브

```bat
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

.venv\Scripts\python.exe quickstart.py    :: 예제 실행 + 시간 측정 (처음이면 모델 8.5GB 다운로드)
run_gui.bat                               :: 테스트 GUI http://127.0.0.1:7860
run_server.bat                            :: 모델 동봉 HTTP API 서버 http://127.0.0.1:8899
```

- 패키지 버전은 `requirements.txt`에 검증한 버전 그대로 고정되어 있습니다. RTX 50 시리즈는 **cu128 이상의 torch**가 필요해서 CUDA 12.8용 torch를 받도록 지정했습니다.
- **모델(8.5GB)은 requirements에 들어 있지 않습니다.** 처음 실행할 때 Hugging Face에서 고정된 revision으로 자동으로 받아 `%USERPROFILE%\.cache\huggingface\hub`에 저장합니다.
- Docker와 같은 7860 포트를 쓰므로, 둘 중 하나만 켜야 합니다.
- Windows에서 안 되는 것: `vllm-sr serve -e`(Engine 모드). 설치 스크립트가 macOS와 Linux만 지원합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `gui.py`, `gui.html` | 테스트 GUI (서버 + 화면) |
| `Dockerfile`, `compose.yaml`, `.env.example` | Docker 실행 설정 |
| `requirements.txt` | 파이썬 패키지 (검증한 버전 고정, Windows·Docker 공용) |
| `quickstart.py` | 예제 실행, 요청별 처리 시간·GPU 메모리 출력 |
| `run_gui.bat`, `run_server.bat` | Windows 실행 스크립트 |
| `req.json` | API 서버 테스트용 요청 예시 |
| `fetch_testset.py` | 공개 데이터셋에서 테스트 이미지를 일부만 받는 스크립트 (받은 데이터는 저장소에 넣지 않음) |

## 보안 메모

- GUI는 기본 설정에서 로그인 없이 열립니다. 외부에 공개할 때는 `GUI_AUTH`를 설정하거나 Cloudflare Access 같은 인증을 앞에 두는 것을 권장합니다.
- 요청 1건이 64MB를 넘으면 거절합니다.
- 미공개 작품이나 개인정보가 담긴 이미지는 테스트에 쓰지 않았습니다.
