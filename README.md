# 프로젝트 개요

직접 구현한 다양한 MCP서버의 컬렉션입니다.  
주요 기능은 코드 생성 및 실행, GitHub 저장소 분석, 추론, 웹 크롤링, 웹 검색, 유튜브 영상 요약 및 분석 등입니다.

## 주요 기능

### 🔍 웹 검색 (`web_search.py`)
- **`rini_google_search_base(...)`**: 구글 검색을 수행하고 결과를 반환합니다.
- **`rini_google_search_link_only(...)`**: 구글 검색 결과에서 링크만 추출합니다.
- **`rini_google_search_shallow(query: str)`**: 구글 검색을 수행하고 각 링크의 콘텐츠를 간략하게 가져옵니다.
- Stealth 브라우저를 사용하여 웹 페이지 콘텐츠를 가져오는 기능도 포함합니다.
- 기본 포트: 65000

### ▶️ 유튜브 영상 처리 (`youtube_summary.py`)
- **`rini_summarize_youtube_audio_only(url: str)`**: 유튜브 영상의 오디오만 요약합니다.
- **`rini_transribe_youtube_audio(url: str)`**: 유튜브 영상의 오디오를 텍스트로 변환합니다.
- **`rini_summarize_youtube_all(video_url: str)`**: 유튜브 영상의 전체 콘텐츠(키프레임, 오디오)를 분석하고 요약합니다.
- 오디오 다운로드, 키프레임 추출, 오디오 분할 및 개별 트랜스크립션, 프레임 캡션 생성 등의 세부 기능을 포함합니다.
- 기본 포트: 65001

### 📂 GitHub 저장소 분석 (`github_repo_analysis.py`)
- **`rini_github_analysis(query: str, url: str)`**: 지정된 GitHub 저장소를 분석하고 관련 정보를 제공합니다.
- 함수 및 클래스 추출, 코드 유사도 분석 등의 기능을 포함합니다.
- 기본 포트: 65002

### 🧠 추론 기능 (`reasoning.py`)
- **`rini_reasoning(query: str, model: str = None)`**: 주어진 쿼리에 대해 논리적 추론을 수행합니다.
- 기본 포트: 65003

### 💻 코드 생성 및 실행 (`coding.py`)
- **`rini_code_generation(query: str, model: str = None)`**: 주어진 쿼리를 기반으로 코드를 생성합니다.
- **`rini_python_code_execution(code: str)`**: 주어진 파이썬 코드를 실행합니다.
- 기본 포트: 65004

### 🌐 웹 크롤링 (`web_crawl.py`)
- **`rini_get_text_only_from_url(url: str)`**: 주어진 URL에서 텍스트 콘텐츠만 추출합니다.
- **`rini_get_all_from_url(url: str, timeout: int = 5)`**: 주어진 URL에서 모든 콘텐츠를 가져옵니다.
- 기본 포트: 65005


**필수 라이브러리 설치:**

프로젝트 실행에 필요한 라이브러리를 설치하려면 다음 명령어를 사용하십시오:

```bash
pip install -r requirements.txt
```

## 사용 방법

각 기능은 해당하는 Python 파일을 직접 실행하여 MCP 서버로 구동할 수 있습니다.  
예를 들어, 웹 검색 기능을 사용하려면 다음 명령어를 실행합니다:

```bash
python web_search.py
```

각 서버는 지정된 포트(예: 웹 검색 서버는 65000번 포트)에서 실행됩니다.
Rini API [server](https://github.com/mori-mmmm/Rini-API-server) / [client](https://github.com/mori-mmmm/Rini-API-client) 를 사용하면 쉽게 테스트 해보실 수 있습니다.

## 향후 개선 사항
- 각 기능에 대한 상세한 사용 예제 추가
- CLI 인터페이스 제공
- 통합 테스트 코드 작성
