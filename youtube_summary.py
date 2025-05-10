import httpx
from mcp.server.fastmcp import FastMCP
import yt_dlp
import asyncio
from functools import partial
from util_ai import stt, summarize, get_t2t
import os
import subprocess
import uuid
from typing import List, Dict, Tuple, Optional, Any
from pydub import AudioSegment


mcp = FastMCP("Rini Youtube Summarize")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = 65001

def get_youtube_url(song_title):
    ydl_opts = {
        'default_search': 'ytsearch',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(song_title, download=False)
        video_url = result['entries'][0]['webpage_url']
        return video_url
    
async def download_audio_from_youtube(output_file, folder=True, video_url=None):
    if folder:
        output_file = 'music/' + output_file
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_file,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(ydl.download, [video_url]))

@mcp.tool()
async def rini_summarize_youtube_audio_only(url: str) -> str:
    """
    유튜브 영상을 소리만 stt를 거쳐 텍스트로 만든 후 요약합니다.
    """
    output_file = "temp_yt_audio"
    try:
        await download_audio_from_youtube(output_file, False, url)
    except Exception as e:
        return f"유튜브 로드 실패: {e}"

    txt = await stt(f'{output_file}.mp3')
    ret = await summarize(txt, None)
    return ret

@mcp.tool()
async def rini_transribe_youtube_audio(url: str) -> str:
    """
    유튜브 영상을 소리만 stt를 거쳐 텍스트로 만든 후 반환합니다.
    """
    output_file = "temp_yt_audio"
    try:
        await download_audio_from_youtube(output_file, False, url)
    except Exception as e:
        return f"유튜브 로드 실패: {e}"

    txt = await stt(f'{output_file}.mp3')
    return txt


from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from google import genai
from google.genai import types
google_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
)

import base64

def img2base64(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'")
        return None

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_encoded_bytes = base64.b64encode(image_data)
            base64_encoded_string = base64_encoded_bytes.decode('utf-8')

            return base64_encoded_string
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def get_it2t(image_path, prompt):
    #model = "gemini-2.5-pro-preview-03-25"
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="""image/png""",
                    data=base64.b64decode(img2base64(image_path)),
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        #thinking_config = types.ThinkingConfig(
        #    thinking_budget=0,
        #),
        response_mime_type="text/plain",
    )

    ret = google_client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return ret.candidates[0].content.parts[0].text

###############################################################################
# 1. 유튜브 영상 다운로드 (yt-dlp)
###############################################################################

def download_video(video_url: str) -> str:
    """
    yt-dlp를 사용해 유튜브 영상을 다운로드하고, 로컬 mp4 파일 경로를 반환합니다.
    """
    # 동영상 저장 시 충돌 방지를 위해 임시 파일명 생성
    output_filename = f"download_{uuid.uuid4().hex}.mp4"

    # yt-dlp 명령어 구성
    command = [
        "yt-dlp",
        "-f", "mp4",          # mp4 포맷
        "-o", output_filename, # 저장 파일명
        video_url
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] yt-dlp 실행 중 오류가 발생했습니다: {e}")
        raise

    return output_filename


###############################################################################
# 2. 키프레임 이미지 및 오디오 추출 (ffmpeg)
###############################################################################

def extract_keyframes_and_audio(video_path: str,
                               keyframe_interval: float = 5.0) -> Tuple[List[Tuple[float, str]], str]:
    """
    ffmpeg를 이용해 특정 간격(keyframe_interval)마다 키프레임 이미지를 추출하고,
    전체 오디오를 별도로 추출합니다.
    - keyframe_interval(초)마다 이미지 추출
    - 이미지 파일명은 timestamp를 포함하도록 설정
    - 리턴값 keyframes: [(timestamp, image_path), ...]
    - audio_path: 추출된 오디오 파일(.wav) 경로
    """

    # 1) 오디오 추출
    audio_path_pre = f"{uuid.uuid4().hex}"
    audio_path = audio_path_pre + ".wav"
    mp3_path = audio_path_pre + ".mp3"
    audio_command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",             # 비디오 무시
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        audio_path,
        "-y"               # 기존 파일 덮어쓰기
    ]
    try:
        subprocess.run(audio_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg 오디오 추출 중 오류가 발생했습니다: {e}")
        raise
    
    AudioSegment.from_wav(audio_path).export(mp3_path, format="mp3")

    # 2) 키프레임 이미지 추출
    #    ffmpeg에서는 "그냥 특정 초 간격으로 캡처"를 할 수 있으므로,
    #    -vf fps=1/(keyframe_interval) 등을 사용할 수 있음
    #    다만, 실제 "키프레임"과는 다를 수 있지만 개념적으로 n초마다 추출
    images_folder = f"frames_{uuid.uuid4().hex}"
    os.makedirs(images_folder, exist_ok=True)

    frame_extract_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{keyframe_interval}",
        f"{images_folder}/frame_%06d.jpg",
        "-hide_banner",
        "-loglevel", "error",
        "-y"
    ]
    try:
        subprocess.run(frame_extract_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg 이미지 추출 중 오류가 발생했습니다: {e}")
        raise

    # 추출된 이미지 파일들에 대해 타임스탬프를 직접 계산해 리스트로 저장
    # frame_%06d.jpg -> index 순서대로 timestamp 계산
    keyframes = []
    # frame_000001.jpg, frame_000002.jpg, ...
    # => index에 따라 timestamp = (index-1) * keyframe_interval
    # (index-1) 이유는 첫번째 프레임이 index=1이므로 0초
    from glob import glob
    frame_paths = sorted(glob(os.path.join(images_folder, "frame_*.jpg")))

    for i, img_path in enumerate(frame_paths, start=1):
        timestamp = (i - 1) * keyframe_interval
        keyframes.append((timestamp, img_path))

    return keyframes, mp3_path


###############################################################################
# 3. 음성 → 텍스트 변환 (OpenAI Whisper API)
###############################################################################

def split_audio(input_file, output_dir="audio_chunks", chunk_length_min=20, output_format="mp3"):
    """
    오디오 파일을 지정된 길이(분 단위)의 청크로 자릅니다.

    Args:
        input_file (str): 입력 오디오 파일 경로.
        output_dir (str): 출력 청크를 저장할 디렉토리 경로.
        chunk_length_min (int): 각 청크의 원하는 길이 (분 단위).
        output_format (str): 출력 파일 형식 (예: "mp3", "wav", "ogg").
    """
    # 1. 입력 파일 확인
    if not os.path.exists(input_file):
        print(f"오류: 입력 파일 '{input_file}'을(를) 찾을 수 없습니다.")
        return
    if not os.path.isfile(input_file):
        print(f"오류: '{input_file}'은(는) 유효한 파일이 아닙니다.")
        return

    # 2. 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리: '{output_dir}'")

    # 3. 청크 길이 계산 (밀리초 단위)
    chunk_length_ms = chunk_length_min * 60 * 1000
    print(f"청크 길이: {chunk_length_min}분 ({chunk_length_ms}ms)")

    # 4. 오디오 파일 로드
    print(f"오디오 파일 로드 중: '{input_file}'")
    try:
        audio = AudioSegment.from_file(input_file)
        print(f"로드 완료. 총 길이: {len(audio) / 1000:.2f}초")
    except Exception as e:
        print(f"오디오 파일 로드 오류: {e}")
        print("ffmpeg가 설치되어 있고 PATH에 등록되었는지 확인하세요.")
        return

    # 5. 오디오 자르기 및 저장
    start_time = 0
    audio_len = len(audio)
    chunk_index = 1

    while start_time < audio_len:
        end_time = start_time + chunk_length_ms
        # pydub 슬라이싱은 끝을 넘어가도 자동으로 오디오 끝까지만 자릅니다.
        chunk = audio[start_time:end_time]

        # 출력 파일 이름 생성 (원본 파일명 + 청크 번호)
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = os.path.join(output_dir, f"{base_filename}_chunk_{chunk_index}.{output_format}")

        print(f"  청크 {chunk_index} 저장 중: ({start_time/1000:.2f}초 ~ {min(end_time, audio_len)/1000:.2f}초) -> '{output_filename}'")
        try:
            # 지정된 포맷으로 청크 내보내기
            chunk.export(output_filename, format=output_format)
        except Exception as e:
            print(f"  오류: 청크 {chunk_index} 저장 실패 - {e}")
            # 오류 발생 시 계속 진행할지 중단할지 결정할 수 있습니다.
            # return

        # 다음 청크 시작 시간 및 인덱스 업데이트
        start_time += chunk_length_ms
        chunk_index += 1

    print(f"\n총 {chunk_index - 1}개의 청크 파일이 '{output_dir}'에 저장되었습니다.")


async def transcribe_audio(audio_path: str) -> List[Dict[str, float]]:
    """
    OpenAI Whisper API 등을 통해 오디오를 텍스트로 변환합니다.
    - 리턴값: [{"start": float, "end": float, "text": str}, ... ]
    """
    audio_file = open(audio_path, "rb")

    whisper_transcription = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="vtt"
    )
    
    print(type(whisper_transcription))

    transcription = openai_client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
        response_format="text"
    )

    print("###")
    print(type(transcription))

    prompt = f"""
    아래는 동일한 오디오 파일에 대한 두 가지 다른 전사 결과입니다.

첫 번째 결과 (Whisper)는 타임스탬프를 포함하지만 텍스트 정확도가 약간 떨어집니다:
--- [Whisper 결과] ---
{whisper_transcription}
---

두 번째 결과 (Transcribe)는 텍스트 정확도가 높지만 타임스탬프가 없습니다:
--- [Transcribe 결과] ---
{transcription}
---

목표: 두 번째 결과(Transcribe)의 정확한 텍스트를 사용하되, 첫 번째 결과(Whisper)의 타임스탬프를 최대한 정확하게 적용하여 WEBVTT 형식으로 다시 작성해주세요. Whisper의 각 타임스탬프 구간에 해당하는 Transcribe 텍스트를 찾아 매칭하고, 그 텍스트에 해당 타임스탬프를 부여하면 됩니다. 텍스트 내용이 약간 다르므로 의미적으로 가장 유사한 부분을 기준으로 맞춰주세요.
    """
    transcription = await get_t2t(prompt)

    return transcription


###############################################################################
# 4. 이미지 캡셔닝 (Vision-Language 모델, 텍스트 동기화)
###############################################################################
import asyncio
from concurrent.futures import ThreadPoolExecutor


def sync_caption(img_path: str, prompt: str) -> str:
    """
    동기적으로 get_it2t를 호출하는 래퍼 함수.
    """
    return get_it2t(img_path, prompt)

async def caption_frame(semaphore: asyncio.Semaphore, keyframe: tuple, transcription_all: str) -> str:
    """
    하나의 키프레임에 대해 caption을 생성하고 반환하는 비동기 함수.
    """
    timestamp, img_path = keyframe
    prompt = f"""
다음 이미지는 어떤 영상의 {timestamp}초 부근의 한 프레임입니다.
영상의 시간대별 음성 transcription은 다음과 같습니다:
{transcription_all}

위 내용들을 바탕으로 해당 프레임의 이미지가 무슨 내용을 담고 있고 어떤 맥락에서 어떤 내용이 진행되고 있는지 상세히 설명하세요. 이미지가 다이어그램, 도식, 표 등인 경우 그에 대한 상세한 설명을 하고, 어떤 장면이나 일반 카메라로 찍은 이미지인 경우 장면을 상세히 묘사하세요. 한국어로 작성하세요."""
    # 동시 실행 제한을 위해 세마포어로 감싸줍니다.
    async with semaphore:
        loop = asyncio.get_event_loop()
        # I/O 바운드인 get_it2t 호출을 쓰레드풀에서 실행
        caption = await loop.run_in_executor(None, sync_caption, img_path, prompt)
        print(img_path, caption)
        return caption

async def caption_keyframes(
    keyframes: List[Tuple[float, str]],
    transcript: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Vision-Language 모델을 사용해 각 키프레임에 대해 캡션을 생성합니다.
    - keyframes: [(timestamp, image_path), ...]
    - transcript: [{"start": ..., "end": ..., "text": ...}, ...]
    - 리턴: [
        {
            "timestamp": float,
            "caption": str,
            "image_path": str,
            "matched_script": str
        }, ...
      ]
    """
    transcription_all = "\n".join(str(transcript))
    captions = []


    semaphore = asyncio.Semaphore(32)

    tasks = [
        caption_frame(semaphore, keyframe, transcription_all)
        for keyframe in keyframes
    ]
    # 모든 태스크 완료를 기다립니다.
    captions = await asyncio.gather(*tasks)
    return captions


###############################################################################
# 전체 파이프라인 함수
###############################################################################
@mcp.tool()
async def rini_summarize_youtube_all(video_url: str):
    """
    유튜브 영상의 키프레임 이미지를 추출해서 이미지에 대한 캡션을 생성하고, 음성은 STT를 통해 텍스트로 변환합니다. 변환된 텍스트와 캡션 pair가 리턴됩니다.
    """
    os.system("rm -rf download_*.mp4")
    os.system("rm -rf *.mp3 *.wav")
    os.system("rm -rf frames*")
    print("\n[1/4] 유튜브 영상 다운로드 중...")
    video_path = download_video(video_url)

    print("[2/4] 키프레임 및 오디오 추출 중...")
    keyframes, audio_path = extract_keyframes_and_audio(video_path)

    print("[3/4] 오디오 → 텍스트 변환(Transcription) 중...")
    os.system("rm -rf audio_chunks")
    split_audio(audio_path)
    transcript = ""
    for audio_path in sorted(list(os.listdir("audio_chunks"))):
        transcript += await transcribe_audio("audio_chunks/" + audio_path)

    print("[4/4] 이미지 캡셔닝 및 텍스트 동기화 중...")
    captions = await caption_keyframes(keyframes, transcript)

    return transcript, captions

if __name__ == "__main__":
    mcp.run("sse")