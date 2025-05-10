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
from util_ai import get_it2t_path


mcp = FastMCP("Rini Youtube Summarize")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = 65001

async def get_youtube_url(song_title):
    ydl_opts = {
        'default_search': 'ytsearch',
        'quiet': True
    }
    def _sync_get_url():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(song_title, download=False)
            if result and 'entries' in result and result['entries']:
                return result['entries'][0]['webpage_url']
            return None
    loop = asyncio.get_event_loop()
    video_url = await loop.run_in_executor(None, _sync_get_url)
    return video_url

async def download_audio_from_youtube(output_file, folder=True, video_url=None):
    if folder:
        output_file = 'music/' + output_file
    
    if folder:
        target_dir = os.path.dirname(output_file)
        if target_dir and not await asyncio.to_thread(os.path.exists, target_dir):
            await asyncio.to_thread(os.makedirs, target_dir)

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

    def _sync_download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_download)

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

async def img2base64(image_path: str) -> Optional[str]:
    def _sync_img2base64():
        if not os.path.exists(image_path):
            print(f"Error: File not found at '{image_path}'")
            return None
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            base64_encoded_bytes = base64.b64encode(image_data)
            return base64_encoded_bytes.decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    return await asyncio.to_thread(_sync_img2base64)

###############################################################################
# 1. 유튜브 영상 다운로드 (yt-dlp)
###############################################################################

async def download_video(video_url: str) -> str:
    """
    yt-dlp를 사용해 유튜브 영상을 다운로드하고, 로컬 mp4 파일 경로를 반환합니다.
    """
    output_filename = f"download_{uuid.uuid4().hex}.mp4"
    command = [
        "yt-dlp",
        "-f", "mp4",
        "-o", output_filename,
        video_url
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_message = stderr.decode(errors='ignore').strip()
        print(f"[ERROR] yt-dlp 실행 중 오류가 발생했습니다: {error_message}")
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    
    return output_filename


###############################################################################
# 2. 키프레임 이미지 및 오디오 추출 (ffmpeg)
###############################################################################

async def extract_keyframes_and_audio(video_path: str,
                               keyframe_interval: float = 5.0) -> Tuple[List[Tuple[float, str]], str]:
    """
    ffmpeg를 이용해 특정 간격(keyframe_interval)마다 키프레임 이미지를 추출하고,
    전체 오디오를 별도로 추출합니다.
    """
    audio_path_pre = f"{uuid.uuid4().hex}"
    wav_audio_path = audio_path_pre + ".wav"
    mp3_audio_path = audio_path_pre + ".mp3"

    # 1) 오디오 추출 (ffmpeg to wav)
    audio_command = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "2", wav_audio_path, "-y"
    ]
    process_audio = await asyncio.create_subprocess_exec(
        *audio_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr_audio = await process_audio.communicate()
    if process_audio.returncode != 0:
        print(f"[ERROR] ffmpeg 오디오 추출 중 오류: {stderr_audio.decode(errors='ignore')}")
        raise subprocess.CalledProcessError(process_audio.returncode, audio_command, stderr=stderr_audio)

    def _convert_wav_to_mp3_sync():
        audio_segment = AudioSegment.from_wav(wav_audio_path)
        audio_segment.export(mp3_audio_path, format="mp3")
        if os.path.exists(wav_audio_path):
             os.remove(wav_audio_path)
    await asyncio.to_thread(_convert_wav_to_mp3_sync)


    # 2) 키프레임 이미지 추출
    images_folder = f"frames_{uuid.uuid4().hex}"
    await asyncio.to_thread(os.makedirs, images_folder, exist_ok=True)

    frame_extract_cmd = [
        "ffmpeg", "-i", video_path, "-vf", f"fps=1/{keyframe_interval}",
        f"{images_folder}/frame_%06d.jpg", "-hide_banner", "-loglevel", "error", "-y"
    ]
    process_frames = await asyncio.create_subprocess_exec(
        *frame_extract_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr_frames = await process_frames.communicate()
    if process_frames.returncode != 0:
        print(f"[ERROR] ffmpeg 이미지 추출 중 오류: {stderr_frames.decode(errors='ignore')}")
        raise subprocess.CalledProcessError(process_frames.returncode, frame_extract_cmd, stderr=stderr_frames)

    from glob import glob
    
    def _get_frame_paths_sync():
        return sorted(glob(os.path.join(images_folder, "frame_*.jpg")))
    
    frame_paths = await asyncio.to_thread(_get_frame_paths_sync)
    
    keyframes = []
    for i, img_path in enumerate(frame_paths, start=1):
        timestamp = (i - 1) * keyframe_interval
        keyframes.append((timestamp, img_path))

    return keyframes, mp3_audio_path


###############################################################################
# 3. 음성 → 텍스트 변환 (OpenAI Whisper API)
###############################################################################

async def split_audio(input_file, output_dir="audio_chunks", chunk_length_min=20, output_format="mp3"):
    """
    오디오 파일을 지정된 길이(분 단위)의 청크로 자릅니다.
    """
    if not await asyncio.to_thread(os.path.exists, input_file):
        print(f"오류: 입력 파일 '{input_file}'을(를) 찾을 수 없습니다.")
        return
    if not await asyncio.to_thread(os.path.isfile, input_file):
        print(f"오류: '{input_file}'은(는) 유효한 파일이 아닙니다.")
        return

    await asyncio.to_thread(os.makedirs, output_dir, exist_ok=True)
    print(f"출력 디렉토리: '{output_dir}'")

    chunk_length_ms = chunk_length_min * 60 * 1000
    print(f"청크 길이: {chunk_length_min}분 ({chunk_length_ms}ms)")

    print(f"오디오 파일 로드 중: '{input_file}'")
    
    def _load_audio_sync():
        try:
            return AudioSegment.from_file(input_file)
        except Exception as e:
            print(f"오디오 파일 로드 오류: {e}")
            print("ffmpeg가 설치되어 있고 PATH에 등록되었는지 확인하세요.")
            return None
    
    audio = await asyncio.to_thread(_load_audio_sync)
    if audio is None:
        return

    print(f"로드 완료. 총 길이: {len(audio) / 1000:.2f}초")

    start_time = 0
    audio_len = len(audio)
    chunk_index = 1

    while start_time < audio_len:
        end_time = start_time + chunk_length_ms
        chunk = audio[start_time:end_time]

        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = os.path.join(output_dir, f"{base_filename}_chunk_{chunk_index}.{output_format}")

        print(f"  청크 {chunk_index} 저장 중: ({start_time/1000:.2f}초 ~ {min(end_time, audio_len)/1000:.2f}초) -> '{output_filename}'")
        
        def _export_chunk_sync():
            try:
                chunk.export(output_filename, format=output_format)
                return True
            except Exception as e:
                print(f"  오류: 청크 {chunk_index} 저장 실패 - {e}")
                return False
        
        if not await asyncio.to_thread(_export_chunk_sync):
            pass

        start_time += chunk_length_ms
        chunk_index += 1

    print(f"\n총 {chunk_index - 1}개의 청크 파일이 '{output_dir}'에 저장되었습니다.")


async def transcribe_audio(audio_path: str) -> str:
    """
    OpenAI Whisper API 등을 통해 오디오를 텍스트로 변환합니다.
    """
    def _sync_openai_transcribe(model_name, file_path, response_format_val):
        with open(file_path, "rb") as audio_f:
            return openai_client.audio.transcriptions.create(
                model=model_name,
                file=audio_f,
                response_format=response_format_val
            )

    # First call (whisper-1, vtt)
    whisper_transcription_obj = await asyncio.to_thread(
        _sync_openai_transcribe, "whisper-1", audio_path, "vtt"
    )
    whisper_transcription_text = str(whisper_transcription_obj)

    # Second call (gpt-4o-transcribe, text)
    transcription_obj = await asyncio.to_thread(
        _sync_openai_transcribe, "gpt-4o-transcribe", audio_path, "text"
    )
    transcription_text = str(transcription_obj)

    prompt = f"""
    아래는 동일한 오디오 파일에 대한 두 가지 다른 전사 결과입니다.

첫 번째 결과 (Whisper)는 타임스탬프를 포함하지만 텍스트 정확도가 약간 떨어집니다:
--- [Whisper 결과] ---
{whisper_transcription_text}
---

두 번째 결과 (Transcribe)는 텍스트 정확도가 높지만 타임스탬프가 없습니다:
--- [Transcribe 결과] ---
{transcription_text}
---

목표: 두 번째 결과(Transcribe)의 정확한 텍스트를 사용하되, 첫 번째 결과(Whisper)의 타임스탬프를 최대한 정확하게 적용하여 WEBVTT 형식으로 다시 작성해주세요. Whisper의 각 타임스탬프 구간에 해당하는 Transcribe 텍스트를 찾아 매칭하고, 그 텍스트에 해당 타임스탬프를 부여하면 됩니다. 텍스트 내용이 약간 다르므로 의미적으로 가장 유사한 부분을 기준으로 맞춰주세요.
    """
    
    final_transcription = await get_t2t(prompt) 
    return final_transcription


###############################################################################
# 4. 이미지 캡셔닝 (Vision-Language 모델, 텍스트 동기화)
###############################################################################
import asyncio

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
    async with semaphore:
        caption = await get_it2t_path(img_path, prompt) 
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
    async def _cleanup_path(path_to_clean, is_dir=False):
        try:
            if await asyncio.to_thread(os.path.exists, path_to_clean):
                if is_dir:
                    import shutil
                    await asyncio.to_thread(shutil.rmtree, path_to_clean)
                    print(f"Cleaned directory: {path_to_clean}")
                else:
                    await asyncio.to_thread(os.remove, path_to_clean)
                    print(f"Cleaned file: {path_to_clean}")
        except Exception as e:
            print(f"Error cleaning path {path_to_clean}: {e}")

    await _cleanup_path("frames_*", is_dir=True)
    await _cleanup_path("audio_chunks", is_dir=True)
    

    print("\n[1/4] 유튜브 영상 다운로드 중...")
    video_path = await download_video(video_url)

    print("[2/4] 키프레임 및 오디오 추출 중...")
    keyframes, audio_file_path = await extract_keyframes_and_audio(video_path)

    print("[3/4] 오디오 → 텍스트 변환(Transcription) 중...")
    await _cleanup_path("audio_chunks", is_dir=True) 
    await asyncio.to_thread(os.makedirs, "audio_chunks", exist_ok=True)

    await split_audio(audio_file_path)

    transcription_parts = []
    
    def _list_audio_chunks_sync():
        if not os.path.exists("audio_chunks"):
            return []
        return sorted([os.path.join("audio_chunks", f) for f in os.listdir("audio_chunks") if os.path.isfile(os.path.join("audio_chunks", f))])

    audio_chunk_paths = await asyncio.to_thread(_list_audio_chunks_sync)

    for chunk_path in audio_chunk_paths:
        transcription_parts.append(await transcribe_audio(chunk_path))

    full_transcript_text = "\n".join(transcription_parts)

    print("[4/4] 이미지 캡셔닝 및 텍스트 동기화 중...")
    captions_result = await caption_keyframes(keyframes, [{"text": full_transcript_text}])

    await _cleanup_path(video_path)
    await _cleanup_path(audio_file_path)
    if keyframes:
        frame_parent_dir = os.path.dirname(keyframes[0][1])
        await _cleanup_path(frame_parent_dir, is_dir=True)
    await _cleanup_path("audio_chunks", is_dir=True)


    return full_transcript_text, captions_result

if __name__ == "__main__":
    mcp.run("sse")
