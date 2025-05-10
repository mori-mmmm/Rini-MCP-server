import os
from dotenv import load_dotenv
import asyncio
import mimetypes
from pydub import AudioSegment
import uuid # For generating unique temporary file names
import io # Potentially for in-memory file handling if supported by API, though temp files are safer for whisper

load_dotenv()

def get_tool_prompt(tools):
    prompt = f"""
당신이 사용할 수 있는 tool의 목록과 각각의 용도와 인자들은 다음과 같습니다:
{tools}

tool을 사용하기로 결정한 경우 **코드 블럭에 넣어서** json 포맷으로 바로 생성하세요.
출력 포맷은 다음과 같습니다:

하고 싶은 말이나 설명을 덧붙여도 좋습니다
```json
{{
    tool_calls: [
        {{
            name: (함수 이름),
            args: {{
                (인자 이름1): (인자 값1),
                (인자 이름2): (인자 값2),
                ...
            }}
        }},
        ...
    ]
}}
```
"""
    return prompt



### Google Gemini ###
#import google.generativeai as genai
from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def _prepare_gemini_contents(messages):
    contents = []
    current_content = None
    for message in messages:
        role = message.get("role", "user")
        content_text = message.get("content")
        if not content_text:
            continue

        gemini_role = "user"
        if role == "assistant":
            gemini_role = "model"
        elif role == "system":
            gemini_role = "user" # 시스템 프롬프트는 user 역할로 처리

        text_part = {"text": content_text}

        if current_content and current_content["role"] == gemini_role:
            current_content["parts"].append(text_part)
        else:
            current_content = {
                "role": gemini_role,
                "parts": [text_part]
            }
            contents.append(current_content)
    return contents


async def gai_get_m2t(messages, tools=None, model=None, thinking_budget=None, generation_config_override=None, **kwargs):
    GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
    if not model:
        model = GEMINI_MODEL
    contents = _prepare_gemini_contents(messages)
    gen_config_dict = generation_config_override if generation_config_override else {}

    if thinking_budget is not None and thinking_budget > 0:
        gen_config_dict['thinking_config'] = {"thinking_budget": thinking_budget}
    response = await gemini_client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=gen_config_dict if gen_config_dict else None,
        **kwargs
        # request_options={'timeout': 600} # 필요시 타임아웃 설정
    )
    return response.text

async def gai_get_t2t(text, tools=None, model=None, thinking_budget=None, generation_config_override=None, **kwargs):
    messages = [
        {"role":"user", "content":text}
    ]
    ret = await get_m2t(messages, tools=tools, model=model, thinking_budget=thinking_budget, generation_config_override=generation_config_override, **kwargs)
    return ret

async def gai_get_m2m(messages, tools=None, model=None, thinking_budget=None, generation_config_override=None, **kwargs):
    ret = await get_m2t(messages, tools=tools, model=model, thinking_budget=thinking_budget, generation_config_override=generation_config_override, **kwargs)
    messages.append(
        {"role":"assistant", "content":ret}
    )
    return messages



### OpenAI ###
from openai import OpenAI
import tiktoken

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_embedding(text, model="text-embedding-3-large"):
    return openai_client.embeddings.create(input=[text], model=model).data[0].embedding

async def get_embedding_batch(texts, model="text-embedding-3-large"):
    ret = openai_client.embeddings.create(input=texts, model=model)
    return [x.embedding for x in ret.data]

async def oai_get_m2t(messages, tools=None, model=None):
    OPENAI_MODEL = "gpt-4.1-nano"
    if not model:
        model = OPENAI_MODEL
    print("oai called:", model)
    if tools:
        prompt = get_tool_prompt(tools)
        new_messages = [x for x in messages]
        new_messages += [{"role":"user", "content": prompt}]
        completion = openai_client.chat.completions.create(
            model=model,
            messages=new_messages,
        )
    else:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
    return completion.choices[0].message.content

async def oai_get_t2t(prompt, tools=None, model=None):
    messages = [{"role":"user", "content":prompt}]
    ret = await oai_get_m2t(messages, tools, model)
    return ret

async def oai_get_m2m(messages, tools=None, model=None):
    ret = await get_m2t(messages, tools, model)
    messages.append(
        {"role":"assistant", "content":ret}
    )
    return messages

def num_tokens_from_messages(messages, model="gpt-4o-2024-08-06"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-4o-2024-08-06":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value, disallowed_special=()))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
    Please check the library docs for more info on message-to-token conversion.""")

async def _transcribe_chunk_openai(chunk_audio_segment, chunk_index, original_format="mp3"):
    """Helper function to transcribe a single audio chunk using OpenAI Whisper."""
    temp_file_path = f"audio_chunks/temp_chunk_{uuid.uuid4()}_{chunk_index}.{original_format}"
    
    if not os.path.exists("audio_chunks"):
        try:
            os.makedirs("audio_chunks")
        except OSError as e:
            print(f"Error creating directory audio_chunks: {e}")

    try:
        chunk_audio_segment.export(temp_file_path, format=original_format)
    except Exception as e:
        print(f"Error exporting audio chunk {chunk_index} to {temp_file_path}: {e}")
        return ""

    transcription = ""
    try:
        with open(temp_file_path, 'rb') as audio_file_chunk:
            response = await asyncio.to_thread(
                openai_client.audio.transcriptions.create,
                model="gpt-4o-transcribe",
                file=audio_file_chunk
            )
        transcription = response.text if response else ""
    except Exception as e:
        print(f"Error transcribing chunk {chunk_index} from {temp_file_path}: {e}")
        transcription = ""
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Error deleting temporary file {temp_file_path}: {e}")
    return transcription

async def stt(file_path):
    try:
        file_format = file_path.split('.')[-1].lower()
        if file_format not in ["mp3", "wav", "m4a", "ogg", "flac"]:
            print(f"Unsupported file format {file_format}, defaulting to mp3 for export.")
            file_format = "mp3"

        audio = AudioSegment.from_file(file_path)
        
        ten_minutes_ms = 10 * 60 * 1000  # pydub uses milliseconds
        
        chunks = []
        for i in range(0, len(audio), ten_minutes_ms):
            chunks.append(audio[i:i + ten_minutes_ms])

        if not chunks:
            print("No audio chunks to process.")
            return ''

        transcription_tasks = []
        for i, chunk_segment in enumerate(chunks):
            transcription_tasks.append(_transcribe_chunk_openai(chunk_segment, i, original_format=file_format))
        
        transcriptions_list = await asyncio.gather(*transcription_tasks)
        
        full_transcription = " ".join(filter(None, transcriptions_list))
        return full_transcription

    except FileNotFoundError:
        print(f"Error: Audio file not found at {file_path}")
        return ''
    except Exception as e:
        print(f"An unexpected error occurred in stt function: {e}")
        return ''


### Anthropic ###

#TODO

async def summarize(text, in_msg=None):
    if in_msg is not None:
        ret = await get_t2t(f"""다음 텍스트를 아래 내용을 감안해서 요약해줘:\n{text}\n\n이 내용에 대해 답하는데 필요한 내용 위주로 요약해야 해:\n{in_msg}""")
    else:
        ret = await get_t2t(f"""다음 텍스트를 아래 내용을 감안해서 요약해줘:\n{text}\n\n""")
    return ret

async def pdf_qa(pdf_text, prompt):
    ret = await get_t2t(f"""다음 텍스트 내용을 바탕으로 아래 지시사항을 따르거나 질문에 대답해줘:\n{pdf_text}\n\n요청이나 질문 사항은 다음과 같고, 만약 비어있으면 그냥 텍스트 내용을 요약해줘.:\n{prompt}""", model="gpt-4o-2024-08-06")
    return ret


async def get_t2t(text, tools=None, model=None):
    if model is None:
        ret = await oai_get_t2t(text, tools, model)
    elif model.startswith("gemini"):
        ret = await gai_get_t2t(text, tools, model)
    else:
        ret = await oai_get_t2t(text, tools, model)
    return ret
    
async def get_m2t(messages, model=None, thinking_budget=None, generation_config_override=None, tools=None, **kwargs):
    if model is None:
        ret = await oai_get_m2t(messages, tools, model)
    elif model.startswith("gemini"):
        ret = await gai_get_m2t(messages, tools=tools, model=model, thinking_budget=thinking_budget, generation_config_override=generation_config_override, **kwargs)
    else:
        ret = await oai_get_m2t(messages, tools, model)
    return ret

async def get_m2m(messages, tools=None, model=None):
    if model is None:
        ret = await oai_get_m2m(messages, tools, model)
    elif model.startswith("gemini"):
        ret = await gai_get_m2m(messages, tools, model)
    else:
        ret = await oai_get_m2m(messages, tools, model)
    return ret

async def get_t2i(text):
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=text,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_urls = [response.data[i].url for i in range(1)]
    return image_urls[0]

async def get_it2t_url(prompt, img_url):
    response = openai_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                    },
                },
            ],
        }],
        max_tokens=300,
    )
    return response.choices[0].message.content

def _read_image_bytes_and_detect_mime(image_path: str):
    """이미지 파일 경로에서 바이트 데이터와 MIME 타입을 읽고 감지합니다 (동기)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: '{image_path}'")

    mime_type, _ = mimetypes.guess_type(image_path)
    supported_image_types = ['image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif']

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return image_bytes, mime_type


async def get_it2t_path(image_path: str, prompt: str):
    """이미지와 텍스트 프롬프트를 받아 비동기적으로 Gemini 멀티모달 모델 호출"""
    fixed_model_name = "gemini-2.0-flash" # 모델 이름 업데이트 (또는 설정 가능하게)

    # 이미지 읽기 (동기 방식 유지)
    image_bytes, detected_mime_type = _read_image_bytes_and_detect_mime(image_path)

    image_part = types.Part.from_bytes(mime_type=detected_mime_type, data=image_bytes)
    text_part = types.Part.from_text(text=prompt)

    contents = [types.Content(role="user", parts=[image_part, text_part])]


    response = await gemini_client.aio.models.generate_content(
        model=fixed_model_name,
        contents=contents,
    )

    response_text = ""
    if hasattr(response, 'text') and response.text:
        response_text = response.text
    elif response.candidates:
            first_candidate = response.candidates[0]
            if first_candidate.content and first_candidate.content.parts and first_candidate.content.parts[0].text:
                response_text = first_candidate.content.parts[0].text
                
    return response_text