import os
import asyncio
import numpy as np
from util_ai import get_embedding, get_t2t
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Rini Github Repository Analysis")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = 65002

ext2lang = {
    'sh': 'bash',
    'c': 'c',
    'cpp': 'cpp',
    'cs': 'csharp',
    'css': 'css',
    'go': 'go',
    'hs': 'haskell',
    'html': 'html',
    'java': 'java',
    'js': 'javascript',
    'jsdoc': 'javascript',
    'json': 'json',
    'odin': 'odin',
    'php': 'php',
    'py': 'python',
    'rb': 'ruby',
    'rs': 'rust',
    'ts': 'typescript',
    're': 'regex',
}

def _read_file_sync(file_path: str):
    """Synchronously reads a file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def extract_functions_and_classes(source_code, language):
    lexer = get_lexer_by_name(language)
    tokens = lex(source_code, lexer)
    
    functions = []
    classes = []
    current_entity = None
    entity_type = None
    entity_code = []


    
    for token_type, token_value in tokens:
        if token_type in Token.Name.Function:
            if current_entity:
                if entity_type == 'function':
                    functions.append(''.join(entity_code))
                elif entity_type == 'class':
                    classes.append(''.join(entity_code))
            current_entity = token_value
            entity_type = 'function'
            entity_code = [token_value]
        elif token_type in Token.Name.Class:
            if current_entity:
                if entity_type == 'function':
                    functions.append(''.join(entity_code))
                elif entity_type == 'class':
                    classes.append(''.join(entity_code))
            current_entity = token_value
            entity_type = 'class'
            entity_code = [token_value]
        elif token_type in Token.Text and current_entity:
            entity_code.append(token_value)
        else:
            if current_entity:
                entity_code.append(token_value)
    
    if current_entity:
        if entity_type == 'function':
            functions.append(''.join(entity_code))
        elif entity_type == 'class':
            classes.append(''.join(entity_code))
    
    return functions, classes

async def process_repo(folder_path):
    results = {}
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            ext = file_name.split('.')[-1]
            if ext in ext2lang:
                file_path = os.path.join(root, file_name)
                
                source_code = await asyncio.to_thread(_read_file_sync, file_path)
                if source_code is None:
                    continue
                
                functions, classes = await asyncio.to_thread(
                    extract_functions_and_classes, source_code, ext2lang[ext]
                )
                results[file_path] = {
                    'functions': functions,
                    'classes': classes
                }
    results = await transform_results(results)
    return results

async def transform_results(results):
    embeddings_list = []

    for file_path, content in results.items():
        # 함수 처리
        for func in content['functions']:
            emb = await get_embedding(func)
            embeddings_list.append((emb, f"{file_path}", f"{func}"))
        
        # 클래스 처리
        for cls in content['classes']:
            emb = await get_embedding(cls)
            embeddings_list.append((emb, f"{file_path}", f"{cls}"))

    return embeddings_list

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (magnitude_vec1 * magnitude_vec2)
    return cosine_sim

@mcp.tool()
async def rini_github_analysis(query: str, url: str):
    """"
    주어진 쿼리와 관련 있는 내용의 코드를 해당 url의 github repository에서 찾아온 후 쿼리에 답변합니다. 
    """
    repo_name = url.split('/')[-1]
    print('repo name:', repo_name)
    repo_path = f'github/{repo_name}'

    if not await asyncio.to_thread(os.path.exists, repo_path):
        await asyncio.to_thread(os.makedirs, repo_path, exist_ok=True)
        git_clone_cmd = f"git clone {url}.git {repo_path}"
        process = await asyncio.create_subprocess_shell(
            git_clone_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            print(f"Error cloning repository: {stderr.decode(errors='ignore')}")
            return f"Error cloning repository: {stderr.decode(errors='ignore')}"
            
    emb_list = await process_repo(repo_path)
    query_embedding = await get_embedding(query)
    sim_list = []
    for emb, path, code in emb_list:
        sim = cosine_similarity(query_embedding, emb)
        sim_list.append((sim, emb, path, code))
    sim_list = sorted(sim_list, key=lambda x:x[0])[::-1]

    prompt = f"아래의 코드를 참고해서 다음 요청사항에 대답해줘\n요청사항:{query}\n코드:\n"
    for i in range(min(10, len(sim_list))):
        sim, emb, path, code = sim_list[i]
        print(i, sim, path, code)
        prompt += f"[{path}]\n{code}\n\n"
    ret = await get_t2t(prompt)
    return ret

if __name__ == "__main__":
    mcp.run("sse")
