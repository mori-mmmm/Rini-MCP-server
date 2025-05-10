import asyncio
import subprocess
import sys
from mcp.server.fastmcp import FastMCP
from util_ai import get_m2t

mcp = FastMCP("Rini Coding")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = 65006

@mcp.tool()
async def rini_create_and_execute_mcp_server(query: str, filename: str, port: int):
    """
    주어진 쿼리에 대해 코드를 생성합니다. 특정 프로그래밍 언어나 프레임워크를 명시하면 더 정확한 코드를 얻을 수 있습니다. model 인자의 경우 유저가 명시하지 않았으면 비워두세요.
    """
    prompt = f"""당신은 코드 생성 전문가입니다. 다음 요청에 따라 코드를 작성해주세요.
요청: {query}

코드만 응답해주세요. 다른 설명은 필요 없습니다.
코드 블럭에 넣지 말고 코드를 바로 생성해주세요.
"""
    messages = [
        {"role":"user", "content": prompt}
    ]
    #ret = await get_m2t(messages, thinking_budget=4096, model="gemini-2.5-pro-preview-05-06")
    ret = await get_m2t(messages, model="o4-mini")


    prompt = f"""다음 코드를 async하게 실행하는데 문제가 없도록 동기(sync) 방식으로 구현되어 있거나 병목이 될만한 부분을 비동기 방식으로 바꾸세요:
{ret}

코드만 응답해주세요. 다른 설명은 필요 없습니다.
다음과 같은 포맷으로 출력해주세요.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("YOUR_MCP_NAME")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = {port}

@mcp.tool()
async def YOUR_FUNCTION_NAME(args):
    \"\"\"
    함수에 대한 설명을 주석으로 넣습니다.
    \"\"\"
    # TODO

if __name__ == "__main__":
    mcp.run("sse")
"""
    messages = [
        {"role":"user", "content": prompt}
    ]
    #ret = await get_m2t(messages, thinking_budget=4096, model="gemini-2.5-pro-preview-05-06")
    code = await get_m2t(messages, model="o4-mini")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(code)
    
    process = await asyncio.create_subprocess_exec(
        sys.executable, filename,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print(f"Successfully executed {filename}")
        if stdout:
            print(f"stdout:\n{stdout.decode()}")
    else:
        print(f"Error executing {filename}")
        if stderr:
            print(f"stderr:\n{stderr.decode()}")



if __name__ == "__main__":
    mcp.run("sse")
