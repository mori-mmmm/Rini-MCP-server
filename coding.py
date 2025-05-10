import subprocess
import sys
from mcp.server.fastmcp import FastMCP
from util_ai import get_m2t

mcp = FastMCP("Rini Coding")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = 65004

@mcp.tool()
async def rini_code_generation(query: str, model: str = None):
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
    return ret

@mcp.tool()
async def rini_python_code_execution(code: str):
    """
    주어진 Python 코드를 실행하고 stdout에 출력된 것, stderr에 출력된 것, return code를 str로 만든 것을 반환합니다. return value가 별도로 제공되지 않으므로 필요한 경우 함수 마지막에 꼭 반환하려는 값을 print 하세요.
    """
    stdout_result = ""
    stderr_result = ""

    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout_result, stderr_result = process.communicate(timeout=200)
        return_code = process.returncode

    
    except subprocess.TimeoutExpired:
        stderr_result = "Code execution timed out."
        if process:
            process.kill()
            stdout_result, stderr_result = process.communicate()
    except Exception as e:
        stderr_result = f"An error occurred during code execution: {str(e)}"

    return stdout_result, stderr_result, return_code

if __name__ == "__main__":
    mcp.run("sse")
