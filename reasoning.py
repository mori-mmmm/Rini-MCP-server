from mcp.server.fastmcp import FastMCP
from util_ai import get_m2t

mcp = FastMCP("Rini Reasoning")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = 65003

@mcp.tool()
async def rini_reasoning(query: str, model: str = None):
    """
    주어진 쿼리에 대해 reasoning이 가능한 모델이 깊이 생각을 해서 답변을 합니다. 수학, 과학, 코딩 등 논리적인 추론이 필요한 작업에 사용하면 좋습니다. model 인자의 경우 유저가 명시하지 않았으면 비워두세요.
    """
    prompt = f"""다음 질문에 대해 신중히 고민한 후 답하세요. 문제를 어떻게 풀지 계획을 세우세요. 문제를 더 풀기 쉬운 작은 단계들로 쪼개고 각 단계를 어떤 식으로 해결할지 생각하세요. 그 다음 계획대로 실제로 문제를 해결하세요. 해결한 후에는 과정에 오류가 없었는지 검증하고 확인하세요.
문제: {query}
"""
    messages = [
        {"role":"user", "content": prompt}
    ]

    #ret = await get_m2t(messages, thinking_budget=4096, model="gemini-2.5-pro-preview-05-06")
    ret = await get_m2t(messages, model="o4-mini")
    return ret

if __name__ == "__main__":
    mcp.run("sse")