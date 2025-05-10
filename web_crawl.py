from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
import trafilatura
import uuid
import asyncio
from util_ai import get_it2t_path

mcp = FastMCP("Rini Web Crawl")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = 65005

@mcp.tool()
async def rini_get_text_only_from_url(url: str):
    """
    주어진 url의 웹사이트로부터 중요한 텍스트만 가져옵니다.
    """
    def _fetch_and_extract_sync(url_to_fetch):
        fetched_content = trafilatura.fetch_url(url_to_fetch)
        if fetched_content is None:
            return None
        return trafilatura.extract(fetched_content)

    text_content = await asyncio.to_thread(_fetch_and_extract_sync, url)
    return text_content

@mcp.tool()
async def rini_get_all_from_url(url: str, timeout: int = 5):
    """
    주어진 웹사이트가 로딩 되기까지 timeout만큼 기다린 후, 웹사이트의 스크린샷을 찍어서 웹페이지의 배치나 이미지를 분석하고 텍스트를 OCR한 결과를 반환합니다.
    """
    url = url.strip()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(timeout)

        generated_filename = f"{uuid.uuid4()}.png"
        await page.screenshot(path=generated_filename, full_page=True)
        prompt = """
다음 웹사이트 스크린샷에 대한 캡션을 LLM이 이해하기 쉽게 작성하세요.
사이트의 전체적인 배치나 내용, 이미지의 내용, 텍스트를 OCR한 결과를 포함해야 합니다.
당신이 작성한 캡션만 보고도 어떤 사이트인지, 무슨 내용이 있는지 쉽게 알 수 있어야 합니다.
"""
        ret = await get_it2t_path(generated_filename, prompt)
        return ret
        
if __name__ == "__main__":
    mcp.run("sse")
