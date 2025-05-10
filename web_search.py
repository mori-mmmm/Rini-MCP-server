import httpx
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
import trafilatura
import http.client
import asyncio
import json
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP("Rini Web Search")
mcp.settings.log_level = "DEBUG"
mcp.settings.port = 65000

class StealthBrowser:
    def __init__(self,
                user_agent: Optional[str] = None,
                timezone: str = "UTC",
                locale: str = "en-US",
                viewport: dict = {"width": 1280, "height": 720},
                headless: bool = False):
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
                                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        self.timezone = timezone
        self.locale = locale
        self.viewport = viewport
        self.headless = headless
        self.playwright = None
        self.browser = None

    async def initialize(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)

    async def _get_context(self):
        return await self.browser.new_context(
            user_agent=self.user_agent,
            locale=self.locale,
            timezone_id=self.timezone,
            viewport=self.viewport,
        )

    async def _apply_stealth(self, page):
        # Stealth patches
        await page.add_init_script("""
        // Remove webdriver
        Object.defineProperty(navigator, 'webdriver', {get: () => false});

        // Fake languages
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});

        // Fake plugins
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});

        // Chrome property
        window.chrome = { runtime: {} };

        // Permissions spoof
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) =>
            parameters.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : originalQuery(parameters);

        // WebGL vendor spoof
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) return 'Intel Inc.';
            if (parameter === 37446) return 'Intel Iris OpenGL Engine';
            return getParameter.call(this, parameter);
        };

        // User-Agent client hints
        Object.defineProperty(navigator, 'userAgentData', {
            get: () => ({
                brands: [{brand: "Google Chrome", version: "120"}],
                mobile: false,
                platform: "Windows"
            })
        });
        """)

    async def new_page(self):
        context = await self._get_context()
        page = await context.new_page()
        await self._apply_stealth(page)
        return page, context

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def get_web_text(url):
    def _sync_get_web_text(url_to_fetch):
        downloaded = trafilatura.fetch_url(url_to_fetch)
        if downloaded is None:
            return None
        return trafilatura.extract(downloaded)
    return await asyncio.to_thread(_sync_get_web_text, url)

async def get_web_text_stealth(url):
    browser = StealthBrowser(headless=True)
    await browser.initialize()
    page, context = await browser.new_page()

    await page.goto(url)
    await page.wait_for_load_state("networkidle")

    html = await page.content()

    result = trafilatura.extract(html)

    await browser.close()

    return result

@mcp.tool()
async def rini_google_search_base(query: str, num: int = 20, start: int = 0, lang: str = 'ko', since: str = None, until: str = None, source: str = None):
    """
    구글 검색을 한 뒤 검색 결과를 가져오는 함수. 외부 정보나 실시간 데이터가 필요할 때 사용하는 함수. 구글 검색 결과 1페이지에 보이는 미리보기 정보만 가져오고 속도가 상대적으로 빠름.
    """
    ban_list = [
        'accounts.google.com', 'support.google.com', 'maps.google.com',
        'policy.naver.com', 'navercorp.com', 'adcr.naver.com',
        'support.microsoft', 'help.naver.com', 'keep.naver.com',
        'policies.google.com', 'www.google.com/preferences', 'www.google.com',
        'facebook.com', 'youtube.com',
    ]

    if since and until:
        tbs = f'cdr:1,cd_min:{since},cd_max:{until}'
    elif since:
        tbs = f'cdr:1,cd_min:{since}'
    elif until:
        tbs = f'cdr:1,cd_max:{until}'
    else:
        tbs = ""
    payload = {
      "q": query + f' site:{source}' if source is not None else query,
      "start": start,
      "num": num,
      "location": "Seoul, Seoul, South Korea",
      "google_domain": "google.co.kr",
      "gl": "kr",
      "hl": lang,
      "tbs": tbs
    }
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=20.0)
            response.raise_for_status() 
            results = response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return "검색 중 오류가 발생했습니다 (HTTP 상태 코드)."
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            return "검색 중 오류가 발생했습니다 (요청 오류)."
        except json.JSONDecodeError:
            print("Error decoding JSON response from search API")
            return "검색 결과를 파싱하는 중 오류가 발생했습니다."


    if 'organic' not in results or not results['organic']:
        # payload['num'] = 50 
        # try:
        #     response = await client.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=20.0)
        #     response.raise_for_status()
        #     results = response.json()
        # except ... (handle errors as above)
        #
        # if 'organic' not in results or not results['organic']:
        print("검색 결과가 없습니다.")
        return "검색 결과가 없습니다."

    links = []
    result = []
    for x in results.get('organic', []):
        link = x.get('link')
        if link and not any(banned_site in link for banned_site in ban_list):
            links.append(link)
            result.append(str(x))

    if not result:
        ret = "검색결과가 없습니다. 검색 기간이 너무 짧지는 않은지, 인자의 종류와 포맷은 잘 지켰는지 확인하세요."
        print(ret)
        return ret
        

    ret = '\\n\\n'.join(result)
    print(ret)
    return ret

@mcp.tool()
async def rini_google_search_link_only(query: str, num: int = 20, start: int = 0, lang: str = 'ko', since: str = None, until: str = None, source: str = None):
    """
    구글 검색을 한 뒤 검색 결과의 링크들만 가져오는 함수. 외부 정보나 실시간 데이터가 필요할 때 사용하는 함수.
    """
    ban_list = [
        'accounts.google.com', 'support.google.com', 'maps.google.com',
        'policy.naver.com', 'navercorp.com', 'adcr.naver.com',
        'support.microsoft', 'help.naver.com', 'keep.naver.com',
        'policies.google.com', 'www.google.com/preferences', 'www.google.com',
        'facebook.com', 'youtube.com',
    ]

    if since and until:
        tbs = f'cdr:1,cd_min:{since},cd_max:{until}'
    elif since:
        tbs = f'cdr:1,cd_min:{since}'
    elif until:
        tbs = f'cdr:1,cd_max:{until}'
    else:
        tbs = ""
    payload = {
      "q": query + (f' site:{source}' if source else ""),
      "start": start,
      "num": num,
      "location": "Seoul, Seoul, South Korea",
      "google_domain": "google.co.kr",
      "gl": "kr",
      "hl": lang,
      "tbs": tbs
    }
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=20.0)
            response.raise_for_status()
            results = response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            return []
        except json.JSONDecodeError:
            print("Error decoding JSON response from search API")
            return []

    if 'organic' not in results or not results['organic']:
        print("검색 결과가 없습니다.")
        return []
        
    links = []
    for x in results.get('organic', []):
        link = x.get('link')
        if link and not any(banned_site in link for banned_site in ban_list):
            links.append(link)

    if not links:
        print("검색결과가 없습니다. 검색 기간이 너무 짧지는 않은지, 인자의 종류와 포맷은 잘 지켰는지 확인하세요.")
        return []
        
    print(f"Found {len(links)} links")
    return links

@mcp.tool()
async def rini_google_search_shallow(query: str):
    """
    구글 검색을 한 뒤 검색 결과로 나온 페이지들을 일일이 방문해서 내용을 가져오는 함수. 외부 정보나 실시간 데이터가 필요할 때 사용하는 함수. 속도가 상대적으로 느림.
    """
    links = await rini_google_search_link_only(query)
    if not links:
        return "관련 링크를 찾지 못했습니다."

    async def fetch_and_extract_content(link):
        try:
            content = await get_web_text(link) 
            return content if content else ""
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            return ""

    tasks = [fetch_and_extract_content(link) for link in links[:15]]
    link_contents = await asyncio.gather(*tasks)
    
    valid_contents = [content for content in link_contents if content] 
    
    if not valid_contents:
        return "링크에서 내용을 추출하지 못했습니다."
        
    return '\n\n'.join(valid_contents)

if __name__ == "__main__":
    mcp.run("sse")
