import httpx
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
import trafilatura
import http.client
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
        await self.browser.close()
        await self.playwright.stop()


def get_web_text(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded)

async def get_web_text_stealth(url):
    browser = StealthBrowser(headless=True)  # Running in headless mode
    await browser.initialize()
    page, context = await browser.new_page()

    await page.goto(url)
    await page.wait_for_load_state("networkidle")

    html = await page.content()

    result = trafilatura.extract(html)

    await browser.close()

    return result

async def pprint(text, prefix="[WebSearchAgent] "):
    text = str(text)
    text_list = text.split("\\n")
    print("\\n".join([prefix + x for x in text_list] ))

@mcp.tool()
async def rini_google_search_base(query: str, num: int = 20, start: int = 0, lang: str = 'ko', since: str = None, until: str = None, source: str = None):
    """
    구글 검색을 한 뒤 검색 결과를 가져오는 함수. 외부 정보나 실시간 데이터가 필요할 때 사용하는 함수. 구글 검색 결과 1페이지에 보이는 미리보기 정보만 가져오고 속도가 상대적으로 빠름.
    """
    ban_list = [
            'accounts.google.com',
            'support.google.com',
            'maps.google.com',
            'policy.naver.com',
            'navercorp.com',
            'adcr.naver.com',
            'support.microsoft',
            'help.naver.com',
            'keep.naver.com',
            'policies.google.com',
            'www.google.com/preferences',
            'www.google.com',
            'facebook.com',
            'youtube.com',
        ]

    conn = http.client.HTTPSConnection("google.serper.dev")

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
    # pprint(f"params: {payload}")
    conn.request("POST", "/search", json.dumps(payload), headers)
    res = conn.getresponse()
    data = res.read()
    ret = data.decode("utf-8")
    results = json.loads(ret)

    #print(results)

    if 'organic' not in results:
        payload['num'] = 50
        conn.request("POST", "/search", json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read()
        ret = data.decode("utf-8")
        results = json.loads(ret)

    if 'organic' not in results:
        print("검색 결과가 없습니다.")
        return "검색 결과가 없습니다."
        

    #pprint(f"results: {results}")

    links = []
    result = []
    for x in results['organic']:
        link = x['link']
        # Skip links that are in the ban_list
        if any(banned_site in link for banned_site in ban_list):
            continue
        links.append(link)
        result.append(str(x))

    if len(result) == 0:
        ret = "검색결과가 없습니다. 검색 기간이 너무 짧지는 않은지, 인자의 종류와 포맷은 잘 지켰는지 확인하세요."
        print(ret)
        return ret
        

    ret = '\\n\\n'.join(result)
    #pprint(f"search result:{ret}")
    print(ret)
    return ret

@mcp.tool()
async def rini_google_search_link_only(query: str, num: int = 20, start: int = 0, lang: str = 'ko', since: str = None, until: str = None, source: str = None):
    """
    구글 검색을 한 뒤 검색 결과의 링크들만 가져오는 함수. 외부 정보나 실시간 데이터가 필요할 때 사용하는 함수.
    """
    ban_list = [
            'accounts.google.com',
            'support.google.com',
            'maps.google.com',
            'policy.naver.com',
            'navercorp.com',
            'adcr.naver.com',
            'support.microsoft',
            'help.naver.com',
            'keep.naver.com',
            'policies.google.com',
            'www.google.com/preferences',
            'www.google.com',
            'facebook.com',
            'youtube.com',
        ]

    conn = http.client.HTTPSConnection("google.serper.dev")

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
    # pprint(f"params: {payload}")
    conn.request("POST", "/search", json.dumps(payload), headers)
    res = conn.getresponse()
    data = res.read()
    ret = data.decode("utf-8")
    results = json.loads(ret)

    if 'organic' not in results:
        payload['num'] = 50
        conn.request("POST", "/search", json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read()
        ret = data.decode("utf-8")
        results = json.loads(ret)

    if 'organic' not in results:
        print("검색 결과가 없습니다.")
        return []
        
    links = []
    for x in results['organic']:
        link = x['link']
        # Skip links that are in the ban_list
        if not any(banned_site in link for banned_site in ban_list):
            links.append(link)

    if len(links) == 0:
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
    results = []
    for link in links:
        try:
            downloaded = trafilatura.fetch_url(link)
            result = trafilatura.extract(downloaded)
        except:
            continue
        results.append(result)
    return '\n\n'.join(results)

if __name__ == "__main__":
    mcp.run("sse")