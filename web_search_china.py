import json
import re
import sys
import io
import os
from urllib.parse import quote  # 新增：用于 URL 编码中文
from bs4 import BeautifulSoup

# Windows 控制台 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class WebSearchChina:
    def __init__(self, headless=True):
        self.headless = headless
        # 优先使用环境变量 PLAYWRIGHT_BROWSERS_PATH 下的 chromium
        self.browser_executable = None
        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        if browsers_path:
            # 在目录中搜索 chrome.exe
            for root, dirs, files in os.walk(browsers_path):
                if "chrome.exe" in files:
                    self.browser_executable = os.path.join(root, "chrome.exe")
                    break
        # 备选硬编码（仅当环境变量失效时使用）
        # self.browser_executable = r"D:\minicodaAPP\envs\ai_agent\playwright-browsers\chromium-1223\chrome-win64\chrome.exe"

        # 延迟导入 playwright，避免未安装时崩溃
        from playwright.sync_api import sync_playwright
        self.sync_playwright = sync_playwright

    def search(self, query, engine="baidu", count=5):
        if engine not in ("baidu", "bing", "360"):
            engine = "bing"   # 默认改为 bing，更稳定
        try:
            with self.sync_playwright() as p:
                launch_args = {"headless": self.headless}
                if self.browser_executable:
                    launch_args["executable_path"] = self.browser_executable
                browser = p.chromium.launch(**launch_args)
                page = browser.new_page()
                try:
                    if engine == "baidu":
                        results = self._search_baidu(page, query, count)
                    elif engine == "bing":
                        results = self._search_bing(page, query, count)
                    elif engine == "360":
                        results = self._search_360(page, query, count)
                    else:
                        results = []
                finally:
                    browser.close()
            return self._format_results(results, count)
        except Exception as e:
            return [{"title": "搜索错误", "url": "", "snippet": f"搜索失败: {str(e)}"}]

    def _search_baidu(self, page, query, count):
        # 对查询词进行 URL 编码
        encoded_query = quote(query)
        page.goto(f"https://www.baidu.com/s?wd={encoded_query}&rn={count}", wait_until="networkidle")
        content = page.content()
        soup = BeautifulSoup(content, "html.parser")
        results = []
        # 百度新版选择器
        for div in soup.select("div.result, div.c-container, div.result-op, div.c-result"):
            title_elem = div.select_one("h3 a, .t a, .c-title a")
            snippet_elem = div.select_one(".c-abstract, .content-right_8Zs40, .c-span-last, .c-summary, .c-line-clamp2")
            if title_elem:
                title = title_elem.get_text(strip=True)
                url = title_elem.get("href") or ""
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                if len(title) >= 4 and url:
                    results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= count:
                    break
        # 如果没结果，尝试匹配移动版链接
        if not results:
            pattern = r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]{5,})</a>'
            for href, title in re.findall(pattern, content):
                if "baidu.com" not in href:
                    results.append({"title": title.strip(), "url": href, "snippet": ""})
                    if len(results) >= count:
                        break
        return results

    def _search_bing(self, page, query, count):
        # 对查询词进行 URL 编码
        encoded_query = quote(query)
        page.goto(f"https://cn.bing.com/search?q={encoded_query}&count={count}", wait_until="networkidle")
        content = page.content()
        soup = BeautifulSoup(content, "html.parser")
        results = []
        for item in soup.select(".b_algo, .sa_cc, .b_result"):
            title_elem = item.select_one("h2 a, .b_title a")
            snippet_elem = item.select_one(".b_caption p, .b_snippet, .b_lineclamp2")
            if title_elem:
                title = title_elem.get_text(strip=True)
                url = title_elem.get("href") or ""
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                if len(title) >= 4:
                    results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= count:
                    break
        if not results:
            pattern = r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]{5,})</a>'
            for href, title in re.findall(pattern, content):
                if "bing.com" not in href and "microsoft.com" not in href:
                    results.append({"title": title.strip(), "url": href, "snippet": ""})
                    if len(results) >= count:
                        break
        return results

    def _search_360(self, page, query, count):
        # 对查询词进行 URL 编码
        encoded_query = quote(query)
        page.goto(f"https://www.so.com/s?q={encoded_query}&num={count}", wait_until="networkidle")
        content = page.content()
        soup = BeautifulSoup(content, "html.parser")
        results = []
        for item in soup.select(".res-list .res-item, .result"):
            title_elem = item.select_one(".res-title a, .t a")
            snippet_elem = item.select_one(".res-desc, .c-abstract")
            if title_elem:
                title = title_elem.get_text(strip=True)
                url = title_elem.get("href") or ""
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                if len(title) >= 4:
                    results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= count:
                    break
        if not results:
            pattern = r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]{5,})</a>'
            for href, title in re.findall(pattern, content):
                if "so.com" not in href:
                    results.append({"title": title.strip(), "url": href, "snippet": ""})
                    if len(results) >= count:
                        break
        return results

    def _format_results(self, results, count):
        formatted = []
        for i, res in enumerate(results[:count]):
            formatted.append({
                "rank": i + 1,
                "title": res.get("title", ""),
                "url": res.get("url", ""),
                "snippet": res.get("snippet", "")
            })
        return formatted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="搜索关键词")
    parser.add_argument("--engine", default="bing", choices=["baidu", "bing", "360"])  # 命令行默认也改 bing
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    searcher = WebSearchChina()
    results = searcher.search(args.query, engine=args.engine, count=args.count)
    print(json.dumps(results, ensure_ascii=False))