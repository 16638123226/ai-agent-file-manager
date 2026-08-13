from playwright.sync_api import sync_playwright

def search_baidu(query):
    """直接通过百度搜索URL搜索，绕过首页"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # 直接访问搜索结果页，绕过百度首页的复杂DOM
        search_url = f"https://www.baidu.com/s?wd={query}"
        page.goto(search_url)
        
        # 等待搜索结果加载
        page.wait_for_selector(".result", timeout=10000)
        
        # 获取第一条搜索结果的标题
        first_result = page.query_selector(".result h3")
        title = first_result.inner_text() if first_result else "没找到结果"
        
        print(f"搜索结果第一条: {title}")
        page.screenshot(path="baidu_result.png")
        print("截图已保存为 baidu_result.png")
        
        browser.close()

if __name__ == "__main__":
    search_baidu("今天天气")