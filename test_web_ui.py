from playwright.async_api import async_playwright
import asyncio

async def test_streaming_ui():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # 로그인 페이지로 이동
        await page.goto("http://localhost:8000/test/streaming")
        
        # 로그인
        await page.fill('input[name="username"]', 'admin')
        await page.fill('input[name="password"]', 'admin123')
        await page.click('button:has-text("Login")')
        
        # 로그인 성공 대기
        await page.wait_for_selector('h2:has-text("Episode Chat Test")')
        
        # Provider 선택 (OpenAI)
        await page.select_option('select#provider', 'openai')
        
        # Model 입력
        await page.fill('input#model', 'gpt-4o-mini')
        
        # 메시지 입력
        await page.fill('textarea#message', '안녕하세요\! 1부터 3까지 세어주세요.')
        
        # Send 버튼 클릭
        await page.click('button#sendBtn')
        
        # 응답 대기 (10초)
        await asyncio.sleep(10)
        
        # 스크린샷 저장
        await page.screenshot(path='streaming_test_result.png')
        
        print("Test completed. Check streaming_test_result.png for results.")
        
        await browser.close()

asyncio.run(test_streaming_ui())
