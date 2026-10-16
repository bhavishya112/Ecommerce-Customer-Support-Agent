from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        user_data_dir="./browser_profile",
        headless=False,
        viewport={"width": 375, "height": 667},
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        is_mobile=True,
        has_touch=True
    )

    page = context.pages[0] if context.pages else context.new_page()

    page.goto("https://www.amazon.in")

    input("Log in manually if needed, then press Enter...")

    context.close()
