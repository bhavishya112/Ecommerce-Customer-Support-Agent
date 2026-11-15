from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        user_data_dir="./browser_profile",
        headless=False,
        channel="chrome",  # real Chrome, if installed
        ignore_default_args=["--enable-automation"],
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://www.amazon.com")

    input("Log in manually, then press Enter...")

    context.storage_state(path="auth.json")  # save BEFORE closing
    context.close()
