"""Playwright browser session lifecycle and operator interactive login mode."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Playwright, async_playwright

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("browser.session")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


class BrowserSessionManager:
    """Manages persistent Playwright browser profile and operator authentication setup."""

    def __init__(self, profile_path: Path | str | None = None):
        settings = get_settings()
        self.profile_path = Path(profile_path) if profile_path else settings.resolved_browser_profile_path
        self.profile_path.mkdir(parents=True, exist_ok=True)
        self.settings = settings

    async def launch_persistent_context(
        self,
        headless: bool | None = None,
    ) -> tuple[Playwright, BrowserContext]:
        """Launch Chromium browser context using local persistent user data directory."""
        is_headless = headless if headless is not None else self.settings.BROWSER_HEADLESS

        logger.info(
            "browser.launching",
            profile_path=str(self.profile_path),
            headless=is_headless,
        )

        playwright = await async_playwright().start()
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_path),
            headless=is_headless,
            viewport={
                "width": self.settings.BROWSER_VIEWPORT_WIDTH,
                "height": self.settings.BROWSER_VIEWPORT_HEIGHT,
            },
            user_agent=USER_AGENT,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        logger.info("browser.started", profile_path=str(self.profile_path))
        return playwright, context

    async def run_interactive_setup(self) -> None:
        """Run visible browser for human operator to perform manual login & 2FA."""
        print("\n" + "=" * 70)
        print("  OPERATOR INTERACTIVE LOGIN MODE")
        print("  1. A visible Chromium browser window is opening.")
        print("  2. Log in to your Instagram account manually.")
        print("  3. Complete any two-factor authentication (2FA) or device confirmations.")
        print("  4. Once your Direct Messages inbox is visible, return here and press ENTER.")
        print("=" * 70 + "\n")

        playwright, context = await self.launch_persistent_context(headless=False)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto("https://www.instagram.com/direct/inbox/", wait_until="networkidle")

            # Wait asynchronously for operator Enter key in console
            await asyncio.to_thread(input, "Press [ENTER] after completing login and reaching your DM inbox: ")
            print("\nSession state captured. Saving profile to disk...\n")
        finally:
            await context.close()
            await playwright.stop()
            logger.info("browser.interactive_setup_completed")
