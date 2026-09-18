"""Browser mock tests using Playwright headless Chromium on simulated Instagram Web DOM."""

import tempfile
from pathlib import Path
import pytest
from playwright.async_api import async_playwright

from app.ai.persona import load_character_profile
from app.ai.router import MockLLMProvider
from app.browser.instagram import InstagramBrowserAgent
from app.conversation.manager import ConversationManager
from app.core.exceptions import SecurityChallengeException
from app.storage.database import DatabaseManager

MOCK_INSTAGRAM_HTML = """
<!DOCTYPE html>
<html>
<head><title>Direct</title></head>
<body style="width: 1280px; height: 800px; margin: 0; padding: 0;">
    <div role="main">
        <div role="grid" aria-label="Messages">
            <!-- Left aligned User message -->
            <div role="row" style="position: absolute; left: 50px; top: 100px; width: 300px; height: 50px;">
                <div dir="auto">Hey Vesper, are you real?</div>
            </div>
            <!-- Right aligned Character message -->
            <div role="row" style="position: absolute; left: 800px; top: 180px; width: 300px; height: 50px;">
                <div dir="auto">Define real. I run on swap memory and irony.</div>
            </div>
            <!-- Second User message -->
            <div role="row" style="position: absolute; left: 50px; top: 260px; width: 300px; height: 50px;">
                <div dir="auto">Fair enough. Can you help me fix my code?</div>
            </div>
        </div>
    </div>
    <div role="textbox" contenteditable="true" aria-label="Message" style="position: absolute; bottom: 20px; left: 50px; width: 800px; height: 40px;"></div>
</body>
</html>
"""

MOCK_CHECKPOINT_HTML = """
<!DOCTYPE html>
<html>
<head><title>Suspicious Activity Checkpoint</title></head>
<body>
    <form id="checkpointSubmitForm">
        <div>Help us confirm it's you</div>
        <button type="submit">Submit</button>
    </form>
</body>
</html>
"""


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_browser_mock.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db


@pytest.mark.asyncio
async def test_mock_message_extraction(temp_db):
    profile = load_character_profile()
    mock_llm = MockLLMProvider()
    conv_mgr = ConversationManager(db=temp_db, llm_provider=mock_llm, character_profile=profile)
    agent = InstagramBrowserAgent(conversation_manager=conv_mgr)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.set_content(MOCK_INSTAGRAM_HTML)

        messages = await agent.extract_visible_messages(page)
        assert len(messages) == 3

        # First message is USER
        assert messages[0].sender_type == "USER"
        assert "are you real" in messages[0].text

        # Second message is CHARACTER (right-aligned, center_x > 640)
        assert messages[1].sender_type == "CHARACTER"
        assert "swap memory" in messages[1].text

        # Third message is USER
        assert messages[2].sender_type == "USER"
        assert "fix my code" in messages[2].text

        await browser.close()


@pytest.mark.asyncio
async def test_mock_typing_and_sending(temp_db):
    profile = load_character_profile()
    mock_llm = MockLLMProvider()
    conv_mgr = ConversationManager(db=temp_db, llm_provider=mock_llm, character_profile=profile)
    agent = InstagramBrowserAgent(conversation_manager=conv_mgr)

    # Set delays to 0 for instant test execution
    agent.settings.MIN_REPLY_DELAY_SECONDS = 0.05
    agent.settings.MAX_REPLY_DELAY_SECONDS = 0.1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.set_content(MOCK_INSTAGRAM_HTML)

        test_text = "Testing typing simulation."
        await agent.type_and_send(page, test_text)

        # Check content of textbox
        input_el = await page.query_selector("div[role='textbox']")
        assert input_el is not None
        typed_content = await input_el.inner_text()
        assert test_text in typed_content

        await browser.close()


@pytest.mark.asyncio
async def test_security_challenge_immediate_halt(temp_db):
    profile = load_character_profile()
    mock_llm = MockLLMProvider()
    conv_mgr = ConversationManager(db=temp_db, llm_provider=mock_llm, character_profile=profile)
    agent = InstagramBrowserAgent(conversation_manager=conv_mgr)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(MOCK_CHECKPOINT_HTML)

        # check_security_checkpoints should raise SecurityChallengeException immediately
        with pytest.raises(SecurityChallengeException) as exc_info:
            await agent.check_security_checkpoints(page)

        assert "security challenge encountered" in str(exc_info.value).lower()

        # Verify audit event was logged as CRITICAL
        events = conv_mgr.audit_repo.get_recent_events(limit=5)
        assert len(events) >= 1
        assert events[0]["severity"] == "CRITICAL"

        await browser.close()


MOCK_PROFILE_HTML = """
<!DOCTYPE html>
<html>
<head><title>haiclop on Instagram</title></head>
<body>
    <header>
        <div>haiclop</div>
        <button>Message</button>
    </header>
    <div role="textbox" contenteditable="true" aria-label="Message"></div>
</body>
</html>
"""


@pytest.mark.asyncio
async def test_mock_text_user_outreach(temp_db):
    profile = load_character_profile()
    mock_llm = MockLLMProvider(canned_response="hey @haiclop, what's up?")
    conv_mgr = ConversationManager(db=temp_db, llm_provider=mock_llm, character_profile=profile)
    agent = InstagramBrowserAgent(conversation_manager=conv_mgr)
    agent.settings.MIN_REPLY_DELAY_SECONDS = 0.05
    agent.settings.MAX_REPLY_DELAY_SECONDS = 0.1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.set_content(MOCK_PROFILE_HTML)

        # Mock page.goto to preserve local DOM
        async def mock_goto(*args, **kwargs):
            return None
        page.goto = mock_goto

        sent = await agent.text_user(
            page=page,
            username="@haiclop",
            force_available=True,
        )
        assert sent is not None
        assert "haiclop" in sent

        # Check textbox content
        input_el = await page.query_selector("div[role='textbox']")
        assert input_el is not None
        text_content = await input_el.inner_text()
        assert sent in text_content

        await browser.close()


MOCK_FLOATING_DRAWER_HTML = """
<!DOCTYPE html>
<html>
<body style="margin:0; padding:0; width: 1280px; height: 800px;">
    <!-- Background profile content -->
    <div style="padding: 20px;">
        <h1>haiclop</h1>
        <div>Followed by tashvi_mordani, pri...</div>
    </div>

    <!-- Floating Chat Modal / Drawer -->
    <div role="dialog" class="x1n2onr6 x78zum5" style="position: fixed; right: 80px; bottom: 20px; width: 380px; height: 500px; background: #fff; display: flex; flex-direction: column;">
        <div style="height: 50px; display: flex; align-items: center; padding: 0 10px;">
            <button>&lt;</button>
            <div dir="auto" style="font-weight: bold; margin-left: 10px;">Arnav Srivastava</div>
        </div>

        <div style="flex: 1; overflow-y: auto; padding: 10px;">
            <div style="display: flex; margin-bottom: 8px; justify-content: flex-start;">
                <div style="background: #efefef; border-radius: 18px; padding: 8px 12px;">
                    <div dir="auto">Hi</div>
                </div>
            </div>
            <div style="display: flex; margin-bottom: 4px; justify-content: flex-end;">
                <div style="background: rgb(55, 151, 240); border-radius: 18px; padding: 8px 12px;">
                    <div dir="auto" style="color: #fff;">waise tu kya kar raha hai maths</div>
                </div>
            </div>
            <div style="display: flex; margin-bottom: 8px; justify-content: flex-end;">
                <div style="background: rgb(55, 151, 240); border-radius: 18px; padding: 8px 12px;">
                    <div dir="auto" style="color: #fff;">dekh ke bore ho gayi hu</div>
                </div>
            </div>
            <div style="display: flex; margin-bottom: 4px; justify-content: flex-start;">
                <div style="background: #efefef; border-radius: 18px; padding: 8px 12px;">
                    <div dir="auto">Chalo movie dekhte hai</div>
                </div>
            </div>
            <div style="text-align: right;">
                <span dir="auto">Seen just now</span>
            </div>
        </div>

        <div style="height: 55px; position: relative;">
            <div style="position: absolute; left: 45px; right: 100px;">
                <div role="textbox" contenteditable="true" aria-label="Message..." style="width: 100%; min-height: 20px;"></div>
            </div>
        </div>
    </div>
</body>
</html>
"""


@pytest.mark.asyncio
async def test_floating_modal_drawer_extraction(temp_db):
    profile = load_character_profile()
    mock_llm = MockLLMProvider()
    conv_mgr = ConversationManager(db=temp_db, llm_provider=mock_llm, character_profile=profile)
    agent = InstagramBrowserAgent(conversation_manager=conv_mgr)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.set_content(MOCK_FLOATING_DRAWER_HTML)

        messages = await agent.extract_visible_messages(page)
        assert len(messages) == 4
        assert messages[0].sender_type == "USER"
        assert messages[0].text == "Hi"
        assert messages[1].sender_type == "CHARACTER"
        assert "maths" in messages[1].text
        assert messages[2].sender_type == "CHARACTER"
        assert "bore ho gayi hu" in messages[2].text
        assert messages[3].sender_type == "USER"
        assert messages[3].text == "Chalo movie dekhte hai"

        await browser.close()

