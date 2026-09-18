"""Instagram Web Playwright automation: DOM extraction, human-like typing, and security halts."""

from __future__ import annotations

import asyncio
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from playwright.async_api import BrowserContext, ElementHandle, Page

from app.browser.models import BrowserDiagnosticSnapshot, DOMMessage
from app.browser.selectors import SELECTORS
from app.conversation.manager import ConversationManager
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    MessageSendFailedException,
    SecurityChallengeException,
    SelectorNotFoundException,
)
from app.core.logging import get_logger

logger = get_logger("browser.instagram")


class InstagramBrowserAgent:
    """Automates Instagram Web DMs via Playwright while strictly obeying anti-circumvention boundaries."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        settings: Settings | None = None,
    ):
        self.conv_mgr = conversation_manager
        self.settings = settings or get_settings()
        self._is_running = False
        self._recently_sent_texts: list[tuple[float, str]] = []
        self._last_processed_user_text: str | None = None

    def _normalize_echo_text(self, text: str) -> str:
        """Normalize text for echo matching by stripping emojis, punctuation, and extra whitespace."""
        t = re.sub(r"[^\w\s]", "", text.lower())
        return " ".join(t.split())

    def _is_recent_echo(self, text: str, max_age_seconds: float = 120.0) -> bool:
        """Check if candidate text matches any recently sent message from the bot."""
        now = time.time()
        self._recently_sent_texts = [
            (ts, norm) for ts, norm in self._recently_sent_texts if now - ts <= max_age_seconds
        ]
        cand_norm = self._normalize_echo_text(text)
        if not cand_norm:
            return False
        for _, sent_norm in self._recently_sent_texts:
            if cand_norm == sent_norm:
                return True
            if len(cand_norm) > 4 and len(sent_norm) > 4:
                if cand_norm in sent_norm or sent_norm in cand_norm:
                    return True
        return False

    def _record_sent_text(self, text: str) -> None:
        """Record dispatched message in echo cache."""
        norm = self._normalize_echo_text(text)
        if norm:
            self._recently_sent_texts.append((time.time(), norm))

    async def check_security_checkpoints(self, page: Page) -> None:
        """Inspect page for CAPTCHAs, suspicious activity modals, or action blocks.
        HALTS IMMEDIATELY IF DETECTED. NEVER ATTEMPTS BYPASS.
        """
        for selector in SELECTORS["checkpoint_indicators"]:
            try:
                el = await page.query_selector(selector)
                if el and await el.is_visible():
                    logger.critical(
                        "browser.security_challenge_detected",
                        selector=selector,
                        url=page.url,
                    )
                    diag = await self.capture_diagnostics(page, trigger_reason=f"Matched checkpoint: {selector}")
                    raise SecurityChallengeException(
                        message=f"Instagram security challenge encountered at {page.url}. System halted safely.",
                        details={"url": page.url, "screenshot": diag.screenshot_path},
                    )
            except SecurityChallengeException:
                raise
            except Exception:
                continue

    async def capture_diagnostics(self, page: Page, trigger_reason: str) -> BrowserDiagnosticSnapshot:
        """Capture full page screenshot and DOM HTML for operator inspection."""
        diag_dir = Path("logs/diagnostics")
        diag_dir.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        screenshot_path = diag_dir / f"diag_{now_str}.png"
        dom_path = diag_dir / f"diag_{now_str}.html"

        try:
            await page.screenshot(path=str(screenshot_path), full_page=True)
            html_content = await page.content()
            dom_path.write_text(html_content, encoding="utf-8")
        except Exception as e:
            logger.warning("browser.failed_saving_diagnostics", error=str(e))

        # Record audit event
        self.conv_mgr.audit_repo.record_event(
            event_type="BROWSER_DIAGNOSTIC_CAPTURED",
            severity="CRITICAL" if "checkpoint" in trigger_reason.lower() else "ERROR",
            component="BROWSER",
            payload={
                "trigger_reason": trigger_reason,
                "url": page.url,
                "screenshot": str(screenshot_path),
                "dom_html": str(dom_path),
            },
        )

        return BrowserDiagnosticSnapshot(
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            url=page.url,
            screenshot_path=str(screenshot_path),
            dom_html_path=str(dom_path),
            trigger_reason=trigger_reason,
        )

    async def find_element_with_fallbacks(self, page: Page, selector_key: str) -> ElementHandle | None:
        """Attempt locating an element using ordered fallback selectors."""
        patterns = SELECTORS.get(selector_key, [])
        for sel in patterns:
            try:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    return el
            except Exception:
                continue
        return None

    async def extract_visible_messages(self, page: Page) -> list[DOMMessage]:
        """Extract recent text bubbles from active conversation DOM with high-precision geometry and styling."""
        # 1. Primary extraction via in-browser DOM geometry and styling
        try:
            dom_data = await page.evaluate(
                r"""
                () => {
                    // 1. Locate message input textbox (anchor for the active chat thread/drawer)
                    const textbox = document.querySelector("div[role='textbox'][contenteditable='true']") 
                                 || document.querySelector("div[aria-label*='Message']")
                                 || document.querySelector("textarea[placeholder*='Message']");
                    if (!textbox) {
                        return null;
                    }
                    const tbRect = textbox.getBoundingClientRect();

                    // 2. Locate active chat drawer / dialog if present
                    const parentEl = textbox.parentElement;
                    const dialog = textbox.closest("div[role='dialog']") 
                                || (parentEl ? parentEl.closest("div[style*='position: fixed'], div[style*='position: absolute']") : null);
                    
                    let scrollContainer = null;
                    if (dialog) {
                        const allDivs = Array.from(dialog.querySelectorAll("div"));
                        for (const d of allDivs) {
                            const cs = window.getComputedStyle(d);
                            if ((cs.overflowY === 'auto' || cs.overflowY === 'scroll') && d.clientHeight > 100) {
                                scrollContainer = d;
                                break;
                            }
                        }
                    }

                    // 2b. Define chat column geometry using the input box as the primary horizontal anchor
                    const chatLeft = tbRect.left;
                    const chatRight = tbRect.right;
                    const chatCenter = chatLeft + (tbRect.width / 2);

                    // Vertical and horizontal boundaries
                    let topBoundary = Math.max(35, tbRect.top - 700);
                    if (scrollContainer) {
                        topBoundary = Math.max(topBoundary, scrollContainer.getBoundingClientRect().top);
                    } else if (dialog) {
                        topBoundary = Math.max(topBoundary, dialog.getBoundingClientRect().top + 48);
                    }
                    const bottomBoundary = tbRect.top + 5;
                    const leftBoundary = chatLeft - 65;
                    const rightBoundary = chatRight + 65;

                    // 3. Query candidate text elements
                    const searchRoot = scrollContainer || dialog || document;
                    const candidates = Array.from(searchRoot.querySelectorAll("div[dir='auto'], span[dir='auto'], div[role='row'], div[role='listitem']"));
                    const messages = [];
                    const statusRegex = /^(seen|delivered|sent|active|today|yesterday|\d{1,2}:\d{2})/i;

                    for (let idx = 0; idx < candidates.length; idx++) {
                        const el = candidates[idx];
                        if (textbox.contains(el)) continue;

                        // Skip buttons, links, headers, and system banners
                        if (el.closest("button") || el.closest("a[role='link']") || el.closest("header") || el.closest("div[role='banner']")) continue;

                        const text = (el.innerText || "").trim();
                        if (!text || text.length === 0) continue;
                        if (statusRegex.test(text)) continue;

                        const lower = text.toLowerCase();
                        if (lower.startsWith("followed by") || lower.includes("followed by")) continue;
                        if (lower.startsWith("suggested") || lower.includes("more posts")) continue;
                        if (lower === "message..." || lower === "message") continue;

                        const rect = el.getBoundingClientRect();
                        if (rect.height === 0 || rect.width === 0) continue;

                        // Geometry filters: must be inside message vertical & horizontal zone of this chat
                        if (rect.bottom > bottomBoundary || rect.top < topBoundary) continue;
                        if (rect.right < leftBoundary || rect.left > rightBoundary) continue;

                        // Deduplicate nested elements with same text and position
                        let duplicate = false;
                        for (const m of messages) {
                            if (m.text === text && Math.abs(m.y - rect.top) < 12) {
                                duplicate = true;
                                break;
                            }
                            if (m.text.includes(text) && Math.abs(m.y - rect.top) < 12) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (duplicate) continue;

                        // 4. Multi-Signal Sender Classification (Positive = CHARACTER / Right, Negative = USER / Left)
                        let senderScore = 0;

                        // Signal A: Avatar image in row (only present on incoming USER messages in Instagram)
                        const row = el.closest("div[role='row'], div[role='listitem']") || el.parentElement.parentElement;
                        if (row) {
                            const avatar = row.querySelector("img, svg[aria-label]");
                            if (avatar) {
                                const avRect = avatar.getBoundingClientRect();
                                if (avRect.left < rect.left && avRect.width >= 14 && avRect.height >= 14) {
                                    senderScore -= 10; // Incoming user message
                                }
                            }
                        }

                        // Signal B: Background Color (Blue/Purple theme = CHARACTER, Neutral White/Grey = USER)
                        let p = el;
                        for (let d = 0; d < 4 && p && p !== document.body; d++) {
                            const style = window.getComputedStyle(p);
                            const bg = style.backgroundColor;
                            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                                const rgb = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
                                if (rgb) {
                                    const r = parseInt(rgb[1], 10);
                                    const g = parseInt(rgb[2], 10);
                                    const b = parseInt(rgb[3], 10);

                                    // Outgoing blue/purple theme
                                    if ((b > r + 25 && b > g + 15) || (b > 160 && g < 140)) {
                                        senderScore += 8;
                                        break;
                                    }
                                    // Incoming neutral grey / white
                                    if (Math.abs(r - g) <= 15 && Math.abs(g - b) <= 15 && Math.abs(r - b) <= 15) {
                                        if ((r > 190 && g > 190 && b > 190) || (r < 80 && g < 80 && b < 80)) {
                                            senderScore -= 6;
                                            break;
                                        }
                                    }
                                }
                            }
                            p = p.parentElement;
                        }

                        // Signal C: Physical horizontal position relative to chatCenter (anchored to textbox)
                        const bubbleCenter = rect.left + (rect.width / 2);
                        if (bubbleCenter > chatCenter + 15) {
                            senderScore += 6;
                        } else if (bubbleCenter < chatCenter - 15) {
                            senderScore -= 6;
                        }

                        // Distance to edges of the input box
                        const distToLeft = rect.left - chatLeft;
                        const distToRight = chatRight - rect.right;
                        if (distToRight < 40 && distToRight < distToLeft) {
                            senderScore += 4;
                        } else if (distToLeft < 80 && distToLeft < distToRight) {
                            senderScore -= 4;
                        }

                        // Signal D: Immediate row flex alignment (only 2 levels up to avoid ancestor modal wrappers)
                        let rowParent = el.parentElement;
                        for (let d = 0; d < 2 && rowParent && rowParent !== document.body; d++) {
                            const style = window.getComputedStyle(rowParent);
                            if (style.justifyContent === 'flex-end' || style.alignSelf === 'flex-end') {
                                senderScore += 3;
                                break;
                            } else if (style.justifyContent === 'flex-start' || style.alignSelf === 'flex-start') {
                                senderScore -= 3;
                                break;
                            }
                            rowParent = rowParent.parentElement;
                        }

                        messages.push({
                            text: text,
                            is_right_aligned: senderScore > 0,
                            y: rect.top,
                            element_index: idx
                        });
                    }

                    // 3b. Query Reel and Media attachments (Instagram video previews, shared reels, clips)
                    const mediaElements = Array.from(searchRoot.querySelectorAll(
                        "a[href*='/reel/'], a[href*='/p/'], a[href*='/stories/'], video, svg[aria-label*='Clip' i], svg[aria-label*='Reels' i], svg[aria-label*='Play' i]"
                    ));
                    const processedMediaContainers = new Set();

                    for (let mIdx = 0; mIdx < mediaElements.length; mIdx++) {
                        const mEl = mediaElements[mIdx];
                        if (textbox.contains(mEl)) continue;
                        if (mEl.closest("header") || mEl.closest("div[role='banner']")) continue;

                        const rowContainer = mEl.closest("div[role='row'], div[role='listitem']") || mEl;
                        if (processedMediaContainers.has(rowContainer)) continue;
                        processedMediaContainers.add(rowContainer);

                        const mRect = (rowContainer !== mEl ? rowContainer : mEl).getBoundingClientRect();
                        if (mRect.height === 0 || mRect.width === 0) continue;
                        if (mRect.bottom > bottomBoundary || mRect.top < topBoundary) continue;
                        if (mRect.right < leftBoundary || mRect.left > rightBoundary) continue;

                        // Check if a message was already registered at this approximate vertical position (+/- 30px)
                        let matched = false;
                        for (const m of messages) {
                            if (Math.abs(m.y - mRect.top) < 30) {
                                matched = true;
                                if (!m.text.includes("[Shared a Reel]")) {
                                    m.text = "[Shared a Reel] " + m.text;
                                }
                                break;
                            }
                        }
                        if (matched) continue;

                        // Multi-signal sender classification for the media element
                        let senderScore = 0;
                        const row = mEl.closest("div[role='row'], div[role='listitem']") || mEl.parentElement.parentElement;
                        if (row) {
                            const avatar = row.querySelector("img, svg[aria-label]");
                            if (avatar) {
                                const avRect = avatar.getBoundingClientRect();
                                if (avRect.left < mRect.left && avRect.width >= 14 && avRect.height >= 14) {
                                    senderScore -= 10; // Incoming user message
                                }
                            }
                        }

                        // Physical horizontal position relative to chatCenter
                        const bubbleCenter = mRect.left + (mRect.width / 2);
                        if (bubbleCenter > chatCenter + 15) {
                            senderScore += 6;
                        } else if (bubbleCenter < chatCenter - 15) {
                            senderScore -= 6;
                        }

                        // Row flex alignment
                        let rowParent = mEl.parentElement;
                        for (let d = 0; d < 3 && rowParent && rowParent !== document.body; d++) {
                            const style = window.getComputedStyle(rowParent);
                            if (style.justifyContent === 'flex-end' || style.alignSelf === 'flex-end') {
                                senderScore += 3;
                                break;
                            } else if (style.justifyContent === 'flex-start' || style.alignSelf === 'flex-start') {
                                senderScore -= 3;
                                break;
                            }
                            rowParent = rowParent.parentElement;
                        }

                        messages.push({
                            text: "[Shared a Reel]",
                            is_right_aligned: senderScore > 0,
                            y: mRect.top,
                            element_index: 9000 + mIdx
                        });
                    }

                    messages.sort((a, b) => a.y - b.y);
                    return messages;
                }
                """
            )
            if dom_data is not None and len(dom_data) > 0:
                sender_handle = self.settings.TARGET_USERNAME or "InstagramUser"
                parsed_messages = []
                for i, d in enumerate(dom_data):
                    msg_text = d["text"]
                    is_right = d["is_right_aligned"]

                    # Bot-echo override: if recently dispatched by this agent, it is ALWAYS CHARACTER
                    if self._is_recent_echo(msg_text):
                        sender_type = "CHARACTER"
                    else:
                        sender_type = "CHARACTER" if is_right else "USER"

                    parsed_messages.append(
                        DOMMessage(
                            sender_handle=sender_handle,
                            sender_type=sender_type,
                            text=msg_text,
                            element_index=d.get("element_index", i),
                        )
                    )
                return parsed_messages
        except Exception as e:
            logger.debug("browser.js_dom_eval_fallback", error=str(e))

        # 2. Fallback selector-based extraction (for mock tests or static DOM)
        messages = []
        rows = []
        for row_sel in SELECTORS.get("message_rows", []):
            try:
                found = await page.query_selector_all(row_sel)
                if found:
                    rows = found
                    break
            except Exception:
                continue

        # Anchor to textbox center for drawer resilience in fallback
        tb_el = await page.query_selector("div[role='textbox'], textarea[placeholder*='Message']")
        tb_box = await tb_el.bounding_box() if tb_el else None
        page_box = await page.evaluate("() => ({ width: window.innerWidth })")
        center_anchor = (tb_box["x"] + tb_box["width"] / 2.0) if (tb_box and tb_box.get("width", 0) > 100) else (page_box["width"] * 0.5)

        for idx, row in enumerate(rows):
            try:
                text_el = None
                for t_sel in SELECTORS.get("message_text", []):
                    text_el = await row.query_selector(t_sel)
                    if text_el:
                        break

                if not text_el:
                    reel_el = await row.query_selector(
                        "a[href*='/reel/'], a[href*='/p/'], a[href*='/stories/'], video, svg[aria-label*='Clip' i], svg[aria-label*='Play' i]"
                    )
                    if reel_el:
                        raw_text = "[Shared a Reel]"
                    else:
                        continue
                else:
                    raw_text = (await text_el.inner_text()).strip()
                    if not raw_text:
                        continue

                box = await row.bounding_box()
                is_right_aligned = False
                if box and center_anchor:
                    center_x = box["x"] + (box["width"] / 2.0)
                    is_right_aligned = center_x > center_anchor

                if self._is_recent_echo(raw_text):
                    sender_type = "CHARACTER"
                else:
                    sender_type = "CHARACTER" if is_right_aligned else "USER"
                sender_handle = self.settings.TARGET_USERNAME or "InstagramUser"

                messages.append(
                    DOMMessage(
                        sender_handle=sender_handle,
                        sender_type=sender_type,
                        text=raw_text,
                        element_index=idx,
                    )
                )
            except Exception:
                continue

        return messages

    async def type_and_send(self, page: Page, text: str) -> None:
        """Type response as a multi-bubble conversational burst with human-like cadence simulation."""
        from app.conversation.splitter import MultiBubbleSplitter

        input_el = await self.find_element_with_fallbacks(page, "message_input")
        if not input_el:
            raise SelectorNotFoundException("message_input", details={"url": page.url})

        # Split into 1 to 3 conversational bubbles
        bubbles = MultiBubbleSplitter.split(text)
        if not bubbles:
            bubbles = [text.strip()]

        # Record full text in echo cache
        self._record_sent_text(text)

        logger.info("browser.burst_started", total_bubbles=len(bubbles), text=text[:50])

        # Initial cognitive reaction pause (simulating reading and thinking)
        await asyncio.sleep(random.uniform(0.3, 0.6))

        for idx, bubble in enumerate(bubbles):
            clean_bubble = bubble.strip()
            if not clean_bubble:
                continue

            # Checkpoint check prior to typing each bubble
            await self.check_security_checkpoints(page)

            await input_el.click()
            await asyncio.sleep(random.uniform(0.15, 0.3))

            logger.info("browser.typing_bubble", index=idx + 1, total=len(bubbles), length=len(clean_bubble))

            # Simulate human keystrokes with brisk natural typing speed
            for char in clean_bubble:
                await page.keyboard.type(char, delay=random.uniform(18, 40))
                if char in [".", ",", "!", "?", "😭"]:
                    await asyncio.sleep(random.uniform(0.05, 0.12))

            # Brief hesitation pause before sending bubble
            await asyncio.sleep(random.uniform(0.12, 0.25))
            await page.keyboard.press("Enter")
            self._record_sent_text(clean_bubble)
            logger.info("browser.bubble_dispatched", index=idx + 1, total=len(bubbles))

            # If more bubbles remain, introduce a brief inter-bubble gap
            if idx < len(bubbles) - 1:
                inter_gap = random.uniform(0.3, 0.7)
                await asyncio.sleep(inter_gap)

        # Randomized post-send cool down
        post_delay = random.uniform(
            self.settings.MIN_REPLY_DELAY_SECONDS,
            self.settings.MAX_REPLY_DELAY_SECONDS,
        )
        logger.info("browser.burst_completed", cooldown_seconds=round(post_delay, 2))
        await asyncio.sleep(post_delay)

    async def monitor_thread(
        self,
        page: Page,
        conversation_id: str,
        sender_handle: str,
        max_turns: int | None = None,
        force_available: bool = False,
    ) -> None:
        """Continuous polling loop observing DOM and dispatching replies."""
        self._is_running = True
        turns_processed = 0

        logger.info(
            "browser.monitoring_started",
            conversation_id=conversation_id,
            target_user=sender_handle,
            force_available=force_available,
        )

        try:
            poll_count = 0
            last_seen_msg_text = None

            while self._is_running:
                poll_count += 1
                # 1. Platform challenge check
                await self.check_security_checkpoints(page)

                # 2. Extract DOM messages
                visible_messages = await self.extract_visible_messages(page)

                if visible_messages:
                    latest = visible_messages[-1]
                    if latest.text != last_seen_msg_text:
                        last_seen_msg_text = latest.text
                        logger.info(
                            "browser.dom_state_updated",
                            total_visible=len(visible_messages),
                            latest_sender=latest.sender_type,
                            latest_snippet=latest.text[:40],
                        )
                elif poll_count % 6 == 0:
                    logger.debug("browser.monitoring_tick", poll_count=poll_count, target=sender_handle)

                # 3. If latest message is from USER, handle it
                if visible_messages and visible_messages[-1].sender_type == "USER":
                    latest_user_msg = visible_messages[-1]
                    clean_user_text = latest_user_msg.text.strip()

                    # Deduplicate: Skip if already processed
                    if clean_user_text == self._last_processed_user_text:
                        await asyncio.sleep(self.settings.INBOX_POLL_INTERVAL_SECONDS)
                        continue

                    # Echo suppression: Skip if this text is a recent dispatch from the bot itself
                    if self._is_recent_echo(clean_user_text):
                        logger.warning("browser.suppressing_self_echo", text=clean_user_text)
                        self._last_processed_user_text = clean_user_text
                        await asyncio.sleep(self.settings.INBOX_POLL_INTERVAL_SECONDS)
                        continue

                    self._last_processed_user_text = clean_user_text

                    logger.info(
                        "browser.inbound_user_message_detected",
                        text=latest_user_msg.text,
                        sender=sender_handle,
                    )

                    reply_text = await self.conv_mgr.handle_incoming_message(
                        conversation_id=conversation_id,
                        sender_handle=sender_handle,
                        message_text=latest_user_msg.text,
                        force_available=force_available,
                    )

                    if reply_text:
                        print(f"\n[Replying to {sender_handle}]: {reply_text}\n")
                        await self.type_and_send(page, reply_text)
                        turns_processed += 1
                        if max_turns and turns_processed >= max_turns:
                            logger.info("browser.max_turns_reached", count=turns_processed)
                            break

                # Sleep before next poll tick
                await asyncio.sleep(self.settings.INBOX_POLL_INTERVAL_SECONDS)

        except asyncio.CancelledError:
            logger.info("browser.monitoring_cancelled")
        finally:
            self._is_running = False

    async def dismiss_popups(self, page: Page) -> None:
        """Dismiss common Instagram popups (Not Now, Turn on notifications, Save login info)."""
        for sel in SELECTORS.get("dismiss_modal", []):
            try:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    await el.click()
                    await asyncio.sleep(0.5)
            except Exception:
                pass

    async def open_user_chat(self, page: Page, username: str) -> bool:
        """Navigate to a target user's DM thread on Instagram Web."""
        clean_handle = username.lstrip("@").strip()
        logger.info("browser.navigating_to_user", user=clean_handle)

        # Strategy 1: Navigate to user profile and click Message button
        profile_url = f"https://www.instagram.com/{clean_handle}/"
        await page.goto(profile_url, wait_until="domcontentloaded", timeout=self.settings.BROWSER_NAVIGATION_TIMEOUT_MS)
        await self.check_security_checkpoints(page)
        await self.dismiss_popups(page)
        await asyncio.sleep(1.5)

        # Extract profile identity if present (e.g. "Arnav Srivastava (@haiclop) • Instagram")
        try:
            page_title = await page.title()
            if f"(@{clean_handle})" in page_title:
                real_name = page_title.split(f"(@{clean_handle})")[0].strip()
                if real_name and real_name.lower() != clean_handle.lower():
                    logger.info("browser.profile_name_detected", user=clean_handle, name=real_name)
                    conv_id = f"thread_{clean_handle}"
                    self.conv_mgr.conv_repo.get_or_create_conversation(conv_id, participant_handle=f"@{clean_handle}")
                    # Avoid duplicate insert if already present
                    existing = self.conv_mgr.memory_mgr.repo.get_memories_for_conversation(conv_id)
                    if not any(f"name is {real_name.lower()}" in m.statement.lower() for m in existing):
                        self.conv_mgr.memory_mgr.repo.add_memory(conv_id, "FACT", f"User's name is {real_name}", confidence=1.0)
        except Exception as exc:
            logger.debug("browser.profile_name_extract_failed", error=str(exc))


        # Check if chat is ALREADY open on page
        input_el = await self.find_element_with_fallbacks(page, "message_input")
        if input_el:
            logger.info("browser.chat_already_open", user=clean_handle)
            return True

        msg_btn = await self.find_element_with_fallbacks(page, "profile_message_button")
        if msg_btn:
            logger.info("browser.found_profile_message_button", user=clean_handle)
            await msg_btn.click()
            await self.check_security_checkpoints(page)
            await self.dismiss_popups(page)

            # Poll for message_input up to 5 seconds as drawer animates
            for _ in range(5):
                await asyncio.sleep(1.0)
                input_el = await self.find_element_with_fallbacks(page, "message_input")
                if input_el:
                    logger.info("browser.chat_ready", user=clean_handle)
                    return True

        # Check if missing Message button is due to unauthenticated session
        login_prompt = await page.query_selector("a[href*='/accounts/login'], button:has-text('Log In'), button:has-text('Log in'), div:has-text('Log in to see photos')")
        if login_prompt or "accounts/login" in page.url:
            logger.warning("browser.login_required")
            print("\n" + "!" * 70)
            print("  INSTAGRAM SESSION NOT FOUND: AUTHENTICATION REQUIRED")
            print("  Instagram requires an authenticated session to send Direct Messages.")
            print("  Please log in once manually using:")
            print("    .venv\\Scripts\\python.exe -m app.main --setup-login")
            print("  Once authenticated, your session is saved in data/browser_profile/")
            print("!" * 70 + "\n")
            return False

        # Strategy 2: Direct Inbox compose search
        logger.info("browser.trying_inbox_search", user=clean_handle)
        await page.goto("https://www.instagram.com/direct/inbox/", wait_until="domcontentloaded", timeout=self.settings.BROWSER_NAVIGATION_TIMEOUT_MS)
        await self.check_security_checkpoints(page)
        await self.dismiss_popups(page)
        await asyncio.sleep(1.0)

        new_msg_btn = await self.find_element_with_fallbacks(page, "new_chat_button")
        if new_msg_btn:
            await new_msg_btn.click()
            await asyncio.sleep(1.0)
            search_input = await self.find_element_with_fallbacks(page, "search_user_input")
            if search_input:
                await search_input.fill(clean_handle)
                await asyncio.sleep(1.5)
                user_result = await page.query_selector(f"div[role='button']:has-text('{clean_handle}'), div[role='checkbox']:has-text('{clean_handle}')")
                if user_result:
                    await user_result.click()
                    await asyncio.sleep(0.5)
                    submit_btn = await self.find_element_with_fallbacks(page, "chat_submit_button")
                    if submit_btn:
                        await submit_btn.click()
                        await asyncio.sleep(2.0)
                        return True

        input_el = await self.find_element_with_fallbacks(page, "message_input")
        return input_el is not None

    async def text_user(
        self,
        page: Page,
        username: str,
        custom_message: str | None = None,
        force_available: bool = False,
    ) -> str | None:
        """Open chat with username, craft or take message, and type/send it."""
        clean_handle = username if username.startswith("@") else f"@{username}"
        conv_id = f"thread_{clean_handle.lstrip('@')}"

        # 1. Open chat on Instagram
        success = await self.open_user_chat(page, clean_handle)
        if not success:
            logger.error("browser.failed_opening_user_chat", user=clean_handle)
            return None

        # 2. Check if there's already an unreplied user message in the thread
        if not custom_message:
            visible = await self.extract_visible_messages(page)
            if visible and visible[-1].sender_type == "USER":
                latest_user_text = visible[-1].text
                logger.info("browser.unreplied_user_message_found", user=clean_handle, text=latest_user_text)
                reply = await self.conv_mgr.handle_incoming_message(
                    conversation_id=conv_id,
                    sender_handle=clean_handle,
                    message_text=latest_user_text,
                    force_available=force_available,
                )
                if reply:
                    print(f"\n[Replying to {clean_handle}]: {reply}\n")
                    await self.type_and_send(page, reply)
                    return reply

        # 3. Otherwise prepare / generate proactive outreach message
        msg_to_send = custom_message
        if not msg_to_send:
            msg_to_send = await self.conv_mgr.initiate_conversation(
                conversation_id=conv_id,
                target_handle=clean_handle,
                force_available=force_available,
            )

        if not msg_to_send:
            logger.warning("browser.text_user_no_message_generated", user=clean_handle)
            return None

        # 4. Type and send
        await self.type_and_send(page, msg_to_send)
        logger.info("browser.text_user_dispatched", user=clean_handle, text=msg_to_send[:30])
        return msg_to_send

    async def scrape_chat_history(
        self,
        page: Page,
        username: str,
        max_scrolls: int = 12,
    ) -> dict[str, Any]:
        """Scrape historical messages from a conversation thread to analyze user style and conversational patterns."""
        clean_handle = username if username.startswith("@") else f"@{username}"
        success = await self.open_user_chat(page, clean_handle)
        if not success:
            logger.error("browser.scrape_chat_failed_open", user=clean_handle)
            return {"error": f"Failed to open chat with {clean_handle}"}

        logger.info("browser.scraping_chat_started", user=clean_handle, max_scrolls=max_scrolls)
        print(f"\n[Scraping chat with {clean_handle}] Scrolling to load previous messages...")

        # Find the scrollable message container and scroll up smoothly
        for s in range(1, max_scrolls + 1):
            await page.evaluate(
                """
                () => {
                    const textbox = document.querySelector("div[role='textbox'][contenteditable='true']") || document.querySelector("div[aria-label*='Message']");
                    if (!textbox) return false;
                    let curr = textbox.parentElement;
                    while (curr && curr !== document.body) {
                        if (curr.scrollHeight > curr.clientHeight + 40) {
                            curr.scrollTop = 0;
                            return true;
                        }
                        curr = curr.parentElement;
                    }
                    window.scrollBy(0, -500);
                    return false;
                }
                """
            )
            print(f"  Loaded scroll step {s}/{max_scrolls}...")
            await asyncio.sleep(1.2)

        # Extract all loaded messages
        all_dom_messages = await self.extract_visible_messages(page)
        logger.info("browser.scrape_chat_extracted", count=len(all_dom_messages))

        import json
        user_messages: list[str] = []
        target_messages: list[str] = []

        for m in all_dom_messages:
            if m.sender_type == "CHARACTER":
                user_messages.append(m.text)
            else:
                target_messages.append(m.text)

        export_path = Path(f"data/scraped_chat_{clean_handle.lstrip('@')}.json")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = {
            "target_user": clean_handle,
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_messages": len(all_dom_messages),
            "account_owner_messages": user_messages,
            "target_user_messages": target_messages,
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        # Train StyleLearner on the extracted messages
        style_learner = self.conv_mgr.style_learner
        for msg in target_messages:
            style_learner.observe_message(clean_handle, msg)

        print(f"\n==========================================")
        print(f"  SCRAPING COMPLETE: @{clean_handle.lstrip('@')}")
        print(f"==========================================")
        print(f"  Total messages extracted: {len(all_dom_messages)}")
        print(f"  Messages sent by you:     {len(user_messages)}")
        print(f"  Messages from friend:     {len(target_messages)}")
        print(f"  Saved raw dataset to:     {export_path}")
        print(f"  Style profile updated in SQLite database.\n")

        return dataset

