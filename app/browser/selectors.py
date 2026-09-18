"""Centralized DOM selector definitions with resilient fallback chains."""

from typing import Final

SELECTORS: Final[dict[str, list[str]]] = {
    # Direct Messages list and container
    "chat_container": [
        "div[role='main'] div[role='grid']",
        "div[role='main'] section",
        "div[aria-label='Messages']",
        "div[role='presentation'] div[role='rowgroup']",
    ],
    "message_rows": [
        "div[role='row']",
        "div[role='listitem']",
        "div[data-testid='message-row']",
        "div[class*='x78zum5'][class*='xdt5ytf']",
    ],
    "message_text": [
        "div[dir='auto']",
        "span[dir='auto']",
        "div[class*='x1lliihq']",
    ],

    # Message input field (Instagram uses contenteditable divs for direct input)
    "message_input": [
        "div[role='textbox'][contenteditable='true']",
        "div[aria-label='Message']",
        "div[aria-label='Message...']",
        "div[aria-placeholder='Message...']",
        "p[class*='xat24cr']",
    ],

    # Send button (fallback if Enter does not dispatch)
    "send_button": [
        "button:has-text('Send')",
        "div[role='button']:has-text('Send')",
        "button[type='submit']",
    ],

    # Profile page navigation & Direct message initiation
    "profile_message_button": [
        "header button:has-text('Message')",
        "header div[role='button']:has-text('Message')",
        "button:has-text('Message')",
        "div[role='button']:has-text('Message')",
        "a[href*='/direct/t/']",
    ],
    "new_chat_button": [
        "svg[aria-label='New message']",
        "div[role='button']:has-text('New message')",
        "svg[aria-label='Compose']",
        "button:has-text('Send message')",
    ],
    "search_user_input": [
        "input[name='queryBox']",
        "input[placeholder*='Search']",
        "input[aria-label*='Search']",
    ],
    "chat_submit_button": [
        "div[role='button']:has-text('Chat')",
        "button:has-text('Chat')",
        "div[role='button']:has-text('Next')",
    ],
    "dismiss_modal": [
        "button:has-text('Not Now')",
        "button:has-text('Cancel')",
        "button:has-text('Later')",
        "button:has-text('Close')",
    ],

    # Security checkpoints & anti-bot challenges (NEVER bypass, always halt)
    "checkpoint_indicators": [
        "form#checkpointSubmitForm",
        "div:has-text('Suspicious Activity')",
        "div:has-text('Confirm your info')",
        "div:has-text('Help us confirm it\\'s you')",
        "div:has-text('Action Blocked')",
        "iframe[src*='recaptcha']",
        "iframe[src*='hcaptcha']",
        "iframe[src*='challenge']",
    ],

    # Login detection
    "login_form": [
        "form#loginForm",
        "input[name='username']",
        "input[name='password']",
    ],
}
