from src.notifications.telegram_controller import TelegramController


def test_escape_markdown() -> None:
    text = "Price: *100_?*[test](url)"
    sanitized = TelegramController._escape_markdown(text)
    assert sanitized == "Price: \\*100\\_?\\*\\[test\\]\\(url\\)"
