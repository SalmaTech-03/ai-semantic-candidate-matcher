import re

def scrub_pii(text: str) -> str:
    """Masks emails and phone numbers to ensure GDPR compliance."""
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\+?\d[\d -]{8,12}\d', '[PHONE]', text)
    return text