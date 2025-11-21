import re
from typing import Dict, List, Optional
import phonenumbers
from phonenumbers import PhoneNumberFormat

PHONE_REGEX = re.compile(
    r"""
    # Country code (optional)
    (?:(?:\+|00)\d{1,3}[\s\-\.()]*)?

    # Area code (optional)
    (?:\(?\d{1,4}\)?[\s\-\.()]*)?

    # Main number digits (at least 6–12 digits total)
    (?:\d[\s\-\.()]*){6,14}
    """,
    re.VERBOSE
)

EMAIL_REGEX = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

date_pattern = re.compile(
    r'''
    (
        \b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b              # YYYY-MM-DD
        |
        \b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b            # DD-MM-YYYY or short year
        |
        \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec
            |January|February|March|April|May|June|July|August|September|
            October|November|December)
        \s+\d{1,2},?\s+\d{2,4}\b                     # Month DD, YYYY
        |
        \b\d{1,2}\s+
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec
            |January|February|March|April|May|June|July|August|September|
            October|November|December)
        \s+\d{2,4}\b                                 # DD Month YYYY
    )
    ''',
    re.VERBOSE | re.IGNORECASE
)

def extract_phone(
    raw_phones: List[str],
    region: Optional[str] = None,

) -> List[str]:
    """
    Extract region-validated phone numbers

    raw_phones: List of phone numbers need to validate
    region: ISO 2-letter country code (e.g., 'US', 'VN', 'GB', 'IN')
    Returns a list of valid phone numbers in E.164 format (+CountryCode Number).
    """
    valid_numbers = []
    for raw in raw_phones:
        cleaned = re.sub(r"[^\d+]", "", raw)    # keep digits and '+'

        try:
            # Parse phone number for given region
            nums = phonenumbers.parse(cleaned, region)

            if phonenumbers.is_possible_number(nums) and phonenumbers.is_valid_number(nums):
                # Convert to international E.164 format
                formatted = phonenumbers.format_number(
                    nums,
                    phonenumbers.PhoneNumberFormat.E164
                )
                valid_numbers.append(formatted)
        except phonenumbers.NumberParseException as e:
            raise(e)
    
    return valid_numbers


def extract_contacts(text: str, region: Optional[str] = None) -> Dict[str, List[str]]:
    """Extract emails, phone, numbers, and dates from free text.
    region: ISO 2-letter country code (e.g, 'US', 'GB', 'VN', 'IN')
    Returns a dict: {'email': [...], 'phone': [...], 'dates': [...]}.
    """

    if not text:
        return {"email": [], "phone": [], "dates": []}

    # Email 
    emails = list(set(EMAIL_REGEX.findall(text)))

    # Phone 
    raw_phones = PHONE_REGEX.findall(text)
    valid_phones = extract_phone(raw_phones, region)

    # Date
    dates = [d.strip() for d in date_pattern.findall(text)]

    # Sort
    def clean_list(items: List[str]) -> List[str]:
        return sorted(list({i.strip() for i in items if i.strip()}))
    
    test = {
        "email": clean_list(emails),
        "phone": clean_list(valid_phones),
        "dates": clean_list(dates)
    }

    return {
        "email": clean_list(emails),
        "phone": clean_list(valid_phones),
        "dates": clean_list(dates)
    }
