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


def extract_titles_degrees(text: str) -> Dict[str, List[str]]:
    if not text:
        return {"titles": [], "degrees": []}

    text_lower = text.lower()

    # TECH JOB TITLES (controlled vocabulary)
    TECH_TITLES = [
        # Engineering
        "software engineer", "backend engineer", "frontend engineer",
        "full stack engineer", "fullstack engineer",
        "machine learning engineer", "ml engineer",
        "data engineer", "platform engineer",
        "cloud engineer", "devops engineer",

        # Developer roles
        "software developer", "backend developer", "frontend developer",
        "full stack developer", "fullstack developer",
        "android developer", "ios developer", "mobile developer",
        "web developer",

        # Data roles
        "data scientist", "applied scientist",
        "data analyst", "business intelligence analyst",
        "data science intern", "ml scientist",

        # Architecture / Infra
        "solutions architect", "cloud architect",
        "system architect", "systems engineer",
        "security engineer", "cybersecurity analyst",

        # Managers/Lead (only tech)
        "engineering manager", "technical lead", "tech lead",
        "product engineer", "ai engineer",
        "qa engineer", "qa tester", "test automation engineer"
    ]

    found_titles = []
    for title in TECH_TITLES:
        pattern = re.compile(rf"\b{re.escape(title)}\b", re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            found_titles.extend(matches)

    # Clean + dedupe
    found_titles = sorted(set([t.strip() for t in found_titles]))

    # DEGREE PATTERNS (accept all sectors)
    DEGREE_REGEX = re.compile(
        r"""
        (?:
            bachelor\sof\s[a-zA-Z ]+ |
            master\sof\s[a-zA-Z ]+   |
            doctor\sof\s[a-zA-Z ]+   |
            associate\sof\s[a-zA-Z ]+|
            diploma\sin\s[a-zA-Z ]+  |

            b\.\s?sc\b |
            bsc\b |
            b\.?eng\b |
            beng\b |
            b\.?tech\b |
            btech\b |

            m\.\s?sc\b |
            msc\b |
            m\.?eng\b |
            meng\b |
            m\.?tech\b |
            mtech\b |
            mba\b   |

            ph\.?d\b |
            phd\b
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    degree_matches = DEGREE_REGEX.findall(text)

    # normalize formatting
    degree_matches = [
        re.sub(r"\s+", " ", d).strip().title() for d in degree_matches
    ]

    degree_matches = sorted(set(degree_matches))

    return {
        "titles": found_titles,
        "degrees": degree_matches
    }
