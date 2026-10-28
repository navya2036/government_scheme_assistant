"""Extract conservative scheme metadata from representative PDFs."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pymupdf


DEFAULT_MANIFEST = Path("data/manifest_all.csv")
DEFAULT_PDF_DIR = Path("gov_myscheme/text_data")
DEFAULT_OUTPUT = Path("data/scheme_metadata.csv")
MAX_FIELD_LENGTH = 500
BENEFICIARY_MAX_LENGTH = 360

OUTPUT_FIELDS = (
    "sha256",
    "relative_path",
    "filename",
    "scheme_name",
    "department",
    "state",
    "beneficiaries",
    "extraction_status",
    "pages_scanned",
    "text_char_count",
    "error_message",
)

SECTION_HEADINGS = (
    "Benefits",
    "Eligibility",
    "Exclusions",
    "Application Process",
    "Documents Required",
    "Frequently Asked Questions",
    "Sources and References",
    "Feedback",
    "Objective",
    "Details",
)

INDIAN_STATES_AND_UTS = (
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
    "The Government of NCT of Delhi",
)


def clean_text(value: str) -> str:
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = value.replace("�", "")
    return re.sub(r"\s+", " ", value).strip(" \t:-")


def shorten(value: str, limit: int = MAX_FIELD_LENGTH) -> str:
    value = clean_text(value)
    if len(value) <= limit:
        return value
    truncated = value[:limit].rsplit(" ", 1)[0]
    return f"{truncated}..."


def useful_candidate(value: str) -> bool:
    value = clean_text(value)
    if not 4 <= len(value) <= 220:
        return False
    if not re.search(r"[A-Za-z]{3}", value):
        return False
    if re.search(
        r"(?:are you sure|sign in|sign out|backdetails|back\s+details|apply now|"
        r"something went wrong|frequently asked questions|feedback)",
        value,
        re.IGNORECASE,
    ):
        return False
    return True


def extract_scheme_name(first_page_text: str) -> str:
    """Use the PDF's first-page heading, never the filename, as the title."""
    marker = re.search(
        r"Are\s+you\s+sure\s+you\s+want\s+to\s+sign\s+out\?",
        first_page_text,
        re.I,
    )
    if marker:
        prefix = clean_text(first_page_text[: marker.start()])
        if useful_candidate(prefix):
            return shorten(prefix, 220)

    candidates = []
    for index, raw_line in enumerate(first_page_text.splitlines()[:40]):
        line = clean_text(raw_line)
        if not useful_candidate(line):
            continue
        score = max(0, 12 - index)
        if re.search(
            r"(?:scheme|yojana|mission|programme|program|scholarship|pension|"
            r"subsidy|grant|insurance|fund|assistance|fellowship)",
            line,
            re.IGNORECASE,
        ):
            score += 8
        if line.isupper() or line.istitle():
            score += 2
        candidates.append((score, index, line))

    if candidates:
        return shorten(max(candidates, key=lambda item: (item[0], -item[1]))[2], 220)
    return ""


def extract_department(text: str) -> str:
    patterns = (
        r"\b((?:Ministry|Department|Directorate|Office|Authority|Commission|"
        r"Corporation|Council|Board|University|Institute|Institution)\s+"
        r"(?:of|for|on|at)\s+[A-Za-z][^.;\n]{1,120})",
        r"\b((?:Ministry|Department|Directorate|Office|Authority|Commission|"
        r"Corporation|Council|Board)\s*[:\-]\s*[^.;\n]{3,120})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = clean_text(match.group(1))
            value = re.split(
                r"\b(?:Government of|Benefits|Eligibility|Details|The entire)",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            if useful_candidate(value):
                return shorten(value)
    return ""


def extract_state(text: str) -> str:
    states = sorted(INDIAN_STATES_AND_UTS, key=len, reverse=True)
    state_pattern = "|".join(re.escape(state) for state in states)
    patterns = (
        rf"\bGovernment of (?:the )?({state_pattern})\b",
        rf"\bCheck Eligibility\s*({state_pattern})\b",
        rf"\b(?:State|U\.T\.|Union Territory)\s+of\s+({state_pattern})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return next(
                state
                for state in states
                if state.casefold() == match.group(1).casefold()
            )
    return ""


def extract_beneficiaries(text: str) -> str:
    heading_pattern = "|".join(re.escape(heading) for heading in SECTION_HEADINGS)
    label_pattern = (
        r"(?:Target Beneficiaries|Eligible Beneficiaries|Target Group|"
        r"Beneficiaries|Eligibility)"
    )
    pattern = re.compile(
        rf"\b{label_pattern}\s*:?\s*(.{{20,700}}?)(?=\b(?:{heading_pattern})\b|$)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        value = clean_text(match.group(1))
        if re.search(
            r"\b(?:eligible|individual|student|students|farmer|farmers|"
            r"women|woman|child|children|worker|workers|entrepreneur|"
            r"institution|institute|society|societies|household|citizen|"
            r"applicant|applicants|beneficiar)",
            value,
            re.IGNORECASE,
        ):
            return shorten(value)
    return ""


def extract_department_from_content(text: str) -> str:
    """Extract a short organization phrase without absorbing site navigation."""
    organization = (
        r"(?:Ministry|Department|Directorate|Office|Authority|Commission|"
        r"Corporation|Council|Board|University|Institute|Institution)"
    )
    pattern = re.compile(
        rf"(?<![A-Za-z])((?:{organization})\s+(?:of|for|on|at)\s+"
        r"[A-Za-z][^.;\n]{1,160})",
        re.IGNORECASE,
    )
    organization_start = re.compile(
        rf"(?<![A-Za-z])(?:{organization})\s+(?:of|for|on|at)\s+",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        value = clean_text(match.group(1))
        later_organization = organization_start.search(value, 1)
        if later_organization:
            value = value[: later_organization.start()]
        for stop in (
            "Government",
            "Ministry",
            "Department",
            "Directorate",
            "Quick Links",
            "About Us",
            "Screen Reader",
            "Accessibility",
        ):
            later_stop = re.search(stop, value[1:], re.IGNORECASE)
            if later_stop:
                value = value[: later_stop.start() + 1]
                break
        value = re.split(
            r"Government\s+of|Quick\s+Links|About\s+Us|Screen\s+Reader|"
            r"Accessibility|Benefits|Eligibility|Details|The entire",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        value = value.rstrip(" ,;-")
        if useful_candidate(value):
            return shorten(value)
    return ""


def extract_beneficiaries_from_content(text: str) -> str:
    """Extract only explicit target/eligibility sections, not FAQ questions."""
    heading_pattern = "|".join(re.escape(heading) for heading in SECTION_HEADINGS)
    label_pattern = r"(?:Target Beneficiaries|Eligible Beneficiaries|Target Group|Eligibility)"
    pattern = re.compile(
        rf"(?<![A-Za-z]){label_pattern}\s*:?\s*(.{{20,700}}?)(?=(?:{heading_pattern})|$)",
        re.IGNORECASE,
    )
    likely_starts = (
        "a ",
        "an ",
        "all ",
        "any ",
        "applicant",
        "beneficiar",
        "child",
        "children",
        "farmer",
        "individual",
        "member",
        "person",
        "pensioner",
        "resident",
        "student",
        "the ",
        "woman",
        "women",
        "worker",
    )
    for match in pattern.finditer(text):
        value = clean_text(match.group(1))
        if not value.casefold().startswith(likely_starts):
            continue
        if re.search(
            r"what is|how can|where can|are there|after the|what if|"
            r"how long|can beneficiaries choose",
            value,
            re.IGNORECASE,
        ):
            continue
        if re.search(
            r"eligible|individual|student|students|farmer|farmers|women|woman|"
            r"child|children|worker|workers|entrepreneur|institution|institute|"
            r"society|societies|household|citizen|applicant|applicants|beneficiar",
            value,
            re.IGNORECASE,
        ):
            return shorten(value)
    return ""


def extract_beneficiaries_improved(text: str) -> str:
    """Extract concise beneficiary descriptions from first-page content."""
    content = text
    check_match = re.search(r"Check\s+Eligibility", content, re.IGNORECASE)
    if check_match:
        details_match = re.search(
            r"Details", content[check_match.end() :], re.IGNORECASE
        )
        if details_match:
            content = content[check_match.end() + details_match.end() :]

    headings = "|".join(re.escape(heading) for heading in SECTION_HEADINGS)
    labels = (
        r"(?:Eligibility|Eligible beneficiaries|Target beneficiaries|Target group|"
        r"Beneficiaries|Who can apply|Eligible applicants|Intended beneficiaries|"
        r"Eligible persons)"
    )
    section_pattern = re.compile(
        rf"(?<![A-Za-z]){labels}\s*:?\s*", re.IGNORECASE
    )
    specific_group = re.compile(
        r"(?:farmer|student|scholar|women|woman|girl|boy|child|worker|labour|"
        r"labor|entrepreneur|institution|institute|organization|organisation|"
        r"societ(?:y|ies)|household|family|resident|patient|pensioner|artisan|"
        r"fisher(?:man|men)?|weaver|employee|youth|tribal|"
        r"scheduled caste|scheduled tribe|sc\s*/\s*st|"
        r"widow|disabled|disab(?:led|ility)|differently abled|"
        r"senior citizen|unemployed|vendor|agency|agencies|association|"
        r"research(?:er| scholar)?|teacher|doctor|sportsperson|artist|"
        r"ex-servicemen|minority|backward class|transgender|homeless|"
        r"caregiver|pupil|trainee)",
        re.IGNORECASE,
    )
    high_value_group = re.compile(
        r"(?:farmer|student|scholar|women|woman|girl|boy|child|worker|labour|"
        r"labor|entrepreneur|artisan|fisher(?:man|men)?|weaver|employee|youth|"
        r"tribal|scheduled caste|scheduled tribe|sc\s*/\s*st|widow|disabled|"
        r"disab(?:led|ility)|"
        r"differently abled|senior citizen|unemployed|vendor|agency|agencies|"
        r"association|institution|organization|organisation|research(?:er| scholar)?|"
        r"teacher|doctor|sportsperson|artist|ex-servicemen|minority|backward class|"
        r"transgender|homeless|caregiver|pupil|trainee)",
        re.IGNORECASE,
    )
    question = re.compile(
        r"(?:what is|who can|how can|where can|are there|after the|what if|"
        r"how long|can beneficiaries choose|selection procedure|sources and "
        r"references|was this helpful|news and updates|official website|"
        r"application forms|frequently asked questions)",
        re.IGNORECASE,
    )

    def concise(candidate: str) -> str:
        candidate = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", candidate)
        candidate = clean_text(candidate).lstrip("•*- ")
        if not candidate or question.search(candidate):
            return ""
        chunks = re.split(r"(?<=[.!?;])\s*(?=[A-Za-z]|[0-9]+\.)", candidate)
        fallback_chunk = ""
        for chunk in chunks[:8]:
            chunk = clean_text(chunk).lstrip("•*- ")
            if len(chunk) < 20 or not specific_group.search(chunk):
                continue
            if fallback_chunk == "":
                fallback_chunk = chunk
            has_eligibility_context = re.search(
                r"eligible|beneficiar|applicant|should|resident|all |any |for |"
                r"provided|available|covered|applicable|opportunit|support|entitled",
                chunk,
                re.IGNORECASE,
            )
            strong_lead = re.match(
                r"(?:the applicant|applicant|all |any |for |beneficiaries?|"
                r"eligible persons?|government/)",
                chunk,
                re.IGNORECASE,
            )
            if strong_lead and has_eligibility_context and high_value_group.search(chunk):
                return shorten(chunk, BENEFICIARY_MAX_LENGTH)
            if high_value_group.search(chunk) and has_eligibility_context:
                return shorten(chunk, BENEFICIARY_MAX_LENGTH)
        return shorten(fallback_chunk, BENEFICIARY_MAX_LENGTH) if fallback_chunk else ""

    for match in section_pattern.finditer(content):
        following = content[match.end() : match.end() + 1600]
        next_heading = re.search(
            rf"(?<![A-Za-z])(?:{headings})", following, re.IGNORECASE
        )
        section = following[: next_heading.start() if next_heading else 1600]
        result = concise(section)
        if result:
            return result

    narrative = re.compile(
        r"(?:the scheme is focused on|the scheme is for|the scheme is aimed at|"
        r"the scheme is applicable to|the scheme targets|intended for|available to|"
        r"provided to|provided for|benefits are available to|support is provided to)",
        re.IGNORECASE,
    )
    for sentence in re.split(
        r"(?<=[.!?;])\s*(?=[A-Za-z]|[0-9]+\.)", content
    ):
        if narrative.search(sentence):
            result = concise(sentence)
            if result:
                return result
    return ""


def representative_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    return [
        sorted(group, key=lambda row: row["relative_path"])[0]
        for _, group in sorted(groups.items())
    ]


def scan_representative(row: dict[str, str], pdf_dir: Path) -> dict[str, str]:
    relative_path = row["relative_path"]
    path = pdf_dir / Path(relative_path)
    result = {
        "sha256": row["sha256"],
        "relative_path": relative_path,
        "filename": row["filename"],
        "scheme_name": "",
        "department": "",
        "state": "",
        "beneficiaries": "",
        "extraction_status": "unreadable",
        "pages_scanned": "0",
        "text_char_count": "0",
        "error_message": "",
    }

    try:
        with pymupdf.open(path) as document:
            page_count = min(2, len(document))
            pages = [document[index].get_text("text") or "" for index in range(page_count)]
        text = "\n".join(pages)
        result["pages_scanned"] = str(page_count)
        result["text_char_count"] = str(len(text))
        if not text.strip():
            result["extraction_status"] = "no_text"
            return result

        result["scheme_name"] = extract_scheme_name(pages[0] if pages else "")
        result["department"] = extract_department_from_content(text)
        result["state"] = extract_state(text)
        result["beneficiaries"] = extract_beneficiaries_improved(text)
        result["extraction_status"] = "ok"
    except Exception as error:
        result["error_message"] = shorten(f"{type(error).__name__}: {error}")

    return result


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    title_count = sum(bool(row["scheme_name"]) for row in rows)
    department_count = sum(bool(row["department"]) for row in rows)
    state_count = sum(bool(row["state"]) for row in rows)
    beneficiary_count = sum(bool(row["beneficiaries"]) for row in rows)
    errors = sum(row["extraction_status"] == "unreadable" for row in rows)
    print(f"unique PDFs processed: {len(rows)}")
    print(f"scheme names successfully extracted: {title_count}")
    print(f"departments extracted: {department_count}")
    print(f"states extracted: {state_count}")
    print(f"beneficiaries extracted: {beneficiary_count}")
    print(f"blank beneficiary count: {len(rows) - beneficiary_count}")
    print(f"errors: {errors}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract conservative metadata from one representative per PDF hash."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    with args.manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    representatives = representative_rows(rows)
    results = [
        scan_representative(row, args.pdf_dir)
        for row in sorted(representatives, key=lambda item: item["relative_path"])
    ]
    write_csv(results, args.out)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
