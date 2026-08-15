import io
import json
import os
import re
import sqlite3
import base64
from datetime import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import boto3
from botocore.exceptions import BotoCoreError, ClientError


# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="AI Sales Assistant",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# OpenAI
# =========================================================

api_key = None
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("找不到 OPENAI_API_KEY，請到 Streamlit Cloud → Settings → Secrets 設定。")
    st.stop()

client = OpenAI(api_key=api_key)


# =========================================================
# AWS (optional integration)
# =========================================================

def _secret_or_env(name, default=None):
    try:
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return str(value)
    except Exception:
        pass
    value = os.getenv(name)
    return value if value not in (None, "") else default


AWS_REGION = _secret_or_env("AWS_REGION") or _secret_or_env("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = _secret_or_env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _secret_or_env("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = _secret_or_env("AWS_SESSION_TOKEN")
AWS_S3_BUCKET = _secret_or_env("AWS_S3_BUCKET")
AWS_S3_PREFIX = (_secret_or_env("AWS_S3_PREFIX", "sync-pipeline") or "sync-pipeline").strip("/")
AWS_BEDROCK_MODEL_ID = _secret_or_env("AWS_BEDROCK_MODEL_ID")


def get_aws_session():
    """Create a boto3 Session from Streamlit secrets/env vars.

    Supports both long-lived access keys and hackathon temporary credentials.
    If AWS_SESSION_TOKEN exists it is passed automatically.
    """
    kwargs = {}
    if AWS_REGION:
        kwargs["region_name"] = AWS_REGION
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        if AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    return boto3.Session(**kwargs)


def get_aws_identity():
    session = get_aws_session()
    sts = session.client("sts")
    return sts.get_caller_identity()


def get_s3_client():
    return get_aws_session().client("s3")


def upload_json_to_s3(payload, key):
    if not AWS_S3_BUCKET:
        raise RuntimeError("尚未設定 AWS_S3_BUCKET。")
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    get_s3_client().put_object(
        Bucket=AWS_S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
        ServerSideEncryption="AES256",
    )
    return f"s3://{AWS_S3_BUCKET}/{key}"


def backup_meeting_to_s3(result):
    """Upload the final CRM JSON exactly as generated, without adding extra fields."""
    meeting_json = result.get("meeting_json", {})
    meeting_id = meeting_json.get("meeting_id") or ("MTG-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    date_part = meeting_json.get("meeting_date") or datetime.now().strftime("%Y-%m-%d")
    key = f"{AWS_S3_PREFIX}/meetings/{date_part}/{meeting_id}.json"
    return upload_json_to_s3(meeting_json, key)


def list_recent_s3_meetings(limit=20):
    if not AWS_S3_BUCKET:
        return []
    response = get_s3_client().list_objects_v2(
        Bucket=AWS_S3_BUCKET, Prefix=f"{AWS_S3_PREFIX}/meetings/", MaxKeys=max(1, min(int(limit), 1000))
    )
    items = response.get("Contents", [])
    items.sort(key=lambda x: x.get("LastModified"), reverse=True)
    return items[:limit]



def _normalize_meeting_payload(payload):
    """Normalize current/legacy meeting JSON shapes without inventing data."""
    if not isinstance(payload, dict):
        return None

    # Current wrappers used by earlier prototypes.
    if isinstance(payload.get("meeting_json"), dict):
        payload = payload["meeting_json"]

    # Current strict CRM JSON already lives at top level.
    if any(k in payload for k in ["meeting_id", "company", "contact_name", "stage", "need"]):
        return payload

    # Some older builds may keep a structured record under sales_intelligence.
    if isinstance(payload.get("sales_intelligence"), dict):
        nested = payload["sales_intelligence"]
        if any(k in nested for k in ["meeting_id", "company", "contact_name", "stage", "need"]):
            return nested

    return payload


def _meeting_company(payload):
    """Read company from current and legacy payload shapes."""
    if not isinstance(payload, dict):
        return None

    company = payload.get("company")
    if company:
        return str(company).strip()

    meeting_info = payload.get("meeting_info")
    if isinstance(meeting_info, dict) and meeting_info.get("company"):
        return str(meeting_info["company"]).strip()

    account = payload.get("account")
    if isinstance(account, dict) and account.get("company"):
        return str(account["company"]).strip()

    return None


@st.cache_data(ttl=60, show_spinner=False)
def list_all_s3_meeting_json(max_objects=500):
    """Read Meeting JSON objects from S3 under sync-pipeline/meetings/.

    Returns:
        [
          {
            "key": "...",
            "last_modified": "...",
            "payload": {...}
          }
        ]
    """
    if not AWS_S3_BUCKET:
        return []

    prefix = f"{AWS_S3_PREFIX}/meetings/"
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    results = []
    seen = 0

    for page_data in paginator.paginate(Bucket=AWS_S3_BUCKET, Prefix=prefix):
        for item in page_data.get("Contents", []):
            if seen >= max_objects:
                return results

            key = item.get("Key", "")
            if not key.lower().endswith(".json"):
                continue

            try:
                obj = s3.get_object(Bucket=AWS_S3_BUCKET, Key=key)
                body = obj["Body"].read().decode("utf-8")
                payload = json.loads(body)
                payload = _normalize_meeting_payload(payload)

                if isinstance(payload, dict):
                    results.append({
                        "key": key,
                        "last_modified": (
                            item.get("LastModified").isoformat()
                            if item.get("LastModified")
                            else None
                        ),
                        "payload": payload,
                    })
                    seen += 1
            except Exception:
                # Skip one malformed/unreadable object instead of breaking the whole KB.
                continue

    return results


def get_local_meeting_records():
    """Convert local SQLite rows into usable knowledge records.

    Supports both the current strict meeting_json column and legacy rows.
    """
    records = []

    for meeting in get_meetings():
        (
            row_id,
            created_at,
            company,
            customer_name,
            salesperson,
            target_language,
            transcript,
            translation,
            analysis,
            meeting_json_raw,
        ) = meeting

        payload = safe_json_loads(meeting_json_raw)
        payload = _normalize_meeting_payload(payload) if payload else None

        if not isinstance(payload, dict):
            parsed_analysis = safe_json_loads(analysis)

            if isinstance(parsed_analysis, dict):
                payload = _normalize_meeting_payload(parsed_analysis)

            if not isinstance(payload, dict):
                # Legacy fallback: preserve only facts already present in SQLite.
                payload = {
                    "meeting_id": f"legacy_local_{row_id}",
                    "meeting_date": (
                        str(created_at)[:10]
                        if created_at else None
                    ),
                    "company": company,
                    "contact_name": customer_name,
                    "salesperson": salesperson,
                    "transcript": transcript,
                    "translation": translation,
                    "legacy_record": True,
                }

        records.append({
            "source": "local",
            "source_id": f"sqlite:{row_id}",
            "payload": payload,
        })

    return records


def get_s3_meeting_records(max_objects=500):
    records = []
    for item in list_all_s3_meeting_json(max_objects=max_objects):
        records.append({
            "source": "s3",
            "source_id": f"s3://{AWS_S3_BUCKET}/{item['key']}",
            "payload": item["payload"],
        })
    return records


def get_all_knowledge_records(include_local=True, include_s3=True):
    """Merge local + AWS S3 records and deduplicate by meeting_id/source."""
    combined = []

    if include_local:
        combined.extend(get_local_meeting_records())

    if include_s3:
        combined.extend(get_s3_meeting_records())

    deduped = []
    seen = set()

    for item in combined:
        payload = item.get("payload") or {}
        meeting_id = (
            payload.get("meeting_id")
            if isinstance(payload, dict)
            else None
        )

        dedupe_key = (
            f"meeting_id:{meeting_id}"
            if meeting_id
            else item.get("source_id")
        )

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        deduped.append(item)

    return deduped


def list_knowledge_companies(records):
    companies = set()
    for item in records:
        company = _meeting_company(item.get("payload"))
        if company:
            companies.add(company)
    return sorted(companies)


def get_company_knowledge_from_records(company_name, records):
    """Return all matching customer records from local SQLite and/or S3."""
    target = (company_name or "").strip()
    matched = []

    for item in records:
        payload = item.get("payload")
        if _meeting_company(payload) == target:
            # Include source traceability for debugging, but keep business JSON intact.
            if isinstance(payload, dict):
                record = dict(payload)
                record["_knowledge_source"] = item.get("source")
                record["_knowledge_source_id"] = item.get("source_id")
                matched.append(record)

    return matched


def bedrock_converse(user_text, system_text=None):
    if not AWS_BEDROCK_MODEL_ID:
        raise RuntimeError("尚未設定 AWS_BEDROCK_MODEL_ID。")
    client_br = get_aws_session().client("bedrock-runtime")
    kwargs = {
        "modelId": AWS_BEDROCK_MODEL_ID,
        "messages": [
            {"role": "user", "content": [{"text": user_text}]}
        ],
        "inferenceConfig": {
            "maxTokens": 800,
            "temperature": 0.2,
        },
    }
    if system_text:
        kwargs["system"] = [{"text": system_text}]
    response = client_br.converse(**kwargs)
    content = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip(), response


# =========================================================
# Constants / schemas
# =========================================================

DB_FILE = "sales_meetings.db"
AUDIO_TYPES = ["mp3", "m4a", "wav", "webm", "mp4", "mpeg", "mpga", "ogg", "flac"]
DOCUMENT_TYPES = ["pdf", "txt", "json"]

# Current planning rates used only for on-screen estimates.
# Recheck official OpenAI pricing before production use.
GPT5_MINI_INPUT_PER_1M = 0.25
GPT5_MINI_OUTPUT_PER_1M = 2.00
DIARIZE_INPUT_PER_1M = 2.50
DIARIZE_OUTPUT_PER_1M = 10.00
# When diarized transcription returns duration-only usage, the API response does not
# expose token counts. For budgeting only, use a conservative planning allowance.
DIARIZE_BUDGET_PER_MINUTE = 0.01

OCR_LANGUAGE_OPTIONS = {
    "繁體中文 + English": "chi_tra+eng",
    "簡體中文 + English": "chi_sim+eng",
    "English": "eng",
    "日本語 + English": "jpn+eng",
    "한국어 + English": "kor+eng",
}

AUDIO_INPUT_LANGUAGE_OPTIONS = {
    "中文（Mandarin / 中文語音）": "zh",
    "English（英文語音）": "en",
}

SALES_INTELLIGENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "meeting_id": {"type": "string"},
        "meeting_date": {"type": "string"},
        "company": {"type": ["string", "null"]},
        "contact_name": {"type": ["string", "null"]},
        "contact_role": {"type": ["string", "null"]},
        "customer_type": {"type": ["string", "null"]},
        "stage": {
            "type": "string",
            "enum": [
                "初次接觸",
                "需求確認",
                "提案中",
                "議價中",
                "決策中",
                "成交",
                "暫緩",
                "流失",
                "未判定"
            ]
        },
        "plan": {"type": ["string", "null"]},
        "need": {"type": ["string", "null"]},
        "budget": {"type": ["integer", "null"]},
        "budget_confidence": {
            "type": "string",
            "enum": ["明確", "推估", "未提及"]
        },
        "timeline": {"type": ["string", "null"]},
        "objection": {"type": ["string", "null"]},
        "decision_maker": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": ["string", "null"]},
                "role": {"type": ["string", "null"]},
                "attended": {"type": "boolean"}
            },
            "required": ["name", "role", "attended"]
        },
        "next_action": {"type": ["string", "null"]},
        "follow_up_raw": {"type": ["string", "null"]},
        "follow_up_date": {"type": ["string", "null"]},
        "quotes": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "budget": {"type": ["string", "null"]},
                "objection": {"type": ["string", "null"]},
                "plan": {"type": ["string", "null"]},
                "decision_maker": {"type": ["string", "null"]}
            },
            "required": ["budget", "objection", "plan", "decision_maker"]
        }
    },
    "required": [
        "meeting_id",
        "meeting_date",
        "company",
        "contact_name",
        "contact_role",
        "customer_type",
        "stage",
        "plan",
        "need",
        "budget",
        "budget_confidence",
        "timeline",
        "objection",
        "decision_maker",
        "next_action",
        "follow_up_raw",
        "follow_up_date",
        "quotes"
    ]
}

DEMO_TRANSCRIPT = """Speaker A: 我們公司目前有 20 個業務，每天都會跟很多客戶開會，但是大家都不太願意把會議內容整理進 CRM，所以主管常常不知道案件現在談到哪裡。
Speaker B: 了解。那如果會議結束後可以自動整理成 CRM 可用資料，對你們會有幫助嗎？
Speaker A: 會。如果一個月 5000 元以內，我覺得可以考慮。不過最後還是要跟我們老闆確認。
Speaker B: 好，那我整理一份方案跟 Demo 給你。
Speaker A: 可以，你下週一再跟我聯絡。"""


# =========================================================
# Database
# =========================================================

def init_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            company TEXT,
            customer_name TEXT,
            salesperson TEXT,
            target_language TEXT,
            transcript TEXT,
            translation TEXT,
            analysis TEXT,
            meeting_json TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS lead_discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            source_company TEXT,
            result_json TEXT
        )
        """
    )

    cursor.execute("PRAGMA table_info(meetings)")
    columns = {row[1] for row in cursor.fetchall()}
    if "meeting_json" not in columns:
        cursor.execute("ALTER TABLE meetings ADD COLUMN meeting_json TEXT")

    conn.commit()
    conn.close()


def save_meeting(result):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO meetings (
            created_at,
            company,
            customer_name,
            salesperson,
            target_language,
            transcript,
            translation,
            analysis,
            meeting_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            result.get("company"),
            result.get("customer_name"),
            result.get("salesperson"),
            result.get("target_language"),
            result.get("transcript"),
            result.get("translation"),
            json.dumps(result.get("sales_intelligence", {}), ensure_ascii=False),
            json.dumps(result.get("meeting_json", {}), ensure_ascii=False),
        ),
    )

    conn.commit()
    conn.close()


def get_meetings():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            id,
            created_at,
            company,
            customer_name,
            salesperson,
            target_language,
            transcript,
            translation,
            analysis,
            meeting_json
        FROM meetings
        ORDER BY id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# =========================================================
# Helpers
# =========================================================

def safe_json_loads(value):
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def stage_to_opportunity_score(stage):
    """Dashboard-only grouping derived from the CRM stage; not part of the final JSON schema."""
    mapping = {
        "成交": "HIGH",
        "決策中": "HIGH",
        "議價中": "HIGH",
        "提案中": "MEDIUM",
        "需求確認": "MEDIUM",
        "初次接觸": "LOW",
        "暫緩": "LOW",
        "流失": "LOW",
        "未判定": "UNKNOWN",
    }
    return mapping.get(stage, "UNKNOWN")


def extract_opportunity_score(analysis):
    """Support both the new stage-based CRM record and legacy opportunity_score records."""
    parsed = safe_json_loads(analysis)
    if isinstance(parsed, dict):
        stage_score = stage_to_opportunity_score(parsed.get("stage"))
        if stage_score != "UNKNOWN":
            return stage_score
        legacy_score = parsed.get("opportunity_score")
        if legacy_score in {"HIGH", "MEDIUM", "LOW"}:
            return legacy_score

    text = str(analysis or "").upper()
    match = re.search(r"OPPORTUNITY SCORE.*?\\b(HIGH|MEDIUM|LOW)\\b", text, re.DOTALL)
    if match:
        return match.group(1)
    for score in ("HIGH", "MEDIUM", "LOW"):
        if score in text:
            return score
    return "UNKNOWN"


def build_speaker_segments(transcript):
    segments = []
    for segment in transcript.segments:
        segments.append(
            {
                "id": getattr(segment, "id", None),
                "speaker": str(segment.speaker),
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": segment.text.strip(),
            }
        )
    return segments


def parse_labeled_transcript(text):
    """將 Speaker A: / 張經理： / 業務： 等逐行稿轉成 segments。"""
    text = (text or "").strip()
    if not text:
        return []

    segments = []
    current_speaker = None
    current_parts = []

    def flush():
        nonlocal current_speaker, current_parts
        if current_parts:
            segments.append(
                {
                    "id": None,
                    "speaker": current_speaker or "Unknown",
                    "start": None,
                    "end": None,
                    "text": " ".join(current_parts).strip(),
                }
            )
        current_parts = []

    speaker_pattern = re.compile(r"^\s*([^：:\n]{1,40})\s*[：:]\s*(.+?)\s*$")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = speaker_pattern.match(line)
        if match:
            label = match.group(1).strip()
            content = match.group(2).strip()
            # 避免把一般含冒號句子誤認為 speaker label
            likely_speaker = (
                label.lower().startswith("speaker")
                or label.lower().startswith("sales")
                or label.lower().startswith("customer")
                or label.lower().startswith("agent")
                or label.lower().startswith("client")
                or any(k in label for k in ["業務", "客戶", "經理", "主管", "老闆", "顧問", "代表"])
                or len(label) <= 12
            )
            if likely_speaker:
                flush()
                current_speaker = label
                current_parts = [content]
                continue
        current_parts.append(line)

    flush()

    if not segments:
        return [
            {
                "id": None,
                "speaker": "Unknown",
                "start": None,
                "end": None,
                "text": text,
            }
        ]
    return segments


def format_speaker_transcript(segments):
    lines = []
    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        start = segment.get("start")
        end = segment.get("end")
        if start is not None and end is not None:
            label = f"{speaker} [{float(start):.2f}s - {float(end):.2f}s]"
        else:
            label = speaker
        lines.append(f"{label}: {segment.get('text', '').strip()}")
    return "\n".join(lines)


def _usage_attr(usage, name, default=0):
    if usage is None:
        return default
    if isinstance(usage, dict):
        return usage.get(name, default)
    return getattr(usage, name, default)


def response_usage_record(label, model, response):
    usage = getattr(response, "usage", None)
    input_tokens = int(_usage_attr(usage, "input_tokens", 0) or 0)
    output_tokens = int(_usage_attr(usage, "output_tokens", 0) or 0)
    cost = 0.0
    if model == "gpt-5-mini":
        cost = (input_tokens / 1_000_000) * GPT5_MINI_INPUT_PER_1M + (output_tokens / 1_000_000) * GPT5_MINI_OUTPUT_PER_1M
    return {
        "label": label,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 6),
        "estimate_basis": "response token usage",
    }


def transcription_usage_record(transcript):
    usage = getattr(transcript, "usage", None)
    usage_type = _usage_attr(usage, "type", None)
    if usage_type == "tokens":
        input_tokens = int(_usage_attr(usage, "input_tokens", 0) or 0)
        output_tokens = int(_usage_attr(usage, "output_tokens", 0) or 0)
        cost = (input_tokens / 1_000_000) * DIARIZE_INPUT_PER_1M + (output_tokens / 1_000_000) * DIARIZE_OUTPUT_PER_1M
        return {
            "label": "speaker diarization transcription",
            "model": "gpt-4o-transcribe-diarize",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "estimate_basis": "response token usage",
        }

    seconds = float(_usage_attr(usage, "seconds", 0) or 0)
    if not seconds:
        seconds = float(getattr(transcript, "duration", 0) or 0)
    budget_cost = (seconds / 60.0) * DIARIZE_BUDGET_PER_MINUTE
    return {
        "label": "speaker diarization transcription",
        "model": "gpt-4o-transcribe-diarize",
        "seconds": round(seconds, 2),
        "estimated_cost_usd": round(budget_cost, 6),
        "estimate_basis": "conservative budget allowance ($0.01/min; verify exact billing in OpenAI Usage/Costs)",
    }


def extract_pdf_native_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    page_texts = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        page_texts.append(text)
    combined = "\n\n".join(
        f"[Page {idx}]\n{text}"
        for idx, text in enumerate(page_texts, start=1)
        if text
    ).strip()
    return combined, page_texts, len(reader.pages)


def pdf_looks_scanned(page_texts):
    if not page_texts:
        return True
    total_chars = sum(len(x.strip()) for x in page_texts)
    nonempty_pages = sum(1 for x in page_texts if len(x.strip()) >= 20)
    page_count = len(page_texts)
    # OCR if there is almost no text layer, or most pages have no usable text.
    return total_chars < max(80, page_count * 40) or nonempty_pages < max(1, page_count * 0.5)


def ocr_pdf_tesseract(file_bytes, ocr_lang="chi_tra+eng", dpi=220, max_pages=40):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if len(doc) > max_pages:
        raise ValueError(
            f"這份 PDF 有 {len(doc)} 頁，目前 OCR 安全上限為 {max_pages} 頁。"
            "請先拆分 PDF，或調高程式中的 max_pages。"
        )

    pages = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, lang=ocr_lang, config="--psm 3").strip()
        pages.append(f"[Page {idx}]\n{text}" if text else f"[Page {idx}]\n")
    return "\n\n".join(pages).strip(), len(doc)


def ocr_pdf_openai_fallback(file_bytes, filename):
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    response = client.responses.create(
        model="gpt-5-mini",
        instructions=(
            "You are an OCR engine. Extract all readable text from every page of the PDF in page order. "
            "Do not summarize, translate, explain, or infer missing words. Preserve numbers, names, tables as plain text, "
            "speaker labels, and page boundaries. Prefix each page with [Page N]."
        ),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "OCR this PDF exactly and return only the extracted text."},
                    {
                        "type": "input_file",
                        "filename": filename or "document.pdf",
                        "file_data": encoded,
                    },
                ],
            }
        ],
        store=False,
    )
    return response.output_text.strip(), response_usage_record("GPT PDF OCR fallback", "gpt-5-mini", response)


def read_pdf_with_ocr(
    file_bytes,
    filename="document.pdf",
    force_ocr=False,
    ocr_lang="chi_tra+eng",
    allow_gpt_fallback=False,
):
    native_text, page_texts, page_count = extract_pdf_native_text(file_bytes)
    scanned = pdf_looks_scanned(page_texts)

    if native_text and not force_ocr and not scanned:
        return native_text, {
            "pdf_pages": page_count,
            "pdf_text_method": "native_text_layer",
            "ocr_used": False,
        }, []

    try:
        text, ocr_pages = ocr_pdf_tesseract(file_bytes, ocr_lang=ocr_lang)
        if not text.strip():
            raise ValueError("Tesseract OCR 沒有辨識到文字。")
        return text, {
            "pdf_pages": ocr_pages,
            "pdf_text_method": "tesseract_ocr",
            "ocr_used": True,
            "ocr_language": ocr_lang,
        }, []
    except Exception as local_ocr_error:
        if not allow_gpt_fallback:
            raise RuntimeError(
                "PDF 看起來是掃描文件，但本機 Tesseract OCR 失敗。"
                f"\n原因：{local_ocr_error}\n"
                "若部署在 Streamlit Cloud，請確認 packages.txt 已安裝 tesseract 語言包；"
                "或勾選『允許 GPT OCR 備援』。"
            ) from local_ocr_error

        text, usage = ocr_pdf_openai_fallback(file_bytes, filename)
        if not text.strip():
            raise ValueError("GPT OCR 備援沒有讀取到文字。")
        return text, {
            "pdf_pages": page_count,
            "pdf_text_method": "gpt_ocr_fallback",
            "ocr_used": True,
            "ocr_language": "auto",
            "local_ocr_error": str(local_ocr_error),
        }, [usage]


def read_txt_text(file_bytes):
    for encoding in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def extract_segments_from_json(data):
    if isinstance(data, dict):
        # 本系統 meeting JSON
        transcription = data.get("transcription")
        if isinstance(transcription, dict):
            segments = transcription.get("segments")
            if isinstance(segments, list) and segments:
                cleaned = []
                for idx, seg in enumerate(segments):
                    if not isinstance(seg, dict):
                        continue
                    cleaned.append(
                        {
                            "id": seg.get("id"),
                            "speaker": str(seg.get("speaker", "Unknown")),
                            "start": seg.get("start"),
                            "end": seg.get("end"),
                            "text": str(seg.get("text", "")).strip(),
                        }
                    )
                if cleaned:
                    full_text = transcription.get("full_text") or " ".join(
                        seg["text"] for seg in cleaned
                    )
                    return str(full_text).strip(), cleaned

        # 常見 transcript 欄位
        for key in ("transcript", "text", "full_text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), parse_labeled_transcript(value)

    # 任意 JSON：保留內容讓後續 AI 能讀
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return text, parse_labeled_transcript(text)


def transcribe_audio(file_bytes, filename, mime_type, source_language_code):
    # 明確指定中文或英文，能給轉錄模型更好的語言提示。
    # gpt-4o-transcribe-diarize 仍負責 Speaker A / B / C 分離。
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe-diarize",
        file=(filename, file_bytes, mime_type or "application/octet-stream"),
        language=source_language_code,
        response_format="diarized_json",
        chunking_strategy="auto",
    )
    full_text = transcript.text.strip()
    segments = build_speaker_segments(transcript)
    duration = round(float(transcript.duration), 2) if getattr(transcript, "duration", None) is not None else None
    return full_text, segments, duration, transcription_usage_record(transcript)


def translate_speaker_transcript(speaker_transcript, target_language):
    response = client.responses.create(
        model="gpt-5-mini",
        instructions=f"""
你是一個專業商業會議翻譯助手。
請將使用者提供的多人講者逐字稿翻譯成 {target_language}。

規則：
1. 保留原意，不摘要、不解釋。
2. 保留每個 speaker label；若有時間戳也要保留。
3. 公司、人名、價格、日期與數字必須保留。
4. 口語要自然。
5. 只輸出翻譯後逐字稿。
""",
        input=speaker_transcript,
        store=False,
    )
    return response.output_text.strip(), response_usage_record("translation", "gpt-5-mini", response)


def analyze_sales_strict(transcription_payload):
    response = client.responses.create(
        model="gpt-5-mini",
        instructions="""
你是一個台灣 B2B Sales Intelligence Assistant。
你的任務是把會議資料轉換成「單一、可直接寫入 CRM / S3 / LINE / Gmail pipeline」的嚴格 JSON Record。

輸出欄位已由 JSON Schema 固定。請遵守以下規則：

1. 只能使用輸入 metadata 與逐字稿中實際存在的資訊，不可捏造。
2. meeting_id：原封不動複製輸入的 meeting_id。
3. meeting_date：取輸入 created_at 的日期部分，格式 YYYY-MM-DD。
4. company：原封不動複製 meeting_info.company；沒有則 null。
5. contact_name：原封不動複製 meeting_info.customer_name；沒有則 null。
6. contact_role：只在逐字稿有足夠證據時填寫，例如「行銷經理」；否則 null。
7. customer_type：只在能從逐字稿合理辨識時填寫，例如品牌方、代理商、企業客戶；否則 null。
8. stage 只能選：初次接觸、需求確認、提案中、議價中、決策中、成交、暫緩、流失、未判定。
   - 不確定時使用「未判定」，不要硬猜。
9. plan：客戶實際討論或接受的方案，例如年約、月約、Pilot；未提及則 null。
10. need：用一句精簡文字整理客戶核心需求；沒有明確需求則 null。
11. budget：只輸出整數新台幣金額，不含「元」「萬」或逗號；沒有則 null。
12. budget_confidence：
   - 客戶明確確認正式預算 → 「明確」
   - 「大概」「左右」「抓一百二」等非精確或需語境換算 → 「推估」
   - 完全沒提 → 「未提及」
13. 台灣商務口語金額可依上下文換算。例如「今年這塊大概抓一百二左右」若語境明確指年度預算百萬元級，可解析為 1200000，並標示「推估」；若語境不足則 budget=null。
14. timeline：產品導入、採購、上線或專案時程；沒有則 null。不要把單純 follow-up 日期重複填進 timeline。
15. objection：只填最主要成交阻礙；沒有則 null。
16. decision_maker：
   - name：有具名才填，否則 null。
   - role：例如「老闆」「業務副總」；沒有則 null。
   - attended：只有決策者本人確定出席本場 Meeting 才是 true。若只是說「要給老闆看」必須是 false。
17. next_action：必須是業務可以執行的一個具體下一步；沒有足夠資訊則 null。
18. follow_up_raw：保留逐字稿中的原始時間說法，例如「下週三」；沒有則 null。
19. follow_up_date：以 meeting_date 為基準把相對日期正規化為 YYYY-MM-DD。無法可靠推算時 null。
20. quotes：必須是逐字稿中「原句證據」，不可改寫、不可摘要、不可創造。
   - budget：支撐 budget 的原句；沒有則 null。
   - objection：支撐 objection 的原句；沒有則 null。
   - plan：支撐 plan 的原句；沒有則 null。
   - decision_maker：支撐 decision_maker 的原句；沒有則 null。
21. Speaker label 只是講者分群，不代表身份；不要因 A/B 標籤自行假設誰是客戶。
22. 不要輸出 JSON Schema 以外的欄位，也不要輸出任何 JSON 以外的文字。
""",
        input=json.dumps(transcription_payload, ensure_ascii=False, indent=2),
        text={
            "format": {
                "type": "json_schema",
                "name": "sales_meeting_record",
                "description": "Strict CRM-ready B2B sales meeting record with evidence quotes.",
                "strict": True,
                "schema": SALES_INTELLIGENCE_SCHEMA,
            }
        },
        store=False,
    )
    return json.loads(response.output_text), response_usage_record("strict sales analysis", "gpt-5-mini", response)


def render_sales_intelligence(data):
    if not isinstance(data, dict):
        st.write(data)
        return

    st.subheader("📇 CRM Meeting Record")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Stage", data.get("stage") or "未判定")
    with c2:
        budget = data.get("budget")
        st.metric("Budget", f"NT${budget:,}" if isinstance(budget, int) else "未提及")
        st.caption(f"信心：{data.get('budget_confidence', '未提及')}")
    with c3:
        st.metric("Plan", data.get("plan") or "未提及")
    with c4:
        st.metric("Follow-up", data.get("follow_up_date") or data.get("follow_up_raw") or "未提及")

    left, right = st.columns(2)
    with left:
        st.subheader("👤 Contact")
        st.write(f"**公司：** {data.get('company') or '未提及'}")
        st.write(f"**姓名：** {data.get('contact_name') or '未提及'}")
        st.write(f"**職稱：** {data.get('contact_role') or '未提及'}")
        st.write(f"**客戶類型：** {data.get('customer_type') or '未提及'}")

        st.subheader("🎯 Need")
        st.write(data.get("need") or "未提及")

        st.subheader("🗓️ Timeline")
        st.write(data.get("timeline") or "未提及")

    with right:
        st.subheader("🚧 Objection")
        st.write(data.get("objection") or "未提及")

        st.subheader("👑 Decision Maker")
        dm = data.get("decision_maker") or {}
        st.write(f"**姓名：** {dm.get('name') or '未提及'}")
        st.write(f"**角色：** {dm.get('role') or '未提及'}")
        st.write(f"**本場出席：** {'是' if dm.get('attended') else '否'}")

        st.subheader("✅ Next Action")
        st.write(data.get("next_action") or "未提及")

    st.subheader("🧾 Evidence Quotes")
    quotes = data.get("quotes") or {}
    quote_labels = {
        "budget": "💰 Budget",
        "objection": "🚧 Objection",
        "plan": "📦 Plan",
        "decision_maker": "👑 Decision Maker",
    }
    for key, label in quote_labels.items():
        quote = quotes.get(key)
        st.markdown(f"**{label}**")
        st.write(f"「{quote}」" if quote else "未提及")

def process_source(
    source_type,
    company,
    customer_name,
    salesperson,
    target_language,
    source_language_code="zh",
    source_language_label="中文（Mandarin / 中文語音）",
    file_bytes=None,
    filename=None,
    mime_type=None,
    text_input=None,
    json_input=None,
    enable_translation=True,
    force_pdf_ocr=False,
    pdf_ocr_lang="chi_tra+eng",
    allow_gpt_ocr_fallback=False,
):
    """統一處理：音訊 / PDF / TXT / JSON / 貼上逐字稿 / Demo。"""
    duration = None
    api_usage = []
    source_meta = {
        "input_type": source_type,
        "filename": filename,
        "mime_type": mime_type,
        "audio_input_language": source_language_label if source_type in {"recording", "audio_upload"} else None,
        "audio_input_language_code": source_language_code if source_type in {"recording", "audio_upload"} else None,
    }

    if source_type in {"recording", "audio_upload"}:
        full_text, segments, duration, audio_usage = transcribe_audio(
            file_bytes=file_bytes,
            filename=filename or "recording.wav",
            mime_type=mime_type or "audio/wav",
            source_language_code=source_language_code,
        )
        api_usage.append(audio_usage)
        transcription_model = "gpt-4o-transcribe-diarize"
        diarization_enabled = True

    elif source_type == "pdf_upload":
        full_text, pdf_meta, pdf_usage = read_pdf_with_ocr(
            file_bytes=file_bytes,
            filename=filename or "document.pdf",
            force_ocr=force_pdf_ocr,
            ocr_lang=pdf_ocr_lang,
            allow_gpt_fallback=allow_gpt_ocr_fallback,
        )
        source_meta.update(pdf_meta)
        api_usage.extend(pdf_usage)
        segments = parse_labeled_transcript(full_text)
        transcription_model = pdf_meta.get("pdf_text_method", "pdf_text_extraction")
        diarization_enabled = any(seg.get("speaker") != "Unknown" for seg in segments)

    elif source_type == "txt_upload":
        full_text = read_txt_text(file_bytes).strip()
        segments = parse_labeled_transcript(full_text)
        transcription_model = "text_file_input"
        diarization_enabled = False

    elif source_type == "json_upload":
        if json_input is None:
            json_input = json.loads(read_txt_text(file_bytes))
        full_text, segments = extract_segments_from_json(json_input)
        transcription_model = "json_file_input"
        diarization_enabled = any(seg.get("speaker") != "Unknown" for seg in segments)

    elif source_type in {"pasted_transcript", "demo"}:
        full_text = (text_input or "").strip()
        if not full_text:
            raise ValueError("逐字稿內容是空的。")
        segments = parse_labeled_transcript(full_text)
        transcription_model = "text_input"
        diarization_enabled = any(seg.get("speaker") != "Unknown" for seg in segments)

    else:
        raise ValueError(f"不支援的 source_type：{source_type}")

    if not full_text.strip():
        raise ValueError("沒有讀取到可分析的文字內容。")

    speaker_transcript = format_speaker_transcript(segments)
    meeting_id = "MTG-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    created_at = datetime.now().isoformat(timespec="seconds")

    transcription_payload = {
        "schema_version": "1.0",
        "meeting_id": meeting_id,
        "created_at": created_at,
        "meeting_info": {
            "company": company or None,
            "customer_name": customer_name or None,
            "salesperson": salesperson or None,
        },
        "source": source_meta,
        "transcription": {
            "model": transcription_model,
            "diarization_enabled": diarization_enabled,
            "source_language": source_language_code if source_type in {"recording", "audio_upload"} else "document_or_text",
            "source_language_label": source_language_label if source_type in {"recording", "audio_upload"} else None,
            "duration_seconds": duration,
            "full_text": full_text,
            "segments": segments,
        },
    }

    if enable_translation:
        translated_text, translation_usage = translate_speaker_transcript(speaker_transcript, target_language)
        api_usage.append(translation_usage)
    else:
        translated_text = "（未啟用翻譯）"

    sales_record, analysis_usage = analyze_sales_strict(transcription_payload)
    api_usage.append(analysis_usage)

    api_usage_estimate = {
        "items": api_usage,
        "total_estimated_cost_usd": round(sum(float(x.get("estimated_cost_usd", 0) or 0) for x in api_usage), 6),
        "note": "Text-call costs use response token usage. Duration-only audio transcription uses a conservative internal planning allowance; verify final charges in OpenAI Usage/Costs.",
    }

    # meeting_json is intentionally the FINAL CRM JSON only.
    # The raw transcription and processing metadata remain available separately in the UI/session.
    final_meeting_json = sales_record

    return {
        "company": company,
        "customer_name": customer_name,
        "salesperson": salesperson,
        "target_language": target_language,
        "transcript": full_text,
        "speaker_segments": segments,
        "speaker_transcript": speaker_transcript,
        "translation": translated_text,
        "sales_intelligence": sales_record,
        "analysis": json.dumps(sales_record, ensure_ascii=False, indent=2),
        "meeting_json": final_meeting_json,
        "transcription_json": transcription_payload,
        "source": source_meta,
        "api_usage": api_usage,
        "api_usage_estimate": api_usage_estimate,
        "estimated_api_cost_usd": api_usage_estimate["total_estimated_cost_usd"],
    }


init_database()



# =========================================================
# Knowledge Base + AI Lead Discovery
# =========================================================

LEAD_DISCOVERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_company": {"type": ["string", "null"]},
        "generated_at": {"type": "string"},
        "knowledge_profile": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "company": {"type": ["string", "null"]},
                "customer_type": {"type": ["string", "null"]},
                "current_stage": {"type": ["string", "null"]},
                "needs": {"type": "array", "items": {"type": "string"}},
                "plans": {"type": "array", "items": {"type": "string"}},
                "budget_signals": {"type": "array", "items": {"type": "string"}},
                "objections": {"type": "array", "items": {"type": "string"}},
                "decision_roles": {"type": "array", "items": {"type": "string"}},
                "lookalike_traits": {"type": "array", "items": {"type": "string"}},
                "search_keywords": {"type": "array", "items": {"type": "string"}}
            },
            "required": [
                "company", "customer_type", "current_stage", "needs", "plans",
                "budget_signals", "objections", "decision_roles",
                "lookalike_traits", "search_keywords"
            ]
        },
        "search_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "purpose": {
                        "type": "string",
                        "enum": ["lookalike", "hiring_signal", "growth_signal"]
                    }
                },
                "required": ["query", "purpose"]
            }
        },
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "company_name": {"type": "string"},
                    "website": {"type": ["string", "null"]},
                    "industry": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                    "fit_score": {"type": "integer"},
                    "confidence": {
                        "type": "string",
                        "enum": ["高", "中", "低"]
                    },
                    "why_match": {"type": "string"},
                    "signals": {"type": "array", "items": {"type": "string"}},
                    "suggested_contact_role": {"type": ["string", "null"]},
                    "recommended_next_action": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["title", "url", "reason"]
                        }
                    }
                },
                "required": [
                    "company_name", "website", "industry", "location",
                    "fit_score", "confidence", "why_match", "signals",
                    "suggested_contact_role", "recommended_next_action",
                    "evidence"
                ]
            }
        }
    },
    "required": [
        "source_company", "generated_at", "knowledge_profile",
        "search_queries", "leads"
    ]
}


KNOWLEDGE_PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "company": {"type": ["string", "null"]},
        "customer_type": {"type": ["string", "null"]},
        "current_stage": {"type": ["string", "null"]},
        "needs": {"type": "array", "items": {"type": "string"}},
        "plans": {"type": "array", "items": {"type": "string"}},
        "budget_signals": {"type": "array", "items": {"type": "string"}},
        "objections": {"type": "array", "items": {"type": "string"}},
        "decision_roles": {"type": "array", "items": {"type": "string"}},
        "lookalike_traits": {"type": "array", "items": {"type": "string"}},
        "search_keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "company", "customer_type", "current_stage", "needs", "plans",
        "budget_signals", "objections", "decision_roles",
        "lookalike_traits", "search_keywords"
    ]
}


SEARCH_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "purpose": {
                        "type": "string",
                        "enum": ["lookalike", "hiring_signal", "growth_signal"]
                    }
                },
                "required": ["query", "purpose"]
            }
        }
    },
    "required": ["queries"]
}


def get_company_knowledge(company_name):
    """Return knowledge records from BOTH local SQLite and AWS S3."""
    all_records = get_all_knowledge_records(
        include_local=True,
        include_s3=True
    )
    return get_company_knowledge_from_records(
        company_name,
        all_records
    )


def summarize_company_knowledge(company_name, records):
    response = client.responses.create(
        model="gpt-5-mini",
        instructions="""
你是 B2B Sales Knowledge Analyst。
請把同一家公司歷次 Meeting CRM JSON 統整成可用於尋找相似潛在客戶的客戶輪廓。

規則：
1. 只能根據提供資料整理，不可補造。
2. lookalike_traits 應描述「什麼樣的其他公司可能有相似需求」，不要包含私人個資。
3. search_keywords 應適合搜尋公開公司網站、新聞、產業資訊與公開職缺訊號。
4. 不要尋找、推斷或輸出私人電話、私人 Email、住址等個人資料。
5. 使用繁體中文。
""",
        input=json.dumps({
            "company": company_name,
            "meeting_records": records
        }, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "company_knowledge_profile",
                "schema": KNOWLEDGE_PROFILE_SCHEMA,
                "strict": True
            }
        },
        store=False
    )
    return json.loads(response.output_text)


def generate_lead_search_plan(knowledge_profile, geography="台灣"):
    response = client.responses.create(
        model="gpt-5-mini",
        instructions=f"""
你是 B2B Lead Research Planner。
根據既有客戶 Knowledge Profile，產生 3 個互補的公開網路搜尋查詢：

1. lookalike：尋找產業、商業模式、需求情境相似的公司。
2. hiring_signal：利用公開職缺作為成長、擴編、數位轉型、海外拓展或特定能力需求的公司級訊號。
3. growth_signal：利用公司官網、新聞、政府／協會／產業公開資訊尋找擴產、投資、新市場、新產品等訊號。

地理範圍：{geography}

重要：
- 目標是「公司級潛在客戶」，不是求職者或個人名單。
- 不得要求私人聯絡資訊或履歷資料。
- 不要把 LinkedIn 自動爬取設計成搜尋策略。
- 查詢要短、可直接交給 Web Search。
""",
        input=json.dumps(knowledge_profile, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "lead_search_plan",
                "schema": SEARCH_PLAN_SCHEMA,
                "strict": True
            }
        },
        store=False
    )
    payload = json.loads(response.output_text)
    # Keep bounded search cost and latency.
    return payload.get("queries", [])[:3]


def extract_web_citations(response):
    """Extract title/url citations from a Responses API web_search response."""
    try:
        data = response.model_dump()
    except Exception:
        return []

    seen = set()
    citations = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            for ann in content.get("annotations", []) or []:
                if ann.get("type") != "url_citation":
                    continue
                url = (ann.get("url") or "").strip()
                title = (ann.get("title") or url).strip()
                # LinkedIn automated scraping/access is intentionally not used.
                if "linkedin.com" in url.lower():
                    continue
                if not url or url in seen:
                    continue
                seen.add(url)
                citations.append({
                    "title": title,
                    "url": url
                })
    return citations


def run_public_lead_search(query, purpose, geography="台灣"):
    source_guidance = {
        "lookalike": (
            "優先尋找公司官網、產品頁、公開新聞、產業協會、政府／公開企業資料。"
        ),
        "hiring_signal": (
            "把公開職缺當作『公司級商業訊號』，例如擴編、海外業務、數位轉型、"
            "新廠、新產品、CRM／行銷／供應鏈能力需求。可參考公開人力銀行職缺頁，"
            "但不要取得履歷、求職者個資或會員限定內容。"
        ),
        "growth_signal": (
            "優先尋找擴廠、投資、新市場、海外拓展、新產品、策略合作、招募成長等公開訊號。"
        ),
    }.get(purpose, "")

    response = client.responses.create(
        model="gpt-5-mini",
        tools=[
            {
                "type": "web_search",
                "search_context_size": "medium"
            }
        ],
        instructions="""
你是 B2B Public-Web Lead Research Agent。

只研究公開可存取、公司級的商業資訊。
不得蒐集或輸出私人電話、私人 Email、住址、履歷內容或其他非必要個人資料。
不要自動爬取 LinkedIn，也不要繞過登入、robots、CAPTCHA、rate limit 或其他存取控制。
若資料來自公開職缺，只把它當成「公司成長／需求訊號」，不要分析求職者。
每個候選公司都要有可驗證的公開來源。
""",
        input=f"""
地理範圍：{geography}
搜尋目的：{purpose}
來源偏好：{source_guidance}

搜尋查詢：
{query}

請找出可能值得 B2B 業務開發的『公司』，並用繁體中文簡短說明：
- 公司名稱
- 公開訊號
- 為何值得進一步研究
- 來源依據

不要列個人聯絡資料。
""",
        store=False
    )

    return {
        "query": query,
        "purpose": purpose,
        "text": response.output_text,
        "citations": extract_web_citations(response)
    }


def normalize_lead_discovery(source_company, knowledge_profile, search_plan, search_runs, max_leads=8):
    # Flatten citations so the strict-output pass has actual URLs available.
    source_catalog = []
    seen = set()
    for run in search_runs:
        for c in run.get("citations", []):
            if c["url"] in seen:
                continue
            seen.add(c["url"])
            source_catalog.append(c)

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=f"""
你是 B2B Lead Qualification Analyst。

根據：
1. 既有客戶 Knowledge Profile
2. 公開 Web Search 的研究摘要
3. 實際 source catalog URL

選出最多 {max_leads} 家『公司級』潛在新客戶。

評分 fit_score 0-100：
- 與既有客戶需求／產業情境相似度
- 是否存在具體公開成長、招聘、擴產、轉型或市場訊號
- 是否有合理的 B2B 切入點
- 證據品質

規則：
- 不要建立或推測個人私人聯絡資料。
- suggested_contact_role 只能是職務類型，例如「採購經理」「業務營運主管」「行銷主管」。
- evidence.url 必須只能使用 source catalog 中提供的 URL，不可自行捏造 URL。
- 若證據不足就降低 confidence / fit_score，不可硬湊。
- 排除 LinkedIn URL。
- 使用繁體中文。
""",
        input=json.dumps({
            "source_company": source_company,
            "knowledge_profile": knowledge_profile,
            "search_plan": search_plan,
            "search_runs": search_runs,
            "source_catalog": source_catalog
        }, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "lead_discovery_result",
                "schema": LEAD_DISCOVERY_SCHEMA,
                "strict": True
            }
        },
        store=False
    )
    return json.loads(response.output_text)


def save_lead_discovery_local(source_company, payload):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO lead_discoveries (
            created_at,
            source_company,
            result_json
        ) VALUES (?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            source_company,
            json.dumps(payload, ensure_ascii=False),
        )
    )
    conn.commit()
    conn.close()


def get_recent_lead_discoveries(limit=10):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, created_at, source_company, result_json
        FROM lead_discoveries
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_latest_lead_discovery_for_company(source_company):
    """Restore the latest lead-discovery conclusion.

    Priority:
    1. Local SQLite for the current app runtime.
    2. AWS S3 for durable recovery after refresh/redeploy/restart.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT result_json
        FROM lead_discoveries
        WHERE source_company = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (source_company,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        local_payload = safe_json_loads(row[0])
        if local_payload:
            return local_payload

    if not AWS_S3_BUCKET:
        return None

    safe_company = re.sub(
        r"[^0-9A-Za-z\u4e00-\u9fff_-]+",
        "-",
        source_company or "unknown-company"
    ).strip("-")

    prefix = f"{AWS_S3_PREFIX}/lead-discovery/"
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    candidates = []
    for page_data in paginator.paginate(
        Bucket=AWS_S3_BUCKET,
        Prefix=prefix
    ):
        for item in page_data.get("Contents", []):
            key = item.get("Key", "")
            basename = key.rsplit("/", 1)[-1]

            if (
                key.lower().endswith(".json")
                and basename.startswith(f"{safe_company}-")
            ):
                candidates.append(item)

    if not candidates:
        return None

    latest = max(
        candidates,
        key=lambda item: item.get("LastModified")
    )
    obj = s3.get_object(
        Bucket=AWS_S3_BUCKET,
        Key=latest["Key"]
    )
    return json.loads(
        obj["Body"].read().decode("utf-8")
    )


def backup_lead_discovery_to_s3(source_company, payload):
    if not AWS_S3_BUCKET:
        return None
    safe_company = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", source_company or "unknown-company").strip("-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    date_part = datetime.now().strftime("%Y-%m-%d")
    key = f"{AWS_S3_PREFIX}/lead-discovery/{date_part}/{safe_company}-{timestamp}.json"
    return upload_json_to_s3(payload, key)


# =========================================================
# Session State
# =========================================================

# Streamlit reruns the script after every widget interaction and page switch.
# Keep durable UI conclusions in Session State instead of local variables.
_STATE_DEFAULTS = {
    "active_page": "🎙️ 新增會議",
    "meeting_result": None,
    "ask_sales_question": "",
    "ask_sales_answer": None,
    "knowledge_profiles": {},
    "lead_results": {},
    "lead_search_plans": {},
    "lead_search_runs_by_company": {},
    "kb_selected_company": None,
}

for _key, _default in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        if isinstance(_default, dict):
            st.session_state[_key] = {}
        else:
            st.session_state[_key] = _default


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("🧠 AI Sales Assistant")
page = st.sidebar.radio(
    "功能",
    [
        "🎙️ 新增會議",
        "📚 Meeting History",
        "📊 Sales Dashboard",
        "💬 Ask Sales AI",
        "🧠 Knowledge Base",
        "☁️ AWS Integration",
    ],
    key="active_page",
)


# =========================================================
# PAGE 1 - 新增會議
# =========================================================

if page == "🎙️ 新增會議":
    st.title("🎙️ AI Sales Meeting Assistant")
    st.write(
        "支援示範會議、錄音、音檔／PDF／TXT／JSON 上傳與貼上逐字稿。"
        "音訊會做多人講者辨識；掃描型 PDF 會自動 OCR；最後輸出 Strict JSON Schema 的 Sales Intelligence。"
    )

    # 最前面先選擇語音輸入語言；只影響瀏覽器錄音與上傳音檔。
    st.subheader("🗣️ Step 0：選擇語音輸入語言")
    source_language_label = st.radio(
        "這場會議主要使用哪一種語言？",
        list(AUDIO_INPUT_LANGUAGE_OPTIONS.keys()),
        horizontal=True,
        help="請選擇會議主要語言。此設定會傳給語音轉錄模型；多人講者辨識仍會保留 Speaker A / B / C。",
    )
    source_language_code = AUDIO_INPUT_LANGUAGE_OPTIONS[source_language_label]
    st.caption(f"目前語音辨識語言：{source_language_label}（{source_language_code}）")

    st.divider()
    st.subheader("👤 Meeting Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        company = st.text_input("🏢 公司名稱", placeholder="例如：ABC科技")
    with c2:
        customer_name = st.text_input("👤 客戶姓名", placeholder="例如：王經理")
    with c3:
        salesperson = st.text_input("💼 業務人員", placeholder="例如：紀承峰")

    target_language = st.selectbox(
        "🌎 翻譯語言",
        ["English", "繁體中文", "日本語", "한국어", "ไทย", "Français", "Deutsch", "Español"],
    )
    enable_translation = st.checkbox(
        "產生翻譯版本（會多一次 GPT-5 mini 呼叫；不需要翻譯時可關閉省額度）",
        value=True,
    )

    st.divider()
    st.subheader("📥 選擇會議輸入方式")

    # 視覺功能卡
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        with st.container(border=True):
            st.markdown("### ▶️ 載入示範會議")
            st.caption("載入內建 B2B Sales 對話，快速測試完整 JSON 分析流程。")
    with r1c2:
        with st.container(border=True):
            st.markdown("### 🎙️ 即時錄音")
            st.caption("用瀏覽器麥克風錄完整場會議，結束後再做多人轉錄與分析。")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        with st.container(border=True):
            st.markdown("### 📁 上傳檔案")
            st.caption("音訊：mp3 / m4a / wav / webm 等；文件：PDF / TXT / JSON。PDF 支援文字層與掃描 OCR。")
    with r2c2:
        with st.container(border=True):
            st.markdown("### 📋 貼上逐字稿")
            st.caption("可使用「Speaker A: ...」或「張經理：...」格式；沒有講者標記也能分析。")

    source_mode = st.radio(
        "輸入方式",
        ["載入示範會議", "即時錄音", "上傳檔案", "貼上逐字稿"],
        horizontal=True,
    )

    try:
        if source_mode == "載入示範會議":
            st.code(DEMO_TRANSCRIPT, language=None)
            if st.button("▶️ 載入並分析示範會議", type="primary", use_container_width=True):
                with st.spinner("🧠 正在分析示範會議並建立 Strict JSON..."):
                    st.session_state.meeting_result = process_source(
                        source_type="demo",
                        company=company or "ABC科技",
                        customer_name=customer_name or "王經理",
                        salesperson=salesperson or "紀承峰",
                        target_language=target_language,
                        source_language_code=source_language_code,
                        source_language_label=source_language_label,
                        enable_translation=enable_translation,
                        text_input=DEMO_TRANSCRIPT,
                    )
                st.success("✅ 示範會議分析完成！")

        elif source_mode == "即時錄音":
            audio = st.audio_input("🎤 按這裡開始錄音；結束後再一次送出分析", sample_rate=16000)
            if audio:
                st.audio(audio)
                if st.button("✨ 轉錄並分析錄音", type="primary", use_container_width=True):
                    with st.spinner("🎧 正在做多人講者辨識 → 翻譯 → Strict JSON 分析..."):
                        st.session_state.meeting_result = process_source(
                            source_type="recording",
                            company=company,
                            customer_name=customer_name,
                            salesperson=salesperson,
                            target_language=target_language,
                            source_language_code=source_language_code,
                            source_language_label=source_language_label,
                            enable_translation=enable_translation,
                            file_bytes=audio.getvalue(),
                            filename="recording.wav",
                            mime_type="audio/wav",
                        )
                    st.success("✅ 錄音轉錄與分析完成！")

        elif source_mode == "上傳檔案":
            uploaded = st.file_uploader(
                "上傳音訊、PDF、TXT 或 JSON",
                type=AUDIO_TYPES + DOCUMENT_TYPES,
                help="音訊會做 Speaker diarization；PDF 會先讀文字層，若偵測為掃描型則自動 Tesseract OCR；TXT/JSON 直接讀取。",
            )

            if uploaded:
                suffix = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
                file_bytes = uploaded.getvalue()
                st.caption(f"已選擇：{uploaded.name} ｜ {len(file_bytes) / 1024:.1f} KB")

                force_pdf_ocr = False
                pdf_ocr_lang = "chi_tra+eng"
                allow_gpt_ocr_fallback = False
                if suffix == "pdf":
                    st.info("PDF 會先嘗試讀取內建文字層；若判定為掃描型 PDF，會自動使用本機 Tesseract OCR（不消耗 OpenAI API）。")
                    ocr_label = st.selectbox("🔎 OCR 語言", list(OCR_LANGUAGE_OPTIONS.keys()))
                    pdf_ocr_lang = OCR_LANGUAGE_OPTIONS[ocr_label]
                    force_pdf_ocr = st.checkbox("強制 OCR（即使 PDF 已有文字層也重新掃描）", value=False)
                    allow_gpt_ocr_fallback = st.checkbox(
                        "本機 OCR 失敗時允許 GPT-5 mini OCR 備援（會消耗 API 額度）",
                        value=False,
                    )

                if suffix in AUDIO_TYPES:
                    st.audio(file_bytes)
                    action_label = "✨ 轉錄音檔並分析"
                    source_type = "audio_upload"
                elif suffix == "pdf":
                    action_label = "📄 讀取 PDF 並分析"
                    source_type = "pdf_upload"
                elif suffix == "txt":
                    action_label = "📝 讀取 TXT 並分析"
                    source_type = "txt_upload"
                elif suffix == "json":
                    action_label = "🧩 讀取 JSON 並分析"
                    source_type = "json_upload"
                else:
                    action_label = "分析"
                    source_type = None

                if source_type and st.button(action_label, type="primary", use_container_width=True):
                    with st.spinner("🧠 正在讀取／轉錄 → 翻譯 → Strict JSON 分析..."):
                        json_input = None
                        if suffix == "json":
                            json_input = json.loads(read_txt_text(file_bytes))
                        st.session_state.meeting_result = process_source(
                            source_type=source_type,
                            company=company,
                            customer_name=customer_name,
                            salesperson=salesperson,
                            target_language=target_language,
                            source_language_code=source_language_code,
                            source_language_label=source_language_label,
                            enable_translation=enable_translation,
                            file_bytes=file_bytes,
                            filename=uploaded.name,
                            mime_type=uploaded.type,
                            json_input=json_input,
                            force_pdf_ocr=force_pdf_ocr,
                            pdf_ocr_lang=pdf_ocr_lang,
                            allow_gpt_ocr_fallback=allow_gpt_ocr_fallback,
                        )
                    st.success("✅ 檔案讀取／轉錄與分析完成！")

        elif source_mode == "貼上逐字稿":
            pasted = st.text_area(
                "貼上逐字稿",
                height=220,
                placeholder="例如：\n張經理：我們的預算大概 120 萬。\n業務：了解，決策者會是您嗎？",
            )
            if st.button("📋 分析逐字稿", type="primary", use_container_width=True):
                if not pasted.strip():
                    st.warning("請先貼上逐字稿。")
                else:
                    with st.spinner("🧠 正在解析講者 → 翻譯 → Strict JSON 分析..."):
                        st.session_state.meeting_result = process_source(
                            source_type="pasted_transcript",
                            company=company,
                            customer_name=customer_name,
                            salesperson=salesperson,
                            target_language=target_language,
                            enable_translation=enable_translation,
                            text_input=pasted,
                        )
                    st.success("✅ 逐字稿分析完成！")

    except Exception as e:
        st.error("❌ 處理失敗")
        st.exception(e)

    result = st.session_state.meeting_result

    if result:
        st.divider()

        source_info = result.get("source", {})
        if source_info.get("input_type") == "pdf_upload":
            method = source_info.get("pdf_text_method", "unknown")
            if method == "tesseract_ocr":
                st.success(f"🔎 PDF OCR 完成：Tesseract / {source_info.get('ocr_language', '')} / {source_info.get('pdf_pages', '?')} 頁（OCR 本身不消耗 OpenAI API）")
            elif method == "native_text_layer":
                st.info(f"📄 PDF 使用內建文字層：{source_info.get('pdf_pages', '?')} 頁，不需要 OCR。")
            elif method == "gpt_ocr_fallback":
                st.warning("⚠️ 本機 OCR 失敗，本次使用 GPT-5 mini OCR 備援，因此 OCR 也會產生 API 成本。")

        est_cost = float(result.get("estimated_api_cost_usd", 0) or 0)
        st.metric("💵 本次 API 估計成本", f"US${est_cost:.4f}")
        with st.expander("查看本次 API 用量估計"): 
            st.json(result.get("api_usage_estimate", {}))

        st.header("📝 Meeting Transcript")

        left, right = st.columns(2)
        with left:
            st.subheader("🗣️ 講者逐字稿")
            for segment in result["speaker_segments"]:
                speaker = segment.get("speaker", "Unknown")
                start = segment.get("start")
                end = segment.get("end")
                if start is not None and end is not None:
                    st.markdown(f"**{speaker}** `{float(start):.2f}s - {float(end):.2f}s`")
                else:
                    st.markdown(f"**{speaker}**")
                st.write(segment.get("text", ""))

        with right:
            st.subheader(f"🌎 翻譯：{result['target_language']}")
            st.text_area(
                "translated_text",
                result["translation"],
                height=420,
                label_visibility="collapsed",
            )

        st.divider()
        st.header("🧠 Final Strict CRM JSON")
        render_sales_intelligence(result["sales_intelligence"])

        with st.expander("🔧 查看最終 CRM JSON"):
            st.json(result["meeting_json"])

        with st.expander("🗣️ 查看轉錄 JSON（Speaker / timestamps / source）"):
            st.json(result.get("transcription_json", {}))

        json_download = json.dumps(result["meeting_json"], ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ 下載最終 CRM JSON",
            data=json_download,
            file_name=f"{result['meeting_json']['meeting_id']}.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.button("💾 儲存這場 Meeting", type="primary", use_container_width=True):
            save_meeting(result)
            st.success("✅ Meeting 與 Strict JSON 已儲存到本機資料庫！")

            if AWS_S3_BUCKET:
                try:
                    s3_uri = backup_meeting_to_s3(result)
                    st.success(f"☁️ AWS S3 備份完成：{s3_uri}")
                except Exception as aws_error:
                    st.warning(f"⚠️ 本機已儲存，但 AWS S3 備份失敗：{aws_error}")
            else:
                st.info("ℹ️ 尚未設定 AWS_S3_BUCKET，因此本次只存到 SQLite。")


# =========================================================
# Knowledge Base + Lead Discovery
# =========================================================

elif page == "🧠 Knowledge Base":
    st.title("🧠 Customer Knowledge Base")
    st.write(
        "先把既有客戶歷次 Meeting JSON 統整成客戶知識輪廓，再使用公開 Web Search 找出具相似需求或成長訊號的潛在新客戶。"
    )

    st.subheader("📦 Knowledge Sources")

    source_col1, source_col2 = st.columns(2)

    with source_col1:
        use_local_kb = st.checkbox(
            "讀取 Streamlit / SQLite",
            value=True,
            help="讀取目前 App 本機 SQLite 的 Meeting records。"
        )

    with source_col2:
        use_s3_kb = st.checkbox(
            "讀取 AWS S3",
            value=True,
            help=f"讀取 s3://{AWS_S3_BUCKET or '(未設定)'}/{AWS_S3_PREFIX}/meetings/"
        )

    if st.button("🔄 重新同步 Knowledge Base", use_container_width=True):
        # Clear the cached S3 snapshot, then fetch it again.
        list_all_s3_meeting_json.clear()
        st.rerun()

    try:
        knowledge_records = get_all_knowledge_records(
            include_local=use_local_kb,
            include_s3=use_s3_kb
        )
    except Exception as e:
        knowledge_records = get_all_knowledge_records(
            include_local=use_local_kb,
            include_s3=False
        )
        st.error(f"AWS S3 Knowledge Base 讀取失敗：{e}")

    local_count = sum(1 for x in knowledge_records if x.get("source") == "local")
    s3_count = sum(1 for x in knowledge_records if x.get("source") == "s3")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Local records", local_count)
    metric2.metric("AWS S3 records", s3_count)
    metric3.metric("Total records", len(knowledge_records))

    if use_s3_kb:
        st.caption(
            f"AWS path：s3://{AWS_S3_BUCKET or '(未設定)'}/{AWS_S3_PREFIX}/meetings/"
        )

    companies = list_knowledge_companies(knowledge_records)

    if not companies:
        st.warning(
            "目前 Knowledge Base 沒讀到任何有 company 欄位的 Meeting JSON。"
            "請先確認 AWS Integration 能列出 sync-pipeline/meetings/ 下的 JSON。"
        )
    else:
        saved_company = st.session_state.get("kb_selected_company")
        selected_index = (
            companies.index(saved_company)
            if saved_company in companies
            else 0
        )

        selected_company = st.selectbox(
            "選擇要作為 Look-alike 基準的既有客戶",
            companies,
            index=selected_index,
            key="_kb_company_widget"
        )
        st.session_state["kb_selected_company"] = selected_company

        records = get_company_knowledge_from_records(
            selected_company,
            knowledge_records
        )

        local_selected = sum(
            1 for r in records
            if r.get("_knowledge_source") == "local"
        )
        s3_selected = sum(
            1 for r in records
            if r.get("_knowledge_source") == "s3"
        )

        st.caption(
            f"Knowledge Base records：{len(records)} "
            f"（Local {local_selected} / AWS S3 {s3_selected}）"
        )

        with st.expander("查看原始 Knowledge Base JSON"):
            st.json(records)

        if st.button("🧠 AI 統整這個客戶的知識庫", use_container_width=True):
            if not records:
                st.warning("找不到這家公司的有效 Meeting JSON。")
            else:
                with st.spinner("正在統整需求、方案、預算訊號、阻礙與 Look-alike 特徵..."):
                    st.session_state["knowledge_profiles"][selected_company] = (
                        summarize_company_knowledge(
                            selected_company,
                            records
                        )
                    )

        knowledge_profile = st.session_state["knowledge_profiles"].get(
            selected_company
        )

        if knowledge_profile:
            st.subheader("Customer Knowledge Profile")
            st.json(knowledge_profile)

            st.divider()
            st.header("🔎 AI Potential Customer Discovery")
            st.write(
                "系統會把 Knowledge Profile 轉成搜尋策略，研究公開公司網站、新聞、產業資訊與公開職缺訊號，"
                "最後產生有來源證據的潛在公司清單。"
            )

            st.info(
                "LinkedIn：此版本不做自動爬取。若未來取得 LinkedIn 正式授權／合作 API，"
                "可以再用 Source Adapter 接入；現在以公開 Web Search 與合法可存取來源為主。"
            )

            c1, c2 = st.columns(2)
            with c1:
                geography = st.text_input("目標市場", value="台灣")
            with c2:
                max_leads = st.slider("最多輸出幾家潛在客戶", 3, 12, 6)

            st.markdown(
                """
                **目前搜尋訊號**
                - Look-alike 公司：產業／商業模式／需求情境相似
                - 公開職缺：擴編、海外業務、數位轉型、新能力需求
                - Growth signals：擴廠、投資、新產品、新市場、策略合作
                """
            )

            if st.button(
                "🚀 從知識庫尋找潛在新客戶",
                type="primary",
                use_container_width=True
            ):
                try:
                    with st.spinner("Step 1/3：AI 正在建立 Lead Search Strategy..."):
                        search_plan = generate_lead_search_plan(
                            knowledge_profile,
                            geography=geography
                        )

                    st.session_state["lead_search_plans"][selected_company] = search_plan

                    search_runs = []
                    progress = st.progress(0)
                    status = st.empty()

                    for idx, q in enumerate(search_plan):
                        status.write(
                            f"Step 2/3：搜尋公開網路 ({idx + 1}/{len(search_plan)}) — {q['purpose']}"
                        )
                        run = run_public_lead_search(
                            q["query"],
                            q["purpose"],
                            geography=geography
                        )
                        search_runs.append(run)
                        progress.progress(int(((idx + 1) / max(len(search_plan), 1)) * 100))

                    status.write("Step 3/3：AI 正在去重、評分並產生潛在客戶 JSON...")
                    result = normalize_lead_discovery(
                        selected_company,
                        knowledge_profile,
                        search_plan,
                        search_runs,
                        max_leads=max_leads
                    )

                    st.session_state["lead_results"][selected_company] = result
                    st.session_state["lead_search_runs_by_company"][selected_company] = search_runs

                    save_lead_discovery_local(selected_company, result)

                    s3_uri = None
                    if AWS_S3_BUCKET:
                        try:
                            s3_uri = backup_lead_discovery_to_s3(selected_company, result)
                        except Exception as e:
                            st.warning(f"Lead JSON 已存本機 DB，但 S3 備份失敗：{e}")

                    status.empty()
                    progress.empty()
                    st.success("✅ Potential Customer Discovery 完成")
                    if s3_uri:
                        st.success(f"☁️ 已備份至：{s3_uri}")

                except Exception as e:
                    st.error("❌ Lead Discovery 失敗")
                    st.exception(e)

            result = st.session_state["lead_results"].get(selected_company)

            # Restore the latest saved conclusion after a browser refresh or app restart.
            if result is None:
                restored_result = get_latest_lead_discovery_for_company(
                    selected_company
                )
                if restored_result:
                    st.session_state["lead_results"][selected_company] = restored_result
                    result = restored_result

            if result:
                st.divider()
                st.subheader("🎯 Potential Leads")

                leads = sorted(
                    result.get("leads", []),
                    key=lambda x: x.get("fit_score", 0),
                    reverse=True
                )

                for i, lead in enumerate(leads, start=1):
                    with st.container(border=True):
                        left, right = st.columns([4, 1])
                        with left:
                            st.markdown(f"### {i}. {lead['company_name']}")
                            meta = " ｜ ".join(
                                x for x in [
                                    lead.get("industry"),
                                    lead.get("location"),
                                    lead.get("suggested_contact_role")
                                ] if x
                            )
                            if meta:
                                st.caption(meta)

                            st.write(lead.get("why_match", ""))

                            if lead.get("signals"):
                                st.markdown("**Signals**")
                                for signal in lead["signals"]:
                                    st.write(f"- {signal}")

                            st.markdown("**Recommended Next Action**")
                            st.write(lead.get("recommended_next_action", ""))

                            if lead.get("evidence"):
                                with st.expander("Evidence / Sources"):
                                    for ev in lead["evidence"]:
                                        st.markdown(
                                            f"- [{ev['title']}]({ev['url']}) — {ev['reason']}"
                                        )

                        with right:
                            st.metric("Fit Score", lead.get("fit_score", 0))
                            st.caption(f"Confidence：{lead.get('confidence', '低')}")

                st.divider()
                with st.expander("查看完整 Lead Discovery JSON"):
                    st.json(result)

                st.download_button(
                    "⬇️ 下載 Potential Leads JSON",
                    data=json.dumps(result, ensure_ascii=False, indent=2),
                    file_name=f"{selected_company}_lead_discovery.json",
                    mime="application/json",
                    use_container_width=True,
                    on_click="ignore"
                )

            recent = get_recent_lead_discoveries(5)
            if recent:
                st.divider()
                st.subheader("Recent Lead Discovery Runs")
                rows = []
                for item in recent:
                    rows.append({
                        "ID": item[0],
                        "Created At": item[1],
                        "Source Company": item[2]
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# =========================================================
# AWS Integration
# =========================================================

elif page == "☁️ AWS Integration":
    st.title("☁️ AWS Integration")
    st.write(
        "這一頁用來確認 Streamlit App 是否真的連到 AWS。"
        "目前整合：Amazon S3（保存 Meeting JSON）與 Amazon Bedrock（模型連線測試）。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("AWS Region", AWS_REGION or "未設定")
    with c2:
        st.metric("S3 Bucket", AWS_S3_BUCKET or "未設定")
        st.caption(f"S3 Prefix：{AWS_S3_PREFIX}/")
    with c3:
        st.metric("Bedrock Model", AWS_BEDROCK_MODEL_ID or "未設定")

    st.caption(
        "若使用 Hackathon 的 Temporary Credentials，Secrets 必須同時包含 "
        "AWS_ACCESS_KEY_ID、AWS_SECRET_ACCESS_KEY、AWS_SESSION_TOKEN 與主辦指定 AWS_REGION。"
    )

    st.divider()
    st.subheader("1️⃣ 測試 AWS 身分")
    if st.button("🔐 Test AWS Connection", use_container_width=True):
        try:
            identity = get_aws_identity()
            st.success("✅ AWS 認證成功")
            st.json({
                "Account": identity.get("Account"),
                "Arn": identity.get("Arn"),
                "UserId": identity.get("UserId"),
                "Region": AWS_REGION,
            })
        except Exception as e:
            st.error("❌ AWS 認證失敗")
            st.exception(e)

    st.divider()
    st.subheader("2️⃣ Amazon S3")
    if not AWS_S3_BUCKET:
        st.warning("尚未設定 AWS_S3_BUCKET。設定後，每次按『儲存這場 Meeting』都會自動備份 Strict Meeting JSON 到 S3。")
    else:
        if st.button("🪣 Test S3 Bucket", use_container_width=True):
            try:
                get_s3_client().head_bucket(Bucket=AWS_S3_BUCKET)
                st.success(f"✅ 可以存取 S3：s3://{AWS_S3_BUCKET}")
            except Exception as e:
                st.error("❌ S3 存取失敗")
                st.exception(e)

        try:
            recent = list_recent_s3_meetings(20)
            if recent:
                st.markdown("#### 最近的 AWS Meeting JSON")
                rows = []
                for item in recent:
                    rows.append({
                        "Key": item.get("Key"),
                        "Size (KB)": round((item.get("Size", 0) or 0) / 1024, 2),
                        "Last Modified": str(item.get("LastModified", "")),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception:
            pass

    st.divider()
    st.subheader("3️⃣ Amazon Bedrock")
    if not AWS_BEDROCK_MODEL_ID:
        st.info(
            "Bedrock 是選配。請先在 AWS 確認可用模型，再把模型 ID 放進 Streamlit Secret："
            "AWS_BEDROCK_MODEL_ID。核心語音轉錄與 Strict JSON 目前仍由 OpenAI 執行，避免今天臨時改壞。"
        )
    else:
        bedrock_test_prompt = st.text_input(
            "Bedrock 測試問題", value="請用一句繁體中文說明 Sales Intelligence 的用途。"
        )
        if st.button("🧠 Test Amazon Bedrock", use_container_width=True):
            try:
                with st.spinner("正在呼叫 Amazon Bedrock..."):
                    answer, raw = bedrock_converse(
                        bedrock_test_prompt,
                        system_text="You are a concise B2B sales assistant. Reply in Traditional Chinese.",
                    )
                st.success("✅ Amazon Bedrock 呼叫成功")
                st.write(answer)
                with st.expander("查看 Bedrock metadata"):
                    st.json({
                        "model_id": AWS_BEDROCK_MODEL_ID,
                        "region": AWS_REGION,
                        "usage": raw.get("usage", {}),
                        "metrics": raw.get("metrics", {}),
                    })
            except Exception as e:
                st.error("❌ Bedrock 呼叫失敗")
                st.exception(e)


# =========================================================
# PAGE 2 - Meeting History
# =========================================================

elif page == "📚 Meeting History":
    st.title("📚 Sales Meeting History")
    st.write("所有客戶會議、逐字稿、翻譯與 Strict JSON Sales Intelligence。")

    meetings = get_meetings()
    if not meetings:
        st.info("目前還沒有儲存任何 Meeting。")
    else:
        st.metric("Total Meetings", len(meetings))
        st.divider()

        for meeting in meetings:
            (
                meeting_id,
                created_at,
                company,
                customer_name,
                salesperson,
                language,
                transcript,
                translation,
                analysis,
                meeting_json_text,
            ) = meeting

            title = f"🏢 {company or '未命名公司'} ｜ 👤 {customer_name or '未命名客戶'} ｜ {created_at}"
            with st.expander(title):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("**公司**")
                    st.write(company or "未填寫")
                with c2:
                    st.write("**客戶**")
                    st.write(customer_name or "未填寫")
                with c3:
                    st.write("**業務人員**")
                    st.write(salesperson or "未填寫")

                saved_json = safe_json_loads(meeting_json_text)
                st.subheader("📝 Transcript")
                st.write(transcript or "未保存逐字稿")

                if isinstance(saved_json, dict):
                    # New records store the final CRM JSON directly.
                    # Legacy records may still wrap it inside sales_intelligence.
                    if "sales_intelligence" in saved_json:
                        intelligence = saved_json.get("sales_intelligence", {})
                    else:
                        intelligence = saved_json
                    st.subheader("🧠 Final CRM Record")
                    render_sales_intelligence(intelligence)
                    with st.expander("🔧 查看已儲存 Final JSON"):
                        st.json(intelligence)
                else:
                    st.subheader("🧠 舊版 Analysis")
                    parsed_analysis = safe_json_loads(analysis)
                    if isinstance(parsed_analysis, dict):
                        render_sales_intelligence(parsed_analysis)
                    else:
                        st.write(analysis)


# =========================================================
# PAGE 3 - Sales Dashboard
# =========================================================

elif page == "📊 Sales Dashboard":
    st.title("📊 Sales Dashboard")
    st.write("將所有 Sales Meetings 轉換成主管可以快速掌握的 Pipeline。")

    meetings = get_meetings()
    if not meetings:
        st.info("目前沒有 Meeting 資料，請先新增並儲存至少一場會議。")
    else:
        dashboard_data = []
        for meeting in meetings:
            (
                meeting_id,
                created_at,
                company,
                customer_name,
                salesperson,
                language,
                transcript,
                translation,
                analysis,
                meeting_json_text,
            ) = meeting

            saved_json = safe_json_loads(meeting_json_text)
            if isinstance(saved_json, dict):
                record = saved_json.get("sales_intelligence", saved_json)
                score = stage_to_opportunity_score(record.get("stage"))
                if score == "UNKNOWN":
                    score = extract_opportunity_score(record)
            else:
                score = extract_opportunity_score(analysis)

            dashboard_data.append(
                {
                    "Meeting ID": meeting_id,
                    "Date": created_at,
                    "Company": company or "未命名公司",
                    "Customer": customer_name or "未命名客戶",
                    "Salesperson": salesperson or "未填寫",
                    "Score": score,
                }
            )

        df = pd.DataFrame(dashboard_data)
        total_meetings = len(df)
        total_companies = df["Company"].nunique()
        high_count = len(df[df["Score"] == "HIGH"])
        medium_count = len(df[df["Score"] == "MEDIUM"])
        low_count = len(df[df["Score"] == "LOW"])

        st.subheader("📈 Sales Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Meetings", total_meetings)
        with c2:
            st.metric("Active Accounts", total_companies)
        with c3:
            st.metric("🔥 High Opportunities", high_count)
        with c4:
            st.metric("🟡 Medium Opportunities", medium_count)

        st.divider()
        st.subheader("🔥 Opportunity Distribution")
        score_data = pd.DataFrame(
            {
                "Opportunity": ["HIGH", "MEDIUM", "LOW"],
                "Count": [high_count, medium_count, low_count],
            }
        )
        st.bar_chart(score_data.set_index("Opportunity"))

        st.divider()
        st.subheader("🎯 Sales Opportunity Pipeline")
        high_col, medium_col, low_col = st.columns(3)
        for column, heading, score_name in [
            (high_col, "🔥 HIGH", "HIGH"),
            (medium_col, "🟡 MEDIUM", "MEDIUM"),
            (low_col, "⚪ LOW", "LOW"),
        ]:
            with column:
                st.markdown(f"## {heading}")
                opportunities = [x for x in dashboard_data if x["Score"] == score_name]
                if not opportunities:
                    st.caption(f"目前沒有 {score_name} Opportunity")
                for item in opportunities:
                    with st.container(border=True):
                        st.markdown(f"### 🏢 {item['Company']}")
                        st.write(f"👤 {item['Customer']}")
                        st.write(f"💼 {item['Salesperson']}")
                        st.caption(item["Date"])

        st.divider()
        st.subheader("🕒 Recent Sales Meetings")
        st.dataframe(
            df[["Date", "Company", "Customer", "Salesperson", "Score"]],
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# PAGE 4 - Ask Sales AI
# =========================================================

elif page == "💬 Ask Sales AI":
    st.title("💬 Ask Sales AI")
    st.write("直接詢問 Sales Memory JSON，找出案件重點、追蹤優先順序與成交訊號。")

    meetings = get_meetings()
    if not meetings:
        st.info("目前沒有 Meeting 資料，請先新增並儲存至少一場會議。")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Meetings in Memory", len(meetings))
        with c2:
            companies = {m[2] for m in meetings if m[2]}
            st.metric("Accounts in Memory", len(companies))

        st.divider()
        quick_question = st.selectbox(
            "⚡ 快速提問",
            [
                "請選擇...",
                "哪些客戶目前最值得優先追蹤？",
                "哪些客戶有明確的預算訊號？",
                "目前有哪些案件卡在決策者？",
                "目前最常見的客戶痛點是什麼？",
                "哪些案件有明確的下一步行動？",
                "請整理目前所有 Sales Opportunities。",
                "如果我是 Sales Manager，今天最應該注意什麼？",
            ],
        )
        custom_question = st.text_area(
            "💬 問 Sales AI",
            placeholder="例如：哪些客戶最值得我這週優先追蹤？",
            height=100,
        )
        final_question = custom_question.strip() or (quick_question if quick_question != "請選擇..." else "")

        if st.button("🧠 Ask Sales AI", type="primary", use_container_width=True):
            if not final_question:
                st.warning("請先輸入問題，或選擇一個快速提問。")
            else:
                try:
                    memories = []
                    for meeting in meetings[:30]:
                        meeting_json_text = meeting[9]
                        if meeting_json_text:
                            memories.append(meeting_json_text)
                        else:
                            memories.append(
                                json.dumps(
                                    {
                                        "meeting_id": meeting[0],
                                        "created_at": meeting[1],
                                        "company": meeting[2],
                                        "customer_name": meeting[3],
                                        "salesperson": meeting[4],
                                        "transcript": meeting[6],
                                        "sales_intelligence": safe_json_loads(meeting[8]) or meeting[8],
                                    },
                                    ensure_ascii=False,
                                )
                            )

                    sales_memory = "\n\n====================\n\n".join(memories)
                    with st.spinner("🧠 Sales AI 正在查詢 Meeting Memory..."):
                        response = client.responses.create(
                            model="gpt-5-mini",
                            instructions="""
你是一個 B2B Sales Manager Copilot。
只能根據提供的 Sales Meeting Memory JSON 回答。
不可捏造；資料不足時要明確說明。
使用繁體中文、結論優先，並在需要時提出具體下一步。
若比較客戶，請依據 stage、budget、need、decision_maker、objection、follow_up_date 與 next_action 排序；需要時引用 quotes 作為證據。
""",
                            input=f"SALES MEETING MEMORY:\n{sales_memory}\n\nQUESTION:\n{final_question}",
                            store=False,
                        )
                    st.session_state["ask_sales_question"] = final_question
                    st.session_state["ask_sales_answer"] = response.output_text
                except Exception as e:
                    st.error("❌ Sales AI 查詢失敗")
                    st.exception(e)

        if st.session_state.get("ask_sales_answer"):
            st.divider()
            st.subheader("🧠 Sales AI Answer")
            st.caption(
                f"Question：{st.session_state.get('ask_sales_question', '')}"
            )
            st.markdown(st.session_state["ask_sales_answer"])

            if st.button("清除目前問答結果", key="clear_ask_sales_answer"):
                st.session_state["ask_sales_question"] = ""
                st.session_state["ask_sales_answer"] = None
                st.rerun()
