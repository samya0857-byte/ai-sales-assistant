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
        "speaker_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "speaker": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": [
                            "sales_rep",
                            "customer",
                            "decision_maker",
                            "other",
                            "unknown",
                        ],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["speaker", "role", "confidence", "reason"],
            },
        },
        "summary": {"type": "string"},
        "customer_needs": {
            "type": "array",
            "items": {"type": "string"},
        },
        "pain_points": {
            "type": "array",
            "items": {"type": "string"},
        },
        "budget": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mentioned": {"type": "boolean"},
                "amount": {"type": ["number", "null"]},
                "currency": {"type": ["string", "null"]},
                "period": {"type": ["string", "null"]},
                "raw_text": {"type": "string"},
            },
            "required": ["mentioned", "amount", "currency", "period", "raw_text"],
        },
        "decision_maker": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mentioned": {"type": "boolean"},
                "name": {"type": ["string", "null"]},
                "role": {"type": ["string", "null"]},
                "evidence": {"type": "string"},
            },
            "required": ["mentioned", "name", "role", "evidence"],
        },
        "objections": {
            "type": "array",
            "items": {"type": "string"},
        },
        "next_actions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "follow_up": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mentioned": {"type": "boolean"},
                "raw_text": {"type": "string"},
                "normalized_date": {"type": ["string", "null"]},
            },
            "required": ["mentioned", "raw_text", "normalized_date"],
        },
        "opportunity_score": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW"],
        },
        "opportunity_reason": {"type": "string"},
        "sales_recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "speaker_roles",
        "summary",
        "customer_needs",
        "pain_points",
        "budget",
        "decision_maker",
        "objections",
        "next_actions",
        "follow_up",
        "opportunity_score",
        "opportunity_reason",
        "sales_recommendations",
    ],
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


def extract_opportunity_score(analysis):
    parsed = safe_json_loads(analysis)
    if isinstance(parsed, dict):
        score = parsed.get("opportunity_score")
        if score in {"HIGH", "MEDIUM", "LOW"}:
            return score

    text = str(analysis or "").upper()
    match = re.search(r"OPPORTUNITY SCORE.*?\b(HIGH|MEDIUM|LOW)\b", text, re.DOTALL)
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
你是一個 B2B Sales Intelligence Assistant。
你會收到一份結構化的會議資料，其中可能來自：多人語音轉錄、上傳逐字稿、PDF、TXT、JSON 或示範資料。

規則：
1. 只能根據輸入資料中實際存在的資訊，不可捏造。
2. Speaker label 只是講者分群；除非語意證據足夠，不可強行指定真實身份。
3. 資料未提及時：字串用「未提及」、陣列用空陣列、nullable 數值或日期用 null。
4. normalized_date 只有在能合理正規化時才填 YYYY-MM-DD；否則 null。
5. opportunity_score 僅能 HIGH / MEDIUM / LOW。
6. 使用繁體中文描述 summary、reason、needs、pain points、objections、actions、recommendations。
7. 請嚴格遵守提供的 JSON Schema，不要輸出 Schema 以外欄位。
""",
        input=json.dumps(transcription_payload, ensure_ascii=False, indent=2),
        text={
            "format": {
                "type": "json_schema",
                "name": "sales_intelligence",
                "description": "Structured B2B sales intelligence extracted from a meeting transcript.",
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

    st.subheader("🗣️ 講者角色判斷")
    roles = data.get("speaker_roles", [])
    if roles:
        for item in roles:
            st.write(
                f"- **{item.get('speaker', '?')}** → `{item.get('role', 'unknown')}` "
                f"({item.get('confidence', 'low')})：{item.get('reason', '')}"
            )
    else:
        st.write("未提及")

    st.subheader("📌 會議摘要")
    st.write(data.get("summary", "未提及"))

    left, right = st.columns(2)
    with left:
        st.subheader("🎯 客戶需求")
        needs = data.get("customer_needs", [])
        st.write("\n".join(f"- {x}" for x in needs) if needs else "未提及")

        st.subheader("⚠️ 客戶痛點")
        pain = data.get("pain_points", [])
        st.write("\n".join(f"- {x}" for x in pain) if pain else "未提及")

        st.subheader("💰 預算 / 價格訊號")
        budget = data.get("budget", {})
        st.write(budget.get("raw_text") or "未提及")

        st.subheader("👤 決策者")
        dm = data.get("decision_maker", {})
        if dm.get("mentioned"):
            st.write(f"{dm.get('name') or '未具名'} / {dm.get('role') or '角色未提及'}")
            st.caption(dm.get("evidence", ""))
        else:
            st.write("未提及")

    with right:
        st.subheader("🚧 客戶疑慮 / Objections")
        objections = data.get("objections", [])
        st.write("\n".join(f"- {x}" for x in objections) if objections else "未提及")

        st.subheader("✅ 下一步行動")
        actions = data.get("next_actions", [])
        st.write("\n".join(f"- {x}" for x in actions) if actions else "未提及")

        st.subheader("📅 Follow-up")
        fu = data.get("follow_up", {})
        st.write(fu.get("raw_text") or "未提及")
        if fu.get("normalized_date"):
            st.caption(f"Normalized: {fu['normalized_date']}")

        st.subheader("🔥 Opportunity Score")
        st.metric("Score", data.get("opportunity_score", "UNKNOWN"))
        st.caption(data.get("opportunity_reason", ""))

    st.subheader("💡 AI Sales 建議")
    recs = data.get("sales_recommendations", [])
    st.write("\n".join(f"- {x}" for x in recs) if recs else "未提及")


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

    sales_intelligence, analysis_usage = analyze_sales_strict(transcription_payload)
    api_usage.append(analysis_usage)

    final_meeting_json = {
        **transcription_payload,
        "translation": {
            "target_language": target_language,
            "speaker_transcript": translated_text,
        },
        "sales_intelligence": sales_intelligence,
        "api_usage_estimate": {
            "items": api_usage,
            "total_estimated_cost_usd": round(sum(float(x.get("estimated_cost_usd", 0) or 0) for x in api_usage), 6),
            "note": "Text-call costs use response token usage. Duration-only audio transcription uses a conservative internal planning allowance; verify final charges in OpenAI Usage/Costs.",
        },
    }

    return {
        "company": company,
        "customer_name": customer_name,
        "salesperson": salesperson,
        "target_language": target_language,
        "transcript": full_text,
        "speaker_segments": segments,
        "speaker_transcript": speaker_transcript,
        "translation": translated_text,
        "sales_intelligence": sales_intelligence,
        "analysis": json.dumps(sales_intelligence, ensure_ascii=False, indent=2),
        "meeting_json": final_meeting_json,
        "api_usage": api_usage,
        "estimated_api_cost_usd": final_meeting_json["api_usage_estimate"]["total_estimated_cost_usd"],
    }


init_database()


# =========================================================
# Session State
# =========================================================

if "meeting_result" not in st.session_state:
    st.session_state.meeting_result = None


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
    ],
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

        source_info = result.get("meeting_json", {}).get("source", {})
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
            st.json(result.get("meeting_json", {}).get("api_usage_estimate", {}))

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
        st.header("🧠 Strict JSON Sales Intelligence")
        render_sales_intelligence(result["sales_intelligence"])

        with st.expander("🔧 查看 Sales Intelligence JSON"):
            st.json(result["sales_intelligence"])

        with st.expander("🧩 查看完整 Meeting JSON"):
            st.json(result["meeting_json"])

        json_download = json.dumps(result["meeting_json"], ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ 下載完整 Meeting JSON",
            data=json_download,
            file_name=f"{result['meeting_json']['meeting_id']}.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.button("💾 儲存這場 Meeting", type="primary", use_container_width=True):
            save_meeting(result)
            st.success("✅ Meeting 與 Strict JSON 已儲存！")


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
                if isinstance(saved_json, dict):
                    segments = saved_json.get("transcription", {}).get("segments", [])
                    if segments:
                        st.subheader("🗣️ Speaker Transcript")
                        for segment in segments:
                            st.markdown(f"**{segment.get('speaker', 'Unknown')}**")
                            st.write(segment.get("text", ""))

                    st.subheader("🧠 Sales Intelligence")
                    intelligence = saved_json.get("sales_intelligence", {})
                    render_sales_intelligence(intelligence)
                    with st.expander("🔧 查看已儲存 Meeting JSON"):
                        st.json(saved_json)
                else:
                    st.subheader("📝 Transcript")
                    st.write(transcript)
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
                score = saved_json.get("sales_intelligence", {}).get("opportunity_score", "UNKNOWN")
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
若比較客戶，請依據 opportunity_score、預算、需求、決策者、objections、follow_up 與 next_actions 排序。
""",
                            input=f"SALES MEETING MEMORY:\n{sales_memory}\n\nQUESTION:\n{final_question}",
                            store=False,
                        )
                    st.divider()
                    st.subheader("🧠 Sales AI Answer")
                    st.markdown(response.output_text)
                except Exception as e:
                    st.error("❌ Sales AI 查詢失敗")
                    st.exception(e)
