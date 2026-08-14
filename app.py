import streamlit as st
from openai import OpenAI
import sqlite3
from datetime import datetime
import re
import pandas as pd

# =========================
# OpenAI
# =========================

client = OpenAI()


# =========================
# Database
# =========================

DB_FILE = "sales_meetings.db"


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
            analysis TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def save_meeting(
    company,
    customer_name,
    salesperson,
    target_language,
    transcript,
    translation,
    analysis
):

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
            analysis
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            company,
            customer_name,
            salesperson,
            target_language,
            transcript,
            translation,
            analysis
        )
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
            analysis
        FROM meetings
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return rows

def extract_opportunity_score(analysis):

    if not analysis:
        return "UNKNOWN"

    text = analysis.upper()

    match = re.search(
        r"OPPORTUNITY SCORE.*?\b(HIGH|MEDIUM|LOW)\b",
        text,
        re.DOTALL
    )

    if match:
        return match.group(1)

    # 備用判斷
    if "HIGH" in text:
        return "HIGH"

    if "MEDIUM" in text:
        return "MEDIUM"

    if "LOW" in text:
        return "LOW"

    return "UNKNOWN"
init_database()


# =========================
# Page
# =========================

st.set_page_config(
    page_title="AI Sales Assistant",
    page_icon="🧠",
    layout="wide"
)


# =========================
# Session State
# =========================

if "meeting_result" not in st.session_state:
    st.session_state.meeting_result = None


# =========================
# Sidebar
# =========================

st.sidebar.title("🧠 AI Sales Assistant")

page = st.sidebar.radio(
    "功能",
    [
        "🎙️ 新增會議",
        "📚 Meeting History",
        "📊 Sales Dashboard",
        "💬 Ask Sales AI"
    ]
)
# =========================================================
# PAGE 1
# 新增會議
# =========================================================

if page == "🎙️ 新增會議":

    st.title("🎙️ AI Sales Meeting Assistant")

    st.write(
        "將業務與客戶的對話，自動轉成逐字稿、翻譯，"
        "並由 AI 擷取銷售訊號與下一步行動。"
    )

    st.divider()


    # =========================
    # Customer information
    # =========================

    st.subheader("👤 Meeting Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        company = st.text_input(
            "🏢 公司名稱",
            placeholder="例如：ABC科技"
        )

    with col2:

        customer_name = st.text_input(
            "👤 客戶姓名",
            placeholder="例如：王經理"
        )

    with col3:

        salesperson = st.text_input(
            "💼 業務人員",
            placeholder="例如：紀承峰"
        )


    target_language = st.selectbox(
        "🌎 翻譯語言",
        [
            "English",
            "繁體中文",
            "日本語",
            "한국어",
            "ไทย",
            "Français",
            "Deutsch",
            "Español"
        ]
    )


    st.divider()


    # =========================
    # Audio
    # =========================

    st.subheader("🎤 Record Meeting")

    audio = st.audio_input(
        "按這裡開始錄音",
        sample_rate=16000
    )


    if audio:

        st.audio(audio)

        if st.button(
            "✨ AI 分析這場會議",
            type="primary",
            use_container_width=True
        ):

            try:

                # =========================
                # Speech to text
                # =========================

                with st.spinner(
                    "🎧 AI 正在辨識語音..."
                ):

                    transcript = (
                        client.audio.transcriptions.create(
                            model="gpt-4o-transcribe",
                            file=(
                                "recording.wav",
                                audio.getvalue(),
                                "audio/wav"
                            )
                        )
                    )

                original_text = transcript.text


                # =========================
                # Translation
                # =========================

                with st.spinner(
                    "🌎 GPT 正在翻譯..."
                ):

                    translation_response = (
                        client.responses.create(

                            model="gpt-5-mini",

                            instructions=f"""
你是一個專業商業翻譯助手。

請將使用者文字翻譯成：

{target_language}

規則：

1. 保留原意。
2. 不要摘要。
3. 不要解釋。
4. 公司、人名、價格、日期與數字必須保留。
5. 口語要自然。
6. 只輸出翻譯結果。
""",

                            input=original_text,

                            store=False
                        )
                    )

                translated_text = (
                    translation_response.output_text
                )


                # =========================
                # Sales Intelligence
                # =========================

                with st.spinner(
                    "🧠 AI 正在分析 Sales Intelligence..."
                ):

                    sales_response = (
                        client.responses.create(

                            model="gpt-5-mini",

                            instructions="""
你是一個 B2B Sales Intelligence Assistant。

請分析這場客戶會議。

只能使用逐字稿中實際存在的資訊。
不可捏造。

如果沒有資訊，請寫「未提及」。

請使用繁體中文。

固定按照以下格式回答：

### 📌 會議摘要
1～3 句摘要。

### 🎯 客戶需求
列出真正需求。

### ⚠️ 客戶痛點
列出目前問題。

### 💰 預算 / 價格訊號
客戶預算或價格態度。
沒有就寫未提及。

### 👤 決策者
誰有購買決策權。
沒有就寫未提及。

### 🚧 客戶疑慮 / Objections
可能阻礙成交的事情。

### ✅ 下一步行動
業務下一步應做什麼。

### 📅 Follow-up
客戶提到的下一次聯絡時間。

### 🔥 Opportunity Score
只能輸出：
HIGH
MEDIUM
LOW

判斷依據：
需求強度、
預算、
購買意願、
決策權、
下一步是否明確。

### 💡 AI Sales 建議
提供具體且簡潔的追蹤策略。
""",

                            input=original_text,

                            store=False
                        )
                    )

                sales_analysis = (
                    sales_response.output_text
                )


                # =========================
                # Store result in session
                # =========================

                st.session_state.meeting_result = {

                    "company": company,

                    "customer_name": customer_name,

                    "salesperson": salesperson,

                    "target_language":
                        target_language,

                    "transcript":
                        original_text,

                    "translation":
                        translated_text,

                    "analysis":
                        sales_analysis

                }

                st.success(
                    "✅ AI Meeting Analysis 完成！"
                )

            except Exception as e:

                st.error(
                    "❌ AI 處理失敗"
                )

                st.write(e)


    # =========================
    # Display result
    # =========================

    result = st.session_state.meeting_result


    if result:

        st.divider()

        st.header(
            "📝 Meeting Transcript"
        )

        left, right = st.columns(2)


        with left:

            st.subheader(
                "🎙️ 原始語音"
            )

            st.text_area(
                "original_text",
                result["transcript"],
                height=220,
                label_visibility="collapsed"
            )


        with right:

            st.subheader(
                f"🌎 {result['target_language']}"
            )

            st.text_area(
                "translated_text",
                result["translation"],
                height=220,
                label_visibility="collapsed"
            )


        # =========================
        # Intelligence
        # =========================

        st.divider()

        st.header(
            "🧠 AI Sales Intelligence"
        )

        st.markdown(
            result["analysis"]
        )


        # =========================
        # Save
        # =========================

        st.divider()

        if st.button(
            "💾 儲存這場 Meeting",
            type="primary",
            use_container_width=True
        ):

            save_meeting(

                result["company"],

                result["customer_name"],

                result["salesperson"],

                result["target_language"],

                result["transcript"],

                result["translation"],

                result["analysis"]
            )

            st.success(
                "✅ Meeting 已儲存！"
            )


        # =========================
        # Download
        # =========================

        download_text = f"""
AI SALES MEETING

Company:
{result["company"]}

Customer:
{result["customer_name"]}

Salesperson:
{result["salesperson"]}


ORIGINAL TRANSCRIPT
==============================

{result["transcript"]}


TRANSLATION
==============================

{result["translation"]}


SALES INTELLIGENCE
==============================

{result["analysis"]}
"""


        st.download_button(
            "⬇️ 下載完整分析",
            download_text,
            file_name="sales_meeting.txt",
            mime="text/plain",
            use_container_width=True
        )


# =========================================================
# PAGE 2
# Meeting History
# =========================================================

elif page == "📚 Meeting History":

    st.title(
        "📚 Sales Meeting History"
    )

    st.write(
        "所有客戶會議與 AI 分析紀錄。"
    )

    meetings = get_meetings()


    # =========================
    # No data
    # =========================

    if not meetings:

        st.info(
            "目前還沒有儲存任何 Meeting。"
        )


    # =========================
    # Meeting list
    # =========================

    else:

        st.metric(
            "Total Meetings",
            len(meetings)
        )

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
                analysis
            ) = meeting


            title = (
                f"🏢 {company or '未命名公司'}"
                f" ｜ 👤 {customer_name or '未命名客戶'}"
                f" ｜ {created_at}"
            )


            with st.expander(title):

                col1, col2, col3 = (
                    st.columns(3)
                )

                with col1:

                    st.write(
                        "**公司**"
                    )

                    st.write(
                        company or "未填寫"
                    )


                with col2:

                    st.write(
                        "**客戶**"
                    )

                    st.write(
                        customer_name or "未填寫"
                    )


                with col3:

                    st.write(
                        "**業務人員**"
                    )

                    st.write(
                        salesperson or "未填寫"
                    )


                st.divider()


                st.subheader(
                    "📝 Transcript"
                )

                st.write(
                    transcript
                )


                st.subheader(
                    "🌎 Translation"
                )

                st.write(
                    translation
                )


                st.subheader(
                    "🧠 Sales Intelligence"
                )

                st.markdown(
                    analysis
                )# =========================================================
# PAGE 3
# Sales Dashboard
# =========================================================

elif page == "📊 Sales Dashboard":

    st.title("📊 Sales Dashboard")

    st.write(
        "將所有 Sales Meetings 轉換成主管可以快速掌握的 Pipeline。"
    )

    meetings = get_meetings()

    if not meetings:

        st.info(
            "目前沒有 Meeting 資料，請先新增並儲存至少一場會議。"
        )

    else:

        # ======================================
        # Prepare Data
        # ======================================

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
                analysis
            ) = meeting

            score = extract_opportunity_score(
                analysis
            )

            dashboard_data.append({

                "Meeting ID":
                    meeting_id,

                "Date":
                    created_at,

                "Company":
                    company or "未命名公司",

                "Customer":
                    customer_name or "未命名客戶",

                "Salesperson":
                    salesperson or "未填寫",

                "Score":
                    score,

                "Transcript":
                    transcript,

                "Analysis":
                    analysis
            })


        df = pd.DataFrame(
            dashboard_data
        )


        # ======================================
        # KPI
        # ======================================

        total_meetings = len(df)

        total_companies = (
            df["Company"]
            .nunique()
        )

        high_count = len(
            df[
                df["Score"]
                == "HIGH"
            ]
        )

        medium_count = len(
            df[
                df["Score"]
                == "MEDIUM"
            ]
        )

        low_count = len(
            df[
                df["Score"]
                == "LOW"
            ]
        )


        st.subheader(
            "📈 Sales Overview"
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )


        with c1:

            st.metric(
                "Total Meetings",
                total_meetings
            )


        with c2:

            st.metric(
                "Active Accounts",
                total_companies
            )


        with c3:

            st.metric(
                "🔥 High Opportunities",
                high_count
            )


        with c4:

            st.metric(
                "🟡 Medium Opportunities",
                medium_count
            )


        st.divider()


        # ======================================
        # Opportunity Chart
        # ======================================

        st.subheader(
            "🔥 Opportunity Distribution"
        )


        score_data = pd.DataFrame({

            "Opportunity": [
                "HIGH",
                "MEDIUM",
                "LOW"
            ],

            "Count": [
                high_count,
                medium_count,
                low_count
            ]

        })


        st.bar_chart(
            score_data.set_index(
                "Opportunity"
            )
        )


        st.divider()


        # ======================================
        # Pipeline
        # ======================================

        st.subheader(
            "🎯 Sales Opportunity Pipeline"
        )


        high_col, medium_col, low_col = (
            st.columns(3)
        )


        # ======================================
        # HIGH
        # ======================================

        with high_col:

            st.markdown(
                "## 🔥 HIGH"
            )

            high_opps = [
                item
                for item in dashboard_data
                if item["Score"] == "HIGH"
            ]


            if not high_opps:

                st.caption(
                    "目前沒有 HIGH Opportunity"
                )


            for item in high_opps:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### 🏢 {item['Company']}"
                    )

                    st.write(
                        f"👤 {item['Customer']}"
                    )

                    st.write(
                        f"💼 {item['Salesperson']}"
                    )

                    st.caption(
                        item["Date"]
                    )


        # ======================================
        # MEDIUM
        # ======================================

        with medium_col:

            st.markdown(
                "## 🟡 MEDIUM"
            )

            medium_opps = [
                item
                for item in dashboard_data
                if item["Score"] == "MEDIUM"
            ]


            if not medium_opps:

                st.caption(
                    "目前沒有 MEDIUM Opportunity"
                )


            for item in medium_opps:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### 🏢 {item['Company']}"
                    )

                    st.write(
                        f"👤 {item['Customer']}"
                    )

                    st.write(
                        f"💼 {item['Salesperson']}"
                    )

                    st.caption(
                        item["Date"]
                    )


        # ======================================
        # LOW
        # ======================================

        with low_col:

            st.markdown(
                "## ⚪ LOW"
            )

            low_opps = [
                item
                for item in dashboard_data
                if item["Score"] == "LOW"
            ]


            if not low_opps:

                st.caption(
                    "目前沒有 LOW Opportunity"
                )


            for item in low_opps:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### 🏢 {item['Company']}"
                    )

                    st.write(
                        f"👤 {item['Customer']}"
                    )

                    st.write(
                        f"💼 {item['Salesperson']}"
                    )

                    st.caption(
                        item["Date"]
                    )


        st.divider()


        # ======================================
        # Recent Meetings
        # ======================================

        st.subheader(
            "🕒 Recent Sales Meetings"
        )


        display_df = df[
            [
                "Date",
                "Company",
                "Customer",
                "Salesperson",
                "Score"
            ]
        ]


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )# =========================================================
# PAGE 4
# Ask Sales AI
# =========================================================

elif page == "💬 Ask Sales AI":

    st.title("💬 Ask Sales AI")

    st.write(
        "直接詢問你的 Sales Meeting Database，"
        "讓 AI 幫主管找出案件重點、追蹤優先順序與成交訊號。"
    )

    meetings = get_meetings()

    # ======================================
    # 沒有資料
    # ======================================

    if not meetings:

        st.info(
            "目前沒有 Meeting 資料，請先新增並儲存至少一場會議。"
        )

    else:

        # ======================================
        # Database Status
        # ======================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Meetings in Memory",
                len(meetings)
            )

        with col2:

            companies = set()

            for meeting in meetings:

                company = meeting[2]

                if company:
                    companies.add(company)

            st.metric(
                "Accounts in Memory",
                len(companies)
            )

        st.divider()


        # ======================================
        # Quick Questions
        # ======================================

        st.subheader("⚡ 快速提問")

        quick_question = st.selectbox(
            "選一個問題，或自己在下面輸入",
            [
                "請選擇...",
                "哪些客戶目前最值得優先追蹤？",
                "哪些客戶有明確的預算訊號？",
                "目前有哪些案件卡在決策者？",
                "目前最常見的客戶痛點是什麼？",
                "哪些案件有明確的下一步行動？",
                "請整理目前所有 Sales Opportunities。",
                "如果我是 Sales Manager，今天最應該注意什麼？"
            ]
        )


        # ======================================
        # Custom Question
        # ======================================

        st.subheader("💬 問 Sales AI")

        custom_question = st.text_area(
            "輸入你的問題",
            placeholder=(
                "例如：哪些客戶最值得我這週優先追蹤？"
            ),
            height=100
        )


        # ======================================
        # 決定問題
        # ======================================

        if custom_question.strip():

            final_question = custom_question.strip()

        elif quick_question != "請選擇...":

            final_question = quick_question

        else:

            final_question = ""


        # ======================================
        # Ask AI
        # ======================================

        if st.button(
            "🧠 Ask Sales AI",
            type="primary",
            use_container_width=True
        ):

            if not final_question:

                st.warning(
                    "請先輸入問題，或選擇一個快速提問。"
                )

            else:

                try:

                    # ======================================
                    # Build Sales Memory
                    # ======================================

                    meeting_context_list = []


                    # MVP 先讀最近 30 場 Meeting
                    for meeting in meetings[:30]:

                        (
                            meeting_id,
                            created_at,
                            company,
                            customer_name,
                            salesperson,
                            language,
                            transcript,
                            translation,
                            analysis
                        ) = meeting


                        meeting_context = f"""
MEETING ID: {meeting_id}

日期：
{created_at}

公司：
{company or "未填寫"}

客戶：
{customer_name or "未填寫"}

業務：
{salesperson or "未填寫"}

原始逐字稿：
{transcript}

AI Sales Intelligence：
{analysis}
"""


                        meeting_context_list.append(
                            meeting_context
                        )


                    sales_memory = "\n\n====================\n\n".join(
                        meeting_context_list
                    )


                    # ======================================
                    # GPT
                    # ======================================

                    with st.spinner(
                        "🧠 Sales AI 正在查詢 Meeting Memory..."
                    ):

                        response = client.responses.create(

                            model="gpt-5-mini",

                            instructions="""
你是一個 B2B Sales Manager Copilot。

你可以讀取公司的 Sales Meeting Memory，
並回答主管關於客戶、案件、成交機會、
Follow-up、預算、痛點與 Sales Pipeline 的問題。

重要規則：

1. 只能根據提供的 Meeting Memory 回答。

2. 不可以捏造沒有出現在資料中的資訊。

3. 如果資料不足，直接說「目前 Meeting Memory 中沒有足夠資訊」。

4. 回答必須使用繁體中文。

5. 回答要簡潔、有結論，適合 Sales Manager 快速閱讀。

6. 如果需要比較多個客戶：
   請使用排序方式回答。

7. 如果問「誰最值得追」：
   請綜合以下訊號：
   - Opportunity Score
   - 明確需求
   - 預算
   - 購買意願
   - 決策者
   - Objection
   - Follow-up
   - 是否有明確下一步

8. 每個重要判斷都應說明原因。

9. 如果有明確下一步，
   請提出具體建議。

10. 不要回答與 Sales Meeting Memory 無關的問題。
""",

                            input=f"""
以下是公司的 Sales Meeting Memory：

==============================
SALES MEETING MEMORY
==============================

{sales_memory}


==============================
SALES MANAGER QUESTION
==============================

{final_question}
""",

                            store=False
                        )


                    answer = response.output_text


                    # ======================================
                    # Display
                    # ======================================

                    st.divider()

                    st.subheader("🧠 Sales AI Answer")

                    st.markdown(answer)


                    # ======================================
                    # Question
                    # ======================================

                    with st.expander(
                        "查看主管問題"
                    ):

                        st.write(
                            final_question
                        )


                except Exception as e:

                    st.error(
                        "❌ Sales AI 查詢失敗"
                    )

                    st.write(e)