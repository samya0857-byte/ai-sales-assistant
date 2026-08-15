# Lead Discovery Integration

This build adds `🧠 Knowledge Base` to the Streamlit sidebar.

Flow:
1. Select an existing customer from saved Meeting JSON.
2. AI aggregates the customer's knowledge profile.
3. AI creates 3 bounded web-search queries.
4. OpenAI Responses API `web_search` researches public company-level signals.
5. A second strict JSON Schema pass deduplicates and ranks potential companies.
6. Results are stored in local SQLite and, when configured, AWS S3:
   `sync-pipeline/lead-discovery/YYYY-MM-DD/...json`

Important source policy:
- Public company websites, news, industry/government sources and public job-posting signals are supported.
- No candidate/resume/private-person data is collected.
- Automated LinkedIn scraping is intentionally excluded. Use an approved LinkedIn integration/API later if authorized.

No new Python dependency is required beyond the existing app dependencies.
