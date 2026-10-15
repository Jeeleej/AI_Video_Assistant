import html
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(page_title="AI Video Assistant", page_icon="🎬", layout="wide")

# ─── Styling: "editing room" — a dark ink UI with a timeline playhead as the signature ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{
  --ink:#0e1116; --panel:#151a22; --panel2:#1b212b; --line:#252d39;
  --text:#e6e9ee; --muted:#8a94a3; --rec:#ff5d4d; --ok:#3fb8a5; --amber:#e8b04a;
}
html, body, .stApp, [class*="css"]{font-family:'IBM Plex Sans',sans-serif; background:var(--ink)!important; color:var(--text)!important;}
h1,h2,h3,h4{font-family:'Bricolage Grotesque',sans-serif!important; color:var(--text)!important; letter-spacing:-.01em;}
[data-testid="stSidebar"]{background:var(--panel)!important; border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stHeader"]{background:transparent!important;}
.block-container{padding-top:2rem; max-width:1200px;}
label, .stCaption{color:var(--muted)!important;}

.brand{font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:2.6rem; line-height:1.05; margin:0;}
.tagline{color:var(--muted); margin:.4rem 0 1.4rem; font-size:.95rem;}
.mono{font-family:'IBM Plex Mono',monospace;}

/* Buttons & inputs */
.stButton>button, .stDownloadButton>button{
  background:var(--rec)!important; color:#fff!important; border:0!important; border-radius:6px!important;
  font-family:'Bricolage Grotesque',sans-serif!important; font-weight:700!important; padding:.55rem 1.2rem!important;}
.stButton>button:hover{filter:brightness(1.1);}
.stButton>button:focus-visible, input:focus-visible{outline:2px solid var(--amber)!important; outline-offset:2px;}
.stTextInput input, [data-baseweb="select"]>div{background:var(--panel2)!important; border:1px solid var(--line)!important; color:var(--text)!important; border-radius:6px!important;}

/* Live pipeline timeline */
.tl{background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:1.1rem 1.2rem 1rem; margin-bottom:1.2rem;}
.tl-head{display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.9rem;}
.tl-head b{font-family:'Bricolage Grotesque',sans-serif; font-size:1.05rem;}
.tl-time{font-family:'IBM Plex Mono',monospace; color:var(--muted); font-size:.8rem;}
.tl-track{display:grid; grid-template-columns:repeat(6,1fr); gap:6px;}
.seg .bar{height:8px; border-radius:2px; background:var(--line); position:relative; overflow:hidden;}
.seg.done .bar{background:var(--ok);}
.seg.error .bar{background:var(--rec);}
.seg.active .bar{background:var(--line);}
.seg.active .bar::after{content:''; position:absolute; inset:0; width:45%;
  background:linear-gradient(90deg,transparent,var(--rec),transparent); animation:scan 1.1s ease-in-out infinite;}
@keyframes scan{from{transform:translateX(-100%)}to{transform:translateX(230%)}}
.seg .nm{margin-top:.55rem; font-size:.82rem; font-weight:500;}
.seg.pending .nm{color:var(--muted);}
.seg .dt{font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:var(--muted); min-height:1em;}
.seg.active .dt{color:var(--rec);}
@media (prefers-reduced-motion:reduce){.seg.active .bar::after{animation:none; width:100%; opacity:.5}}
@media (max-width:700px){.tl-track{grid-template-columns:repeat(3,1fr); row-gap:14px}}

/* Content panels */
.panel{background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:1.2rem 1.3rem; margin-bottom:1rem;}
.panel h4{margin:0 0 .8rem; font-size:1rem;}
.title-out{font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.8rem; line-height:1.2; margin:.2rem 0 1rem;}
.body{line-height:1.75; font-size:.95rem; max-width:75ch;}
.item{display:flex; gap:.7rem; padding:.55rem 0; border-top:1px solid var(--line); font-size:.9rem; line-height:1.55;}
.item:first-of-type{border-top:0;}
.item i{flex:0 0 10px; height:10px; margin-top:.4rem; border-radius:2px; border:2px solid var(--ok);}
.item.dec i{border-color:var(--amber);}
.item.q i{border-color:var(--rec); border-radius:50%;}
.stat{background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:.8rem 1rem;}
.stat .v{font-family:'Bricolage Grotesque',sans-serif; font-size:1.6rem; font-weight:700;}
.stat .l{color:var(--muted); font-size:.78rem;}
.transcript{background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:1.1rem; font-family:'IBM Plex Mono',monospace;
  font-size:.82rem; line-height:1.8; max-height:520px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; color:#b9c0cc;}
.empty{border:1px dashed var(--line); border-radius:8px; padding:3.5rem 1.5rem; text-align:center; color:var(--muted);}
.empty b{display:block; color:var(--text); font-family:'Bricolage Grotesque',sans-serif; font-size:1.3rem; margin-bottom:.4rem;}
.stTabs [data-baseweb="tab"]{font-family:'Bricolage Grotesque',sans-serif; font-weight:700;}
.stTabs [aria-selected="true"]{color:var(--rec)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--rec)!important;}
</style>
""", unsafe_allow_html=True)

# ─── State ──────────────────────────────────────────────────────────────────────
for k, v in {"result": None, "chat_history": [], "steps": {}, "times": {}}.items():
    st.session_state.setdefault(k, v)

STEPS = [
    ("audio", "Audio"), ("transcript", "Transcribe"), ("title", "Title"),
    ("summary", "Summarise"), ("extract", "Extract"), ("rag", "Index"),
]

# ─── Helpers ────────────────────────────────────────────────────────────────────
def esc(t) -> str:
    return html.escape(str(t)).replace("\n", "<br>")

def items(text) -> list[str]:
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).replace("**", "").strip()
             for l in str(text).splitlines()]
    return [l for l in lines if l]

def item_list(text, cls="") -> str:
    rows = items(text)
    if not rows:
        return '<div class="body" style="color:var(--muted)">None found.</div>'
    return "".join(f'<div class="item {cls}"><i></i><span>{html.escape(r)}</span></div>' for r in rows)

def timeline_html(steps: dict, times: dict, total: float | None = None) -> str:
    segs = ""
    for key, name in STEPS:
        state = steps.get(key, "pending")
        t = times.get(key)
        label = {"done": f"{t:.1f}s" if t is not None else "", "active": "running", "error": "failed"}.get(state, "")
        segs += f'<div class="seg {state}"><div class="bar"></div><div class="nm">{name}</div><div class="dt">{label}</div></div>'
    done = sum(1 for k, _ in STEPS if steps.get(k) == "done")
    right = f"{total:.1f}s total" if total is not None else f"{done}/{len(STEPS)} steps"
    return f'<div class="tl"><div class="tl-head"><b>Pipeline</b><span class="tl-time">{right}</span></div><div class="tl-track">{segs}</div></div>'

def build_report(r) -> str:
    return (f"# {r['title']}\n\n## Summary\n{r['summary']}\n\n## Action items\n{r['action_items']}\n\n"
            f"## Key decisions\n{r['key_decisions']}\n\n## Open questions\n{r['open_questions']}\n\n"
            f"## Transcript\n{r['transcript']}\n")

# ─── Sidebar: input ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand" style="font-size:1.7rem">AI Video<br>Assistant</div>', unsafe_allow_html=True)
    st.caption("Turn a recording into notes you can search.")
    st.markdown("---")
    input_type = st.radio("Source", ["YouTube URL", "Upload Video / Audio"], label_visibility="collapsed")
    source, uploaded_file = None, None
    if input_type == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader("File", type=["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a"])
    language = st.selectbox("Spoken language", ["english", "hinglish"])
    run_btn = st.button("Analyse recording", use_container_width=True)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="brand">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Transcribe, summarise and ask questions about any meeting or video.</div>', unsafe_allow_html=True)

live_slot = st.empty()   # live pipeline timeline
feed_slot = st.empty()   # results appear here as each step completes

# ─── Run pipeline (updates the UI after every step) ─────────────────────────────
if run_btn:
    target = None
    if input_type == "YouTube URL":
        if source and source.strip():
            target = source.strip()
        else:
            st.error("Enter a YouTube URL in the sidebar.")
    elif uploaded_file is not None:
        os.makedirs("downloads", exist_ok=True)
        target = os.path.join("downloads", uploaded_file.name)
        with open(target, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.error("Choose a file to upload in the sidebar.")

    if target:
        st.session_state.update(result=None, chat_history=[], steps={}, times={})
        steps, times, partial = st.session_state.steps, st.session_state.times, {}
        t0, started = time.time(), {}

        def paint():
            live_slot.markdown(timeline_html(steps, times), unsafe_allow_html=True)
            parts = ""
            if "title" in partial:
                parts += f'<div class="panel"><div class="title-out">{esc(partial["title"])}</div></div>'
            if "summary" in partial:
                parts += f'<div class="panel"><h4>Summary</h4><div class="body">{esc(partial["summary"])}</div></div>'
            feed_slot.markdown(parts, unsafe_allow_html=True)

        def begin(key):
            steps[key] = "active"; started[key] = time.time(); paint()

        def finish(key, **out):
            steps[key] = "done"; times[key] = time.time() - started[key]; partial.update(out); paint()

        current = None
        try:
            current = "audio";      begin(current); chunks = process_input(target);                finish(current)
            current = "transcript"; begin(current); transcript = transcribe_all(chunks, language); finish(current)
            current = "title";      begin(current); title = generate_title(transcript);            finish(current, title=title)
            time.sleep(3)  # cooldown between LLM calls
            current = "summary";    begin(current); summary = summarize(transcript);               finish(current, summary=summary)
            time.sleep(3)
            current = "extract";    begin(current)
            action_items = extract_action_items(transcript); time.sleep(2)
            decisions = extract_key_decisions(transcript);   time.sleep(2)
            questions = extract_questions(transcript);       finish(current)
            current = "rag";        begin(current); rag_chain = build_rag_chain(transcript);       finish(current)

            st.session_state.result = {
                "title": title, "transcript": transcript, "summary": summary,
                "action_items": action_items, "key_decisions": decisions,
                "open_questions": questions, "rag_chain": rag_chain,
                "total": time.time() - t0,
            }
            st.rerun()
        except Exception as e:
            if current:
                steps[current] = "error"
            paint()
            st.error(f"Failed during “{current}”: {e}")

# ─── Results ────────────────────────────────────────────────────────────────────
r = st.session_state.result
if r:
    live_slot.markdown(timeline_html(st.session_state.steps, st.session_state.times, r.get("total")), unsafe_allow_html=True)
    feed_slot.empty()

    st.markdown(f'<div class="title-out">{esc(r["title"])}</div>', unsafe_allow_html=True)

    words = len(r["transcript"].split())
    stats = [
        (f"{words:,}", "words transcribed"),
        (f"~{max(1, round(words / 150))} min", "estimated speech length"),
        (len(items(r["action_items"])), "action items"),
        (len(items(r["key_decisions"])), "decisions"),
        (len(items(r["open_questions"])), "open questions"),
    ]
    for col, (v, l) in zip(st.columns(5), stats):
        col.markdown(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    st.write("")

    tab_over, tab_trans, tab_chat = st.tabs(["Overview", "Transcript", "Ask the meeting"])

    with tab_over:
        st.markdown(f'<div class="panel"><h4>Summary</h4><div class="body">{esc(r["summary"])}</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        c1.markdown(f'<div class="panel"><h4>Action items</h4>{item_list(r["action_items"])}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="panel"><h4>Key decisions</h4>{item_list(r["key_decisions"], "dec")}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="panel"><h4>Open questions</h4>{item_list(r["open_questions"], "q")}</div>', unsafe_allow_html=True)
        st.download_button("Download report (.md)", build_report(r), file_name="meeting-report.md", mime="text/markdown")

    with tab_trans:
        st.markdown(f'<div class="transcript">{html.escape(r["transcript"])}</div>', unsafe_allow_html=True)

    with tab_chat:
        if not st.session_state.chat_history:
            st.markdown('<div class="empty"><b>Ask anything about this recording</b>Try “What did we decide about the deadline?”</div>', unsafe_allow_html=True)
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        q = st.chat_input("Ask about the meeting…")
        if q and q.strip():
            st.session_state.chat_history.append({"role": "user", "content": q.strip()})
            with st.chat_message("user"):
                st.write(q.strip())
            with st.chat_message("assistant"):
                with st.spinner("Searching the transcript…"):
                    ans = ask_question(r["rag_chain"], q.strip())
                st.write(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
        if st.session_state.chat_history and st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

elif not run_btn:
    st.markdown('<div class="empty"><b>Add a recording to begin</b>Paste a YouTube link or upload a file in the sidebar, then choose Analyse recording.</div>', unsafe_allow_html=True)