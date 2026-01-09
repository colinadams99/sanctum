from pathlib import Path
import sys
import time
import streamlit as st

# ensures that the project root is in the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# sanctum imports
from sdk.core import SanctumAI
from sdk.doc_loader import load_pdf_text
from sdk.retrieval import chunk_text, build_tfidf_index, retrieve_top_k

# page configuration
st.set_page_config(
    page_title='Sanctum — Portfolio Finance Doc Assistant',
    page_icon='S',
    layout='wide',
)

st.session_state.setdefault('warmed', False)


def sanctum_mark_svg(size: int = 22) -> str:
    # Minimal, geometric 'S' mark in a rounded square (non-AI, clean, brandable)
    return f'''
    <svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg' style='vertical-align:-4px;'>
      <rect x='1.5' y='1.5' width='21' height='21' rx='6' stroke='currentColor' stroke-width='1.5' opacity='0.9'/>
      <path d='M16.6 8.2c-.7-1-2.1-1.7-4-1.7-2.2 0-3.9 1.1-3.9 2.8 0 1.6 1.4 2.2 3.7 2.6 2.5.4 4.8 1.2 4.8 3.6 0 2.4-2.2 3.9-5 3.9-2.6 0-4.5-1-5.4-2.7'
            stroke='currentColor' stroke-width='1.7' stroke-linecap='round'/>
    </svg>
    '''

# CSS
st.markdown(
    '''
    <style>
      /* App background + typography */
      .stApp {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }

      /* Make the main container feel like a centered product page */
      .block-container {
        padding-top: 2.0rem;
        padding-bottom: 2.0rem;
        max-width: 1180px;
      }

      /* Title spacing */
      h1, h2, h3 {
        letter-spacing: -0.02em;
      }

      /* Sleeker input look */
      textarea, input {
        border-radius: 12px !important;
      }

      /* Buttons */
      .stButton button {
        border-radius: 14px;
        padding: 0.65rem 1rem;
        font-weight: 650;
      }

      /* 'Card' utility */
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.035);
        border-radius: 16px;
        padding: 1.0rem 1.1rem;
        margin-bottom: 1.0rem;
      }

      /* Small badge/chip */
      .chip {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
        font-size: 0.82rem;
        margin-right: 0.4rem;
      }

      /* Subtle divider */
      .divider {
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin: 1rem 0;
      }

      /* Sidebar polish */
      section[data-testid='stSidebar'] {
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      /* Softer labels */
      label {
        opacity: 0.85;
      }

      /* Output readability */
      .stMarkdown {
        line-height: 1.55;
      }
    </style>
    ''',
    unsafe_allow_html=True,
)

# ---------- header ----------
st.markdown(
    f'''
    <div class='card'>
      <div style='display:flex; align-items:center; justify-content:space-between; gap: 1rem;'>
        <div>
          <div style='font-size: 0.95rem; opacity: 0.85;'>Local • Private • On-device</div>
          <div style='font-size: 2.0rem; font-weight: 780; margin-top: 0.2rem; display:flex; align-items:center; gap: 10px;'>
            <span>{sanctum_mark_svg(22)}</span>
            <span>Sanctum</span>
          </div>
          <div style='font-size: 1.05rem; opacity: 0.88; margin-top: 0.2rem;'>
            Portfolio Finance / NAV Facility Document Assistant
          </div>
        </div>
        <div style='text-align:right;'>
          <span class='chip'>No cloud calls</span>
          <span class='chip'>Sensitive-doc safe</span>
          <span class='chip'>GPU-ready</span>
        </div>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ---------- model init ----------
@st.cache_resource
def get_ai() -> SanctumAI:
    return SanctumAI()

ai = get_ai()

# warms up the cuda kernels
if not st.session_state['warmed']:
    with st.spinner('Warming up model...'):
        _ = ai.analyze_finance_doc(
            doc_text='Test.',
            doc_type='other',
            task='Respond with OK.',
            context=None,
            max_new_tokens=8,
        )
    st.session_state['warmed'] = True


# ---------- sidebar controls ----------
with st.sidebar:
    st.markdown('### Controls')
    st.caption('Configure analysis settings and input type.')

    mode = st.radio('Input', ['Paste text', 'Upload PDF'], horizontal = False)

    doc_type = st.selectbox(
        'Document type',
        ['term_sheet', 'credit_agreement', 'LPA', 'memo', 'nav_report', 'other'],
    )

    st.markdown('### Retrieval')
    top_k = st.slider('Top-k chunks', min_value = 1, max_value = 10, value = 6, step = 1)
    chunk_size = st.slider('Chunk size', min_value = 600, max_value = 2000, value = 1200, step = 100)
    overlap = st.slider('Overlap (chars)', min_value = 0, max_value = 400, value = 200, step = 50)
    show_debug = st.toggle('Show retrieved chunks', value = False)

    st.markdown('### Output')
    max_new_tokens = st.slider('Max new tokens', min_value = 60, max_value = 400, value = 200, step = 20)

# -========= main input area =====
col_left, col_right = st.columns([1.05, 0.95], gap = 'large')

with col_left:
    st.markdown('<div class=\'card\'>', unsafe_allow_html = True)
    st.markdown('### Document input')

    doc_text = ''
    if mode == 'Paste text':
        doc_text = st.text_area(
            'Paste document text',
            height=320,
            placeholder='Paste term sheet / credit agreement / LPA excerpt…',
        )
    else:
        pdf_file = st.file_uploader('Upload PDF', type = ['pdf'])
        if pdf_file is not None:
            tmp_path = Path('tmp_upload.pdf')
            tmp_path.write_bytes(pdf_file.read())

            # If your loader supports max_chars, keep this. If not, switch to load_pdf_text(tmp_path)
            doc_text = load_pdf_text(tmp_path, max_chars = 60000)

            st.caption('PDF text extracted locally. Large documents may be truncated for performance.')

    st.markdown('</div>', unsafe_allow_html = True)

    st.markdown('<div class=\'card\'>', unsafe_allow_html = True)
    st.markdown('### Task')

    task = st.text_input(
        'What do you want to know?',
        value='What are the main risks to the lender?',
    )

    context = st.text_input(
        'Context (optional)',
        value='Audience: portfolio finance team at an insurer or asset manager.',
    )

    run = st.button('Analyze', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class=\'card\'>', unsafe_allow_html = True)
    st.markdown('### Output')
    st.caption('This runs locally. No documents leave this machine.')
    output_placeholder = st.empty()
    meta_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# runs the pipeline
if run:
    if not doc_text.strip():
        output_placeholder.error('Please paste text or upload a PDF first.')
    else:
        t0 = time.time()

        # Detect summary-style tasks to reduce retrieval work
        is_summary = any(
            kw in task.lower()
            for kw in ['summary', 'summarize', 'brief', 'overview', 'high-level']
        )

        # Retrieval
        with st.spinner('Retrieving relevant sections...'):

            doc_text = doc_text.replace('\u00a0', ' ')
            chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=overlap)

            if not chunks:
                output_placeholder.error('No text found to analyze.')
                st.stop()

            idx = build_tfidf_index(chunks)
            query = f'{task}\n{context}'.strip()

            # Cap k to the number of chunks so we never request more than exists
            k = 2 if is_summary else min(top_k, len(chunks))

            top = retrieve_top_k(idx, query=query, k=k)
            retrieved_text = '\n\n'.join([f'[CHUNK {i}]\n{txt}' for i, _, txt in top])

        # inference
        with st.spinner('Running local model...'):
            try:
                out = ai.analyze_finance_doc(
                    doc_text=retrieved_text,
                    doc_type=doc_type,
                    task=task,
                    context=context or None,
                    max_new_tokens=max_new_tokens,
                )
            except TypeError:
                out = ai.analyze_finance_doc(
                    doc_text=retrieved_text,
                    doc_type=doc_type,
                    task=task,
                    context=context or None,
                )

        dt = time.time() - t0

        output_placeholder.markdown(out)
        meta_placeholder.markdown(
            f'<span class=\'chip\'>chunks: {len(chunks)}</span>'
            f'<span class=\'chip\'>top-k: {k}</span>'
            f'<span class=\'chip\'>time: {dt:.2f}s</span>',
            unsafe_allow_html=True,
        )

        if show_debug:
            with st.expander('Retrieved chunks'):
                for i, score, txt in top:
                    st.markdown(f'**Chunk {i}** (score={score:.3f})')
                    st.text(txt[:1500] + ('...' if len(txt) > 1500 else ''))

# ---------- footer ----------
st.markdown(
    '''
    <div class='divider'></div>
    <div style='opacity:0.75; font-size:0.9rem;'>
      Tip: For long PDFs, retrieval reduces latency by sending only the most relevant sections to the model.
    </div>
    ''',
    unsafe_allow_html=True,
)
