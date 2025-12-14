from pathlib import Path
import sys

import streamlit as st

# make sure the project root is on the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sdk.core import SanctumAI  # noqa: E402
from sdk.doc_loader import load_pdf_text  # noqa: E402


st.title('Sanctum – Locally Run Portfolio Finance Doc Assistant')
st.caption('All inference is local. No documents leave this machine.')


@st.cache_resource
def get_ai() -> SanctumAI:
    return SanctumAI()


ai = get_ai()

mode = st.radio('Input type', ['Paste text', 'Upload PDF'], horizontal = True)

# initialize doc text
doc_text = ''


if mode == 'Paste text':
    doc_text = st.text_area(
        'Paste document text (e.g., term sheet, NAV facility, LPA excerpt):',
        height = 300,
    )
else:
    pdf_file = st.file_uploader('Upload a PDF', type = ['pdf'])
    if pdf_file is not None:
        tmp_path = Path('tmp_upload.pdf')
        tmp_path.write_bytes(pdf_file.read())
        doc_text = load_pdf_text(tmp_path)

doc_type = st.selectbox(
    'Document type',
    ["term_sheet", "credit_agreement", "LPA", "memo", "nav_report", "other"],
)

# sets what the task is
task = st.text_input(
    'Task',
    value=(
        'Summarize key terms, collateral, cross-collateralization, structural '
        'subordination, and major risks for a portfolio finance or insurer client.'
    ),
)

# sets the context and audience (portfolio finance teams, etc.)
context = st.text_input(
    'Context (optional)',
    value = 'Audience: portfolio finance team at an insurer or asset manager.',
)

# button
run = st.button('Analyze document', use_container_width = True)

# if button is pressed
if run:
    if not doc_text.strip():
        # error if there's no doc to strip the text
        st.error("Please paste text or upload a PDF first.")
    else:
        with st.spinner("Running local model..."):
            out = ai.analyze_finance_doc(
                doc_text = doc_text,
                doc_type = doc_type,
                task = task,
                context = context or None,
            )
        st.markdown("### Output")
        st.write(out)
        st.caption("Generated completely offline with a local small language model.")
