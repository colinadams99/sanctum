import json, streamlit as st
from sdk.core import SanctumAI

st.title("Sanctum (local) – Private Wellness Insight")
st.caption("All on-device • No cloud • Demo")

ai = SanctumAI()

default = json.dumps(json.load(open("data/sample_logs.json","r",encoding="utf-8")), indent=2)
text = st.text_area("Paste your (local) wellness log JSON:", value=default, height=220)

if st.button("Generate insight", use_container_width=True):
    try:
        data = json.loads(text)
        out = ai.generate_insight(data)
        st.success("Insight")
        st.write(out)
        st.caption("Generated fully offline.")
    except Exception as e:
        st.error(str(e))
