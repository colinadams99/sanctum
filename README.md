# Sanctum (local) SDK – Portfolio Finance Document Insights
- 100% on-device; no data leaves the machine.
- Small language model (Phi-3 Mini) with quantized GPU inference (bitsandbytes).
- Single call API: generate_insight(user_data_dict)

## Quickstart
conda activate sanctum
pip install -r requirements.txt
streamlit run app_demo/app.py

## Purpose
This was developed for asset managers, hedge funds, and other lenders/borrowers to use in order to analyze portfolio finance documents.
