# Sanctum (local) SDK – Private Wellness Insights
- 100% on-device; no data leaves the machine.
- Small language model (Phi-3 Mini) with quantized GPU inference (bitsandbytes).
- Single call API: generate_insight(user_data_dict)

## Quickstart
conda activate sanctum
pip install -r requirements.txt
streamlit run app_demo/app.py
