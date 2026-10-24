uv run hf upload your-username/bingi-gpt ./artifacts/hf_model .

uv run hellgpt train --download-defaults --export-format both --output-dir ./artifacts

uv run hellgpt generate "dik en lelijk" "Who is hopping on tonight?" --model-path ./artifacts/hf_model