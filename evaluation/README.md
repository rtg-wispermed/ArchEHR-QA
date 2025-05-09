# Evaluation

This directory contains the scoring script for the ArchEHR-QA Shared Task.

## Setup

The script is tested using Python 3.8.18.

1. Install the required dependencies.

```bash
$ pip install --no-cache-dir -r requirements.txt
```

2. Generate the QuickUMLS data files following the directions provided at https://github.com/Georgetown-IR-Lab/QuickUMLS (this requires a UMLS license).

> The path to the QuickUMLS data files is used in the scoring script as `quickumls_path`.

3. Run the scoring script.

```bash
python scoring.py \
    --submission_path submission.json \
    --key_path archehr-qa_key.json \
    --data_path archehr-qa.xml \
    --quickumls_path quickumls/ \
    --out_file_path scores.json
```

If you have any questions, please reach out to sarvesh.soni@nih.gov.