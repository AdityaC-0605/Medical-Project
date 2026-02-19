# RAGAS Evaluation

This folder adds evaluation metrics for the Medical AI project using RAGAS.

## Dataset format (`eval_cases.jsonl`)
Each line is one JSON object with:
- `question` (required): clinical input string
- `contexts` (optional): list of retrieved context chunks
- `answer` (optional): model answer to evaluate
- `ground_truth` (optional): reference answer for correctness scoring
- `metadata` (optional): metadata passed to preprocessing

If `contexts` is missing, the runner retrieves top chunks from `data/knowledge_base`.
If `answer` is missing, run with `--generate-answers`.

## Run

Set your key in `/Users/aditya/Documents/Medical-Project/.env`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

(`.env` is git-ignored.)

Then run:

```bash
python main.py --mode eval --generate-answers
```

Recommended workflow (fast iteration):

```bash
# 1) Slow step: generate answers once
python main.py --mode eval --generate-only --max-cases 12 --generation-timeout-sec 300 --eval-max-new-tokens 96 --generated-output evaluation/results/generated_answers.jsonl

# 2) Fast step: score repeatedly on the saved answers
python main.py --mode eval --score-only --eval-dataset evaluation/results/generated_answers.jsonl --eval-output evaluation/results/my_report.json
```

Useful options:

```bash
python main.py --mode eval --dry-run
python main.py --mode eval --max-cases 5 --generate-answers
python main.py --mode eval --eval-output evaluation/results/my_report.json --generate-answers
python main.py --mode eval --generate-only --eval-max-new-tokens 96
python main.py --mode eval --score-only --eval-dataset evaluation/results/generated_answers.jsonl
```

## Output

A JSON report is written to `evaluation/results/` containing:
- aggregate metric scores
- per-case metric scores
- dataset metadata

## Notes

- RAGAS requires evaluator model access. Ensure provider credentials are configured (for example `OPENAI_API_KEY`) if your RAGAS setup uses OpenAI-based evaluators.
- Install dependencies with:

```bash
pip install -r requirements.txt
```
