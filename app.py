from __future__ import annotations

import gradio as gr
import pandas as pd
from pathlib import Path

from src.candidate_io import read_docx_text, read_jsonl
from src.ranker import rank_candidates

SAMPLE_CANDIDATES = Path("data/candidates_sample.jsonl")
JD_PATH = Path("data/job_description.docx")

_jd_default = read_docx_text(JD_PATH) if JD_PATH.exists() else ""


def run_ranker(jd_text: str, top_n: int) -> tuple[pd.DataFrame, str]:
    if not jd_text.strip():
        return pd.DataFrame(), "Please paste a job description."
    if not SAMPLE_CANDIDATES.exists():
        return pd.DataFrame(), "Sample candidate file not found."

    results = rank_candidates(jd_text, lambda: read_jsonl(SAMPLE_CANDIDATES))
    results = results[:top_n]

    rows = []
    for r in results:
        f = r["_features"]
        rows.append({
            "Rank": r["rank"],
            "Candidate ID": r["candidate_id"],
            "Score": f"{r['score']:.4f}",
            "Career Evidence": f"{f.career_evidence_fit:.2f}",
            "Production": f"{f.production_experience_fit:.2f}",
            "Behavioral": f"{f.behavioral_fit:.2f}",
            "Capability": f"{f.capability_match:.2f}",
            "Reasoning": r["reasoning"],
        })

    df = pd.DataFrame(rows)
    status = f"Ranked {len(results)} candidates from a 3,000-candidate sample."
    return df, status


with gr.Blocks(title="AI Candidate Ranker") as demo:
    gr.Markdown("## AI-Powered Candidate Ranker")
    gr.Markdown(
        "Paste a job description and click **Rank** to score candidates. "
        "Running on a 3,000-candidate demo sample (includes the actual top-100)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            jd_input = gr.Textbox(
                label="Job Description",
                value=_jd_default,
                lines=20,
                placeholder="Paste job description here...",
            )
            top_n = gr.Slider(
                label="Number of results to show",
                minimum=5,
                maximum=50,
                value=20,
                step=5,
            )
            run_btn = gr.Button("Rank Candidates", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            output_table = gr.Dataframe(
                label="Top Candidates",
                wrap=True,
            )

    run_btn.click(
        fn=run_ranker,
        inputs=[jd_input, top_n],
        outputs=[output_table, status_box],
    )

if __name__ == "__main__":
    demo.launch()
