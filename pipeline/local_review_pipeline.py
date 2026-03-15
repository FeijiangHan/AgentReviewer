import json
import os
from typing import Dict, Any

from role.simulator import SimulationManager
from .ingest import ingest_local_pdf


def run_local_pdf_review(
    pdf_path: str,
    provider_name: str,
    api_key: str,
    model_name: str,
    output_dir: str = 'outputs',
) -> Dict[str, Any]:
    paper = ingest_local_pdf(pdf_path)
    paper_payload = {
        'id': paper.paper_id,
        'source': paper.source,
        'content': paper.full_text,
        'actual_rating': None,
    }

    manager = SimulationManager.from_config(
        provider_name=provider_name,
        api_key=api_key,
        model_name=model_name,
        paper_list=[paper_payload],
        num_rounds=1,
        seed=42,
    )
    manager.num_papers_per_round = 1
    manager.run_all_experiments(output_path='simulation_results.json')

    os.makedirs(output_dir, exist_ok=True)
    paper_out_dir = os.path.join(output_dir, paper.paper_id)
    os.makedirs(paper_out_dir, exist_ok=True)

    reviews_json_path = os.path.join(paper_out_dir, 'reviews.json')
    with open(reviews_json_path, 'w', encoding='utf-8') as f:
        json.dump(manager.results, f, indent=2)

    report_md_path = os.path.join(paper_out_dir, 'report.md')
    first_result = manager.results[0] if manager.results else {}
    decision = first_result.get('ac_decision', {}).get('final_decision', 'Unknown')
    justification = first_result.get('ac_decision', {}).get('final_justification', '')
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Review Report: {paper.paper_id}\n\n")
        f.write(f"- Source: `{paper.source}`\n")
        f.write(f"- Provider: `{provider_name}`\n")
        f.write(f"- Model: `{model_name}`\n")
        f.write(f"- Final Decision: **{decision}**\n\n")
        f.write("## AC Justification\n\n")
        f.write(f"{justification}\n")

    return {
        'paper_id': paper.paper_id,
        'references_count': len(paper.references),
        'reviews_json': reviews_json_path,
        'report_md': report_md_path,
    }
