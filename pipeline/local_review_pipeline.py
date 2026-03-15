import json
import os
from typing import Dict, Any, List

from llm import create_provider
from persona import DynamicPersonaPipeline
from persona.models import PersonaCard
from role.area_chair import AreaChairAgent
from role.reviewer import LLMAgentReviewer
from role.simulator import SimulationManager

from .ingest import ingest_local_pdf


def _write_outputs(
    output_dir: str,
    paper_id: str,
    provider_name: str,
    model_name: str,
    source: str,
    results: List[Dict[str, Any]],
    references_count: int,
    persona_cards: List[Dict[str, Any]] | None = None,
    dynamic_trace: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    paper_out_dir = os.path.join(output_dir, paper_id)
    os.makedirs(paper_out_dir, exist_ok=True)

    reviews_json_path = os.path.join(paper_out_dir, 'reviews.json')
    with open(reviews_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    persona_cards_path = None
    if persona_cards is not None:
        persona_cards_path = os.path.join(paper_out_dir, 'persona_cards.json')
        with open(persona_cards_path, 'w', encoding='utf-8') as f:
            json.dump(persona_cards, f, indent=2)

    dynamic_trace_path = None
    if dynamic_trace is not None:
        dynamic_trace_path = os.path.join(paper_out_dir, 'dynamic_persona_trace.json')
        with open(dynamic_trace_path, 'w', encoding='utf-8') as f:
            json.dump(dynamic_trace, f, indent=2, ensure_ascii=False)

    first_result = results[0] if results else {}
    decision = first_result.get('ac_decision', {}).get('final_decision', 'Unknown')
    justification = first_result.get('ac_decision', {}).get('final_justification', '')

    report_md_path = os.path.join(paper_out_dir, 'report.md')
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Review Report: {paper_id}\n\n")
        f.write(f"- Source: `{source}`\n")
        f.write(f"- Provider: `{provider_name}`\n")
        f.write(f"- Model: `{model_name}`\n")
        f.write(f"- Final Decision: **{decision}**\n")
        f.write(f"- References Parsed: {references_count}\n\n")
        f.write("## AC Justification\n\n")
        f.write(f"{justification}\n")

    output = {
        'paper_id': paper_id,
        'references_count': references_count,
        'reviews_json': reviews_json_path,
        'report_md': report_md_path,
    }
    if persona_cards_path:
        output['persona_cards_json'] = persona_cards_path
    if dynamic_trace_path:
        output['dynamic_trace_json'] = dynamic_trace_path
    return output


def _run_dynamic_persona_review(
    paper_payload: Dict[str, Any],
    provider_name: str,
    api_key: str,
    model_name: str,
    top_k: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    provider = create_provider(provider_name=provider_name, api_key=api_key, model_name=model_name)
    use_llm_search = os.getenv('DYNAMIC_PERSONA_USE_LLM_SEARCH', '0').lower() in {'1', 'true', 'yes'}
    pipeline = DynamicPersonaPipeline(top_k=top_k, provider=provider, use_llm_search=use_llm_search)
    persona_cards, trace = pipeline.run_with_trace(paper_payload['content'], paper_payload.get('references', []))

    fallback_needed = len(persona_cards) < top_k
    if fallback_needed:
        start_idx = len(persona_cards) + 1
        for idx in range(start_idx, top_k + 1):
            persona_cards.append(
                PersonaCard(
                    name='Generic Dynamic Reviewer' if idx == 1 else f'Generic Dynamic Reviewer #{idx}',
                    affiliation='Unknown',
                    research_areas=['machine learning'],
                    methodological_preferences=['clear empirical validation'],
                    common_concerns=['novelty', 'rigor', 'clarity'],
                    style_signature='balanced, technical, evidence-focused',
                    potential_biases=['none explicitly inferred'],
                    confidence=0.3,
                    evidence_sources=['fallback persona: insufficient validated persona candidates'],
                )
            )

    trace['selected_reviewers'] = [
        {
            'name': c.name,
            'affiliation': c.affiliation,
            'research_areas': c.research_areas,
            'confidence': c.confidence,
            'evidence_sources': c.evidence_sources,
            'is_fallback': c.name.startswith('Generic Dynamic Reviewer'),
        }
        for c in persona_cards
    ]
    trace['fallback_used'] = fallback_needed
    trace['candidate_stats']['final_selected_count'] = len(persona_cards)
    trace['paper'] = {
        'paper_id': paper_payload.get('id', ''),
        'source': paper_payload.get('source', ''),
        'content': paper_payload.get('content', ''),
    }

    reviewers = {}
    for idx, card in enumerate(persona_cards, start=1):
        reviewers[idx] = LLMAgentReviewer(
            reviewer_id=idx,
            persona=f"dynamic_{idx}",
            provider=provider,
            persona_prompt=card.to_prompt(),
        )

    stage1_reviews: Dict[int, Dict[str, Any]] = {}
    for rid, reviewer in reviewers.items():
        stage1_reviews[rid] = reviewer.generate_review_stage_1(paper_payload['content'], paper_label=paper_payload['source'])

    stage2_reviews: Dict[int, Dict[str, Any]] = {}
    for rid, reviewer in reviewers.items():
        other_reviews = {k: v for k, v in stage1_reviews.items() if k != rid}
        stage2_reviews[rid] = reviewer.generate_review_stage_2(
            paper_payload['content'],
            stage1_reviews[rid],
            other_reviews,
            paper_label=paper_payload['source'],
        )

    ac = AreaChairAgent(provider=provider)
    ac_decision = ac.make_final_decision(paper_payload['id'], stage2_reviews)

    results = [{
        'round': 1,
        'paper_id': paper_payload['id'],
        'actual_rating': paper_payload.get('actual_rating'),
        'ac_decision': ac_decision,
        'reviews': {rid: {**stage1_reviews[rid], **stage2_reviews[rid]} for rid in reviewers.keys()},
        'runtime_metadata': {
            'llm_provider': provider.provider_name,
            'review_model': provider.model_name,
            'ac_model': provider.model_name,
            'reviewer_ids': list(reviewers.keys()),
            'persona_mode': 'dynamic',
        },
    }]

    persona_cards_json = [
        {
            'name': c.name,
            'affiliation': c.affiliation,
            'research_areas': c.research_areas,
            'methodological_preferences': c.methodological_preferences,
            'common_concerns': c.common_concerns,
            'style_signature': c.style_signature,
            'potential_biases': c.potential_biases,
            'confidence': c.confidence,
            'evidence_sources': c.evidence_sources,
        }
        for c in persona_cards
    ]

    return results, persona_cards_json, trace


def run_local_pdf_review(
    pdf_path: str,
    provider_name: str,
    api_key: str,
    model_name: str,
    output_dir: str = 'outputs',
    persona_mode: str = 'fixed',
    top_k_reviewers: int = 3,
) -> Dict[str, Any]:
    paper = ingest_local_pdf(pdf_path)
    paper_payload = {
        'id': paper.paper_id,
        'source': paper.source,
        'content': paper.full_text,
        'references': paper.references,
        'actual_rating': None,
    }

    if persona_mode == 'dynamic':
        results, persona_cards, trace = _run_dynamic_persona_review(
            paper_payload=paper_payload,
            provider_name=provider_name,
            api_key=api_key,
            model_name=model_name,
            top_k=top_k_reviewers,
        )
        return _write_outputs(
            output_dir=output_dir,
            paper_id=paper.paper_id,
            provider_name=provider_name,
            model_name=model_name,
            source=paper.source,
            results=results,
            references_count=len(paper.references),
            persona_cards=persona_cards,
            dynamic_trace=trace,
        )

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

    return _write_outputs(
        output_dir=output_dir,
        paper_id=paper.paper_id,
        provider_name=provider_name,
        model_name=model_name,
        source=paper.source,
        results=manager.results,
        references_count=len(paper.references),
        persona_cards=None,
    )
