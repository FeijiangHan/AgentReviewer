import argparse
import json
import os

from pipeline.local_review_pipeline import run_local_pdf_review
from role.simulator import SimulationManager


# Optional in-file defaults (prefer env vars)
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""


def load_dotenv_file(filepath: str = '.env'):
    """
    Lightweight .env loader (no extra dependency required).
    Existing environment variables are preserved and not overwritten.
    """
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):].strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_papers_from_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_sample_papers():
    return [
        {'id': 'paper_001', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 7.5},
        {'id': 'paper_002', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 6.8},
        {'id': 'paper_003', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 8.2},
        {'id': 'paper_004', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 5.5},
        {'id': 'paper_005', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 7.0},
        {'id': 'paper_006', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 6.2},
        {'id': 'paper_007', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 8.5},
        {'id': 'paper_008', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 7.8},
        {'id': 'paper_009', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 6.5},
        {'id': 'paper_010', 'url': 'https://openreview.net/forum?id={paper_ID}', 'actual_rating': 7.2},
    ]


def resolve_runtime_config(args):
    provider = (args.provider or os.environ.get('LLM_PROVIDER') or 'gemini').lower()

    if provider == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY') or GEMINI_API_KEY
        model = args.model or os.environ.get('LLM_MODEL') or 'gemini-2.5-flash'
    elif provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY') or OPENAI_API_KEY
        model = args.model or os.environ.get('LLM_MODEL') or 'gpt-4o-mini'
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not api_key:
        raise ValueError(f"API key for provider '{provider}' is not set")

    return provider, api_key, model


def main():
    parser = argparse.ArgumentParser(description='Agent Reviewer core pipeline')
    parser.add_argument('--provider', choices=['gemini', 'openai'], default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--pdf', default=None, help='Local PDF path for Milestone C pipeline')
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--persona-mode', choices=['fixed', 'dynamic'], default='fixed')
    parser.add_argument('--top-k-reviewers', type=int, default=3)
    args = parser.parse_args()

    load_dotenv_file()
    provider, api_key, model = resolve_runtime_config(args)

    print("=" * 60)
    print("Agent Reviewer - Core Pipeline")
    print("=" * 60)
    print(f"Provider: {provider} | Model: {model}")

    # Milestone C: local PDF ingest + parse + review + outputs
    if args.pdf:
        result = run_local_pdf_review(
            pdf_path=args.pdf,
            provider_name=provider,
            api_key=api_key,
            model_name=model,
            persona_mode=args.persona_mode,
            top_k_reviewers=args.top_k_reviewers,
        )
        print("\nLocal PDF review completed.")
        print(json.dumps(result, indent=2))
        return

    paper_list_file = 'papers.json'
    if os.path.exists(paper_list_file):
        print(f"\nLoading papers from {paper_list_file}...")
        paper_list = load_papers_from_file(paper_list_file)
    else:
        print(f"\nNo {paper_list_file} found. Using sample papers...")
        paper_list = create_sample_papers()
        with open(paper_list_file, 'w', encoding='utf-8') as f:
            json.dump(paper_list, f, indent=2)

    manager = SimulationManager.from_config(
        provider_name=provider,
        api_key=api_key,
        model_name=model,
        paper_list=paper_list,
        num_rounds=args.rounds,
        seed=42,
    )

    manager.run_all_experiments(output_path='simulation_results.json')

    print("\nPipeline Completed Successfully!")
    print(f"Results saved to: simulation_results.json")
    print(f"Total records saved: {len(manager.results)}")


if __name__ == "__main__":
    main()
