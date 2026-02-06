
import argparse
import os
import logging
from pathlib import Path
from datetime import datetime

from src.utils import METHODS, TECHNIQUES
# Import MSP modules directly
from src.memzero_multi_speaker.add import MemoryADD
from src.memzero_multi_speaker.search_async import MemorySearch

class Experiment:
    def __init__(self, technique_type, chunk_size):
        self.technique_type = technique_type
        self.chunk_size = chunk_size

    def run(self):
        print(f"Running experiment with technique: {self.technique_type}, chunk size: {self.chunk_size}")


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments (Global Observer MSP Mode)")
    parser.add_argument("--technique_type", choices=TECHNIQUES, default="mem0", help="Memory technique to use")
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to use (add or search)")
    parser.add_argument("--mode", type=str, default="msp_global", help="Mode identifier (defaults to msp_global)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output_folder", type=str, default="results/msp", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to process")
    parser.add_argument("--figure_view", action="store_true", default=False, help="Whether to include figure view in memory")
    parser.add_argument("--embedder_model", type=str, default="Pro/BAAI/bge-m3", help="Embedding model name for the embedder")
    parser.add_argument("--qdrant_path", type=str, default="./qdrant_data/tmp_msp", help="Path for the Qdrant vector store")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="locomo10_failed",
        help="Dataset name or path (supports .json / .jsonl)",
    )
    parser.add_argument("--workspace_dir", type=str, default=".", help="Directory for all experiment outputs including logs.")
    
    # Defaults for MSP Global Observer
    parser.add_argument("--fact_extraction_mode", type=str, default="15_MSP", help="Fact extraction prompt mode (supported: 0, 15_MSP)")
    parser.add_argument("--memory_decision_mode", type=str, default="147_MSP", help="Memory decision prompt mode (supported: 0, 147_MSP, 149_MSP)")
    parser.add_argument("--search_mode", type=str, default="5", help="Search mode identifier (supported: 0, 5, 6, 1427)")
    parser.add_argument("--answer_mode", type=str, default="0", help="Answer prompt mode (use 0 or 15 for MSP prompts)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--collection_name", type=str, default="global_memories", help="Override Qdrant collection name")
    
    # LLM Configs
    parser.add_argument("--llm_model", type=str, default=None, help="Model name for LLM calls")
    parser.add_argument("--llm_base_url", type=str, default=None, help="Base URL for LLM provider API")
    parser.add_argument("--llm_api_key", type=str, default=None, help="API key for LLM provider")
    parser.add_argument("--search_llm_model", type=str, default=None, help="Model name for memory search operations")
    parser.add_argument("--search_llm_base_url", type=str, default=None, help="Base URL for the search LLM provider")
    parser.add_argument("--search_llm_api_key", type=str, default=None, help="API key for the search LLM provider")
    parser.add_argument("--answer_llm_model", type=str, default=None, help="Model name for answer generation")
    parser.add_argument("--answer_llm_base_url", type=str, default=None, help="Base URL for the answer LLM provider")
    parser.add_argument("--answer_llm_api_key", type=str, default=None, help="API key for the answer LLM provider")
    parser.add_argument("--embedder_base_url", type=str, default=None, help="Base URL for embedder provider API")
    parser.add_argument("--embedder_api_key", type=str, default=None, help="API key for embedder provider")
    parser.add_argument("--embedder_dims", type=int, default=None, help="Output dimensionality for the embedder model")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for MemoryADD ingestion")
    parser.add_argument("--add_mode", type=str, default="0", help="Memory ingestion overlap mode")

    args = parser.parse_args()

    def build_provider_config(model_value, base_url_value, api_key_value, optional_fields=None):
        config = {}
        if model_value:
            config["model"] = model_value
        if base_url_value:
            config["base_url"] = base_url_value
        if api_key_value:
            config["api_key"] = api_key_value
        if optional_fields:
            for key, value in optional_fields.items():
                if value not in (None, ""):
                    config[key] = value
        return config

    # allowed_fact_modes = {"0", "15_MSP"}
    # allowed_mem_modes = {"0", "147_MSP", "149_MSP"}
    # allowed_answer_modes = {"0", "15"}
    # allowed_search_modes = {"0", "5", "6", "1427", "14.27"}
    # if args.fact_extraction_mode not in allowed_fact_modes:
    #     raise ValueError(f"fact_extraction_mode must be one of {sorted(allowed_fact_modes)}")
    # if args.memory_decision_mode not in allowed_mem_modes:
    #     raise ValueError(f"memory_decision_mode must be one of {sorted(allowed_mem_modes)}")
    # if args.answer_mode not in allowed_answer_modes:
    #     raise ValueError(f"answer_mode must be one of {sorted(allowed_answer_modes)}")
    # if args.search_mode not in allowed_search_modes:
    #     raise ValueError(f"search_mode must be one of {sorted(allowed_search_modes)}")

    add_llm_config = build_provider_config(args.llm_model, args.llm_base_url, args.llm_api_key)
    search_llm_model = args.search_llm_model or args.llm_model
    search_llm_base_url = args.search_llm_base_url or args.llm_base_url
    search_llm_api_key = args.search_llm_api_key or args.llm_api_key
    search_llm_config = build_provider_config(search_llm_model, search_llm_base_url, search_llm_api_key)
    answer_llm_model = args.answer_llm_model or args.search_llm_model or args.llm_model
    answer_llm_base_url = args.answer_llm_base_url or args.search_llm_base_url or args.llm_base_url
    answer_llm_api_key = args.answer_llm_api_key or args.search_llm_api_key or args.llm_api_key
    answer_llm_config = build_provider_config(answer_llm_model, answer_llm_base_url, answer_llm_api_key)
    embedder_config = build_provider_config(
        args.embedder_model,
        args.embedder_base_url,
        args.embedder_api_key,
        {"embedding_dims": args.embedder_dims},
    )

    os.makedirs(args.workspace_dir, exist_ok=True)
    if not os.environ.get("MEM0_DIR"):
        derived_mem0_dir = os.path.join(args.workspace_dir, ".mem0_state_msp")
        os.makedirs(derived_mem0_dir, exist_ok=True)
        os.environ["MEM0_DIR"] = derived_mem0_dir

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f"MSP_{timestamp}_{args.technique_type}_{args.method}"
    if args.method == "search":
        log_file_name += f"_topK{args.top_k}_mode{args.search_mode}"

    log_file_path = os.path.join(args.workspace_dir, f"{log_file_name}.log")

    log_formatter = logging.Formatter(
        '%(asctime)s - %(threadName)s - [%(levelname)s] - %(message)s'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    print("="*80)
    print(f"📝 Logging MSP Experiment to: {log_file_path}")
    print("="*80)
    print(f"Running MSP Global Observer Experiment: {args.method}, Search Mode: {args.search_mode}")

    def _convert_jsonl_to_json(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            fout.write("[\n")
            first = True
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if not first:
                    fout.write(",\n")
                fout.write(line)
                first = False
            fout.write("\n]\n")

    def _resolve_dataset_path(dataset_name: str) -> tuple[Path, str]:
        script_dir = Path(__file__).resolve().parent
        dataset_dir = script_dir / "dataset"

        name_path = Path(dataset_name)
        if name_path.suffix in {".json", ".jsonl"}:
            if name_path.is_absolute() and name_path.exists():
                resolved = name_path
            elif name_path.exists():
                resolved = name_path.resolve()
            else:
                candidate = dataset_dir / name_path.name
                resolved = candidate if candidate.exists() else name_path
        else:
            json_path = dataset_dir / f"{dataset_name}.json"
            jsonl_path = dataset_dir / f"{dataset_name}.jsonl"
            if json_path.exists():
                resolved = json_path
            elif jsonl_path.exists():
                resolved = jsonl_path
            else:
                candidate = dataset_dir / dataset_name
                resolved = candidate if candidate.exists() else json_path

        dataset_tag = resolved.stem
        if resolved.suffix == ".jsonl":
            converted = dataset_dir / f"{resolved.stem}__jsonl.json"
            needs_convert = True
            if converted.exists() and resolved.exists():
                try:
                    needs_convert = resolved.stat().st_mtime > converted.stat().st_mtime
                except OSError:
                    needs_convert = True
            if needs_convert and resolved.exists():
                print(f"🔄 Converting JSONL to JSON: {resolved} -> {converted}")
                _convert_jsonl_to_json(resolved, converted)
            resolved = converted
        return resolved, dataset_tag

    dataset_path, dataset_tag = _resolve_dataset_path(args.dataset_name)

    if args.technique_type == "mem0":
        # Force MSP Global modules
        if args.method == "add":
            memory_manager = MemoryADD(
                data_path=str(dataset_path),
                batch_size=args.batch_size,
                is_graph=args.is_graph, 
                logger=logger,
                figure_view=args.figure_view, 
                qdrant_path=args.qdrant_path,
                fact_extraction_mode=args.fact_extraction_mode,
                memory_decision_mode=args.memory_decision_mode,
                add_mode=args.add_mode,
                collection_name=args.collection_name,
                llm_config=add_llm_config,
                embedder_config=embedder_config,
            )
            try:
                memory_manager.process_all_conversations(max_workers=args.max_workers)
            finally:
                if hasattr(memory_manager, "close"):
                    memory_manager.close()
        elif args.method == "search":
            output_file_path = os.path.join(
                args.output_folder,
                f"msp_{dataset_tag}_results_top_{args.top_k}_mode_{args.search_mode}.json",
            )
            os.makedirs(args.output_folder, exist_ok=True)
            
            memory_searcher = MemorySearch(
                output_file_path,
                args.top_k,
                args.filter_memories,
                args.is_graph,
                logger=logger,
                qdrant_path=args.qdrant_path,
                search_method=args.search_mode,
                answer_mode=args.answer_mode,
                collection_name=args.collection_name,
                llm_config=search_llm_config,
                answer_llm_config=answer_llm_config,
                embedder_config=embedder_config,
            )
            try:
                memory_searcher.process_data_file(
                    str(dataset_path), max_workers=args.max_workers
                )
            finally:
                if hasattr(memory_searcher, "close"):
                    memory_searcher.close()
    else:
        print(f"Technique {args.technique_type} not supported in MSP mode.")

if __name__ == "__main__":
    main()
