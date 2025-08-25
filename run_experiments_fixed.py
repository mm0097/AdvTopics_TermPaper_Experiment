#!/usr/bin/env python3
"""
Fixed REALM-Bench experiment runner with working imports.
"""

import asyncio
import argparse
import logging
import sys
import os
import json
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_config(model_type: str) -> Dict[str, Any]:
    """Load model configuration from YAML files."""
    config_path = project_root / "configs" / "models" / f"{model_type}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract the model config, handling the nested structure
    if 'model' in config:
        model_config = config['model'].copy()
    else:
        raise ValueError(f"No 'model' section found in {config_path}")
    
    # Add default timeout and retries if not present
    model_config.setdefault('timeout', 120.0)
    model_config.setdefault('max_retries', 3)
    
    return model_config


# Import REALM-Bench components
try:
    from third_party.realm_bench.evaluation.task_definitions import TASK_DEFINITIONS
    REALM_BENCH_AVAILABLE = True
    logger.info("✓ REALM-Bench evaluation framework loaded successfully")
except ImportError as e:
    logger.error(f"❌ REALM-Bench not available: {e}")
    REALM_BENCH_AVAILABLE = False
    sys.exit(1)

# Import orchestrator components
try:
    from src.agents.orchestrator import MASOrchestrator
    from src.models.registry import create_model_client
    from src.utils.settings import get_settings
    ORCHESTRATOR_AVAILABLE = True
    logger.info("✓ Orchestrator components imported successfully")
except ImportError as e:
    logger.error(f"❌ Orchestrator components not available: {e}")
    ORCHESTRATOR_AVAILABLE = False
    sys.exit(1)


async def run_single_experiment(
    task_id: str,
    seed: int = 42,
    experiment_type: str = "baseline",
    model_type: str = "openai"
) -> Dict[str, Any]:
    """Run a single REALM-Bench experiment."""
    
    logger.info(f"Running {experiment_type} experiment on task {task_id} with seed {seed}")
    start_time = time.time()
    
    try:
        # Get task definition
        if task_id not in TASK_DEFINITIONS:
            raise ValueError(f"Task {task_id} not found in REALM-Bench definitions")
        
        task_def = TASK_DEFINITIONS[task_id]
        
        # Convert to our format
        task_config = {
            "task_id": task_def.task_id,
            "name": task_def.name,
            "description": task_def.description,
            "category": task_def.category.value,
            "goals": [
                {
                    "id": goal.goal_id,
                    "description": goal.description,
                    "weight": goal.weight,
                    "success_criteria": goal.success_criteria
                }
                for goal in task_def.goals
            ],
            "constraints": [
                {
                    "id": constraint.constraint_id,
                    "type": constraint.constraint_type,
                    "description": constraint.description,
                    "parameters": constraint.parameters,
                    "weight": constraint.weight
                }
                for constraint in task_def.constraints
            ],
            "resources": task_def.resources,
            "disruption_scenarios": task_def.disruption_scenarios,
            "evaluation_weights": task_def.evaluation_weights,
            "seed": seed
        }
        
        # Create model and orchestrator - load from config
        model_config = load_model_config(model_type)
        
        model_client = create_model_client(model_config)
        
        orchestrator_config = {
            "agents": {
                "max_agents": 8,
                "max_steps": 100,
                "timeout_seconds": 720  # Increased to 3 minutes to avoid timeouts
            },
            "tools": {
                "code_interpreter_enabled": True
            }
        }
        
        orchestrator = MASOrchestrator(
            model_client=model_client,
            config=orchestrator_config,
            retrieval_enabled=(experiment_type == "rag")
        )
        
        # Run the experiment
        run_id = f"{experiment_type}_{task_id}_{seed}_{int(time.time())}"
        result = await orchestrator.execute_task(task_config, run_id)
        execution_time = time.time() - start_time
        
        # Add experiment metadata
        result.update({
            "experiment_type": experiment_type,
            "task_id": task_id,
            "seed": seed,
            "total_execution_time": execution_time,
            "run_id": run_id
        })
        
        logger.info(f"Experiment completed in {execution_time:.2f}s - Success: {result.get('success', False)}")
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Experiment failed: {str(e)}")
        return {
            "experiment_type": experiment_type,
            "task_id": task_id,
            "seed": seed,
            "success": False,
            "error": str(e),
            "total_execution_time": execution_time
        }


async def run_experiments(
    experiments: List[str],
    tasks: List[str], 
    seeds: List[int],
    output_dir: Path,
    model_config: str
) -> Dict[str, Any]:
    """Run multiple experiments and save results."""
    
    logger.info("🧪 Starting REALM-Bench experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    summary = {
        "experiments": experiments,
        "tasks": tasks,
        "seeds": seeds,
        "total_runs": len(experiments) * len(tasks) * len(seeds),
        "successful_runs": 0,
        "failed_runs": 0,
        "results": [],
        "realm_bench_integration": True,
        "evaluation_framework": "REALM-Bench v2.0 (fixed)"
    }
    
    for experiment_type in experiments:
        for task_id in tasks:
            if task_id not in TASK_DEFINITIONS:
                logger.warning(f"Task {task_id} not found, skipping")
                continue
                
            task_def = TASK_DEFINITIONS[task_id]
            logger.info(f"📋 {task_id}: {task_def.name} ({task_def.category.value})")
            if task_def.disruption_scenarios:
                logger.info(f"🚨 Task has {len(task_def.disruption_scenarios)} disruption scenarios for adaptation testing")
            
            for seed in seeds:
                logger.info(f"Running {experiment_type}/{task_id}/seed_{seed}")
                
                result = await run_single_experiment(task_id, seed, experiment_type, model_config)
                
                all_results.append(result)
                summary["results"].append(result)
                
                if result.get("success", False):
                    summary["successful_runs"] += 1
                    
                    # Show achieved goals and constraints with descriptions
                    achieved_goals = result.get('achieved_goals', [])
                    satisfied_constraints = result.get('satisfied_constraints', [])
                    
                    logger.info(f"✅ Goals achieved: {len(achieved_goals)}/{len(task_def.goals)}")
                    for goal in achieved_goals:
                        if isinstance(goal, dict):
                            logger.info(f"   📈 {goal.get('id', 'unknown')}: {goal.get('description', 'No description')}")
                    
                    logger.info(f"✅ Constraints satisfied: {len(satisfied_constraints)}/{len(task_def.constraints)}")
                    for constraint in satisfied_constraints:
                        if isinstance(constraint, dict):
                            logger.info(f"   ✅ {constraint.get('id', 'unknown')}: {constraint.get('description', 'No description')}")
                else:
                    summary["failed_runs"] += 1
                    logger.info(f"❌ Task failed - expected REALM-Bench goals/constraints:")
                    for goal in task_def.goals:
                        logger.info(f"      📋 {goal.goal_id}: {goal.description}")
                    for constraint in task_def.constraints:
                        logger.info(f"      📋 {constraint.constraint_id}: {constraint.description}")
    
    # Save results
    results_file = output_dir / "realm_bench_detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    summary_file = output_dir / "realm_bench_experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ REALM-Bench evaluation completed: {summary['successful_runs']}/{summary['total_runs']} successful")
    logger.info(f"Results saved to {output_dir}")
    
    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run MAS REALM-Bench experiments (fixed)")
    
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["baseline", "rag"],
        default=["baseline"],
        help="Experiment types to run"
    )
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_DEFINITIONS.keys()),
        default=["P1", "P3", "P6"],
        help=f"Tasks to evaluate. Available: {list(TASK_DEFINITIONS.keys())}"
    )
    
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for experiments"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/experiments_fixed"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "groq"],
        help="Model configuration to use (default: openai)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display configuration
    logger.info("Starting MAS REALM-Bench experiments (fixed version)")
    logger.info(f"REALM-Bench integration: ✅ ACTIVE")
    logger.info(f"Experiments: {args.experiments}")
    logger.info(f"Model Config: {args.model_config}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Available REALM-Bench tasks: {len(TASK_DEFINITIONS)} total")
    
    try:
        summary = asyncio.run(run_experiments(
            args.experiments,
            args.tasks,
            args.seeds,
            args.output_dir,
            args.model_config
        ))
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Evaluation Framework: {summary.get('evaluation_framework', 'Unknown')}")
        print(f"REALM-Bench Integration: ✅ Active")
        print(f"Total runs: {summary['total_runs']}")
        print(f"Successful: {summary['successful_runs']}")
        print(f"Failed: {summary['failed_runs']}")
        print(f"Success rate: {summary['successful_runs']/summary['total_runs']*100:.1f}%")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())