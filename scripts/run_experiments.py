"""
Main experiment runner for MAS REALM-Bench evaluation.
Uses the real REALM-Bench evaluation framework with proper task definitions.
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add paths for REALM-Bench integration
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
realm_bench_path = str(project_root / "third_party" / "realm_bench")
if realm_bench_path not in sys.path:
    sys.path.insert(0, realm_bench_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import REALM-Bench components
try:
    from evaluation.evaluator import BenchmarkEvaluator, EvaluationConfig
    from evaluation.task_definitions import TASK_DEFINITIONS, TaskDefinition
    from evaluation.framework_runners_mas import get_mas_framework_runners
    REALM_BENCH_AVAILABLE = True
    logger.info("✓ REALM-Bench evaluation framework loaded successfully")
except ImportError as e:
    logger.warning(f"❌ REALM-Bench not available: {e}")
    logger.warning("Falling back to legacy sample tasks")
    REALM_BENCH_AVAILABLE = False

# Import orchestrator components at module level to avoid path issues
MASOrchestrator = None
create_model_client = None
get_settings = None
ORCHESTRATOR_AVAILABLE = False

def _import_orchestrator_components():
    """Import orchestrator components with multiple fallback strategies."""
    global MASOrchestrator, create_model_client, get_settings, ORCHESTRATOR_AVAILABLE
    
    import sys
    from pathlib import Path
    
    print(f"DEBUG: Starting import strategies...")
    print(f"DEBUG: Current sys.path: {sys.path[:3]}")
    
    # Strategy 3: Direct path manipulation (most reliable)
    try:
        print("DEBUG: Trying direct path strategy")
        
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        src_dir = project_root / "src"
        
        print(f"DEBUG: Script dir: {script_dir}")
        print(f"DEBUG: Project root: {project_root}")
        print(f"DEBUG: Src dir: {src_dir}")
        
        # Add both project root and src to path (orchestrator imports use "src.agents.types")
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            print(f"DEBUG: Added {project_root} to sys.path")
        
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            print(f"DEBUG: Added {src_dir} to sys.path")
        
        from agents.orchestrator import MASOrchestrator as _MASOrchestrator
        from models.registry import create_model_client as _create_model_client
        from utils.settings import get_settings as _get_settings
        
        MASOrchestrator = _MASOrchestrator
        create_model_client = _create_model_client
        get_settings = _get_settings
        
        ORCHESTRATOR_AVAILABLE = True
        print("DEBUG: Direct path strategy succeeded!")
        logger.info("✓ Orchestrator components imported successfully")
        return
        
    except ImportError as e:
        print(f"DEBUG: Direct path strategy failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.warning("⚠️ Orchestrator components not available with any strategy")
    ORCHESTRATOR_AVAILABLE = False

def __import_with_strategy(orch_module, registry_module, settings_module):
    """Import with specific module names."""
    global MASOrchestrator, create_model_client, get_settings
    
    orch_mod = __import__(orch_module, fromlist=['MASOrchestrator'])
    registry_mod = __import__(registry_module, fromlist=['create_model_client'])
    settings_mod = __import__(settings_module, fromlist=['get_settings'])
    
    MASOrchestrator = orch_mod.MASOrchestrator
    create_model_client = registry_mod.create_model_client
    get_settings = settings_mod.get_settings

def __import_relative_strategy():
    """Import using relative paths."""
    global MASOrchestrator, create_model_client, get_settings
    
    # Add src to path if not already there
    import sys
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    from agents.orchestrator import MASOrchestrator as _MASOrchestrator
    from models.registry import create_model_client as _create_model_client
    from utils.settings import get_settings as _get_settings
    
    MASOrchestrator = _MASOrchestrator
    create_model_client = _create_model_client
    get_settings = _get_settings

# Import components at module load time
_import_orchestrator_components()

# Legacy fallback imports
if not REALM_BENCH_AVAILABLE:
    try:
        from src.realm.adapter import create_baseline_adapter, create_rag_adapter
    except ImportError as e:
        logger.error(f"Neither REALM-Bench nor legacy adapters available: {e}")
        sys.exit(1)


def run_simple_realm_bench_experiments(
    experiments: List[str],
    tasks: List[str],
    seeds: List[int],
    output_dir: Path
) -> Dict[str, Any]:
    """Run experiments using the real REALM-Bench tasks with direct orchestrator calls."""
    
    logger.info("🧪 Starting REALM-Bench evaluation with real task definitions")
    
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
        "evaluation_framework": "REALM-Bench v2.0 (simplified)"
    }
    
    for experiment_type in experiments:
        # Only support baseline for now since that's what's implemented
        if experiment_type != "baseline":
            logger.warning(f"Only baseline experiments supported currently, skipping {experiment_type}")
            continue
            
        for task_id in tasks:
            if task_id not in TASK_DEFINITIONS:
                logger.warning(f"Task {task_id} not found in REALM-Bench tasks, skipping")
                continue
            
            # Convert REALM-Bench task definition to our format
            task_def = TASK_DEFINITIONS[task_id]
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
                "evaluation_weights": task_def.evaluation_weights
            }
            
            for seed in seeds:
                logger.info(f"Running {experiment_type}/{task_id}/seed_{seed}")
                
                result = run_single_direct_experiment(
                    task_id,
                    task_config,
                    seed
                )
                
                # Enhance result with REALM-Bench metadata
                result.update({
                    "realm_bench_task": True,
                    "task_category": task_def.category.value,
                    "task_name": task_def.name,
                    "total_goals": len(task_def.goals),
                    "total_constraints": len(task_def.constraints),
                    "has_disruptions": len(task_def.disruption_scenarios) > 0
                })
                
                all_results.append(result)
                summary["results"].append(result)
                
                if result.get("success", False):
                    summary["successful_runs"] += 1
                    
                    # Log with actual REALM-Bench goal/constraint descriptions
                    achieved_goals = result.get('achieved_goals', [])
                    satisfied_constraints = result.get('satisfied_constraints', [])
                    
                    logger.info(f"✅ Goals achieved: {len(achieved_goals)}/{len(task_def.goals)}")
                    logger.info(f"✅ Constraints satisfied: {len(satisfied_constraints)}/{len(task_def.constraints)}")
                    
                    # Show actual REALM-Bench goal descriptions
                    for goal_entry in achieved_goals:
                        if isinstance(goal_entry, dict) and 'id' in goal_entry:
                            goal_id = goal_entry['id']
                            goal = next((g for g in task_def.goals if g.goal_id == goal_id), None)
                            if goal:
                                logger.info(f"   📈 {goal_id}: {goal.description}")
                        elif isinstance(goal_entry, str):
                            goal = next((g for g in task_def.goals if g.goal_id == goal_entry), None)
                            if goal:
                                logger.info(f"   📈 {goal_entry}: {goal.description}")
                    
                    # Show actual REALM-Bench constraint descriptions
                    for constraint_entry in satisfied_constraints:
                        if isinstance(constraint_entry, dict) and 'id' in constraint_entry:
                            constraint_id = constraint_entry['id']
                            constraint = next((c for c in task_def.constraints if c.constraint_id == constraint_id), None)
                            if constraint:
                                logger.info(f"   ✅ {constraint_id}: {constraint.description}")
                        elif isinstance(constraint_entry, str):
                            constraint = next((c for c in task_def.constraints if c.constraint_id == constraint_entry), None)
                            if constraint:
                                logger.info(f"   ✅ {constraint_entry}: {constraint.description}")
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


def run_legacy_experiments(
    experiments: List[str],
    tasks: List[str],
    seeds: List[int],
    output_dir: Path
) -> Dict[str, Any]:
    """Fallback to legacy sample task experiments."""
    
    logger.warning("🔄 Falling back to legacy sample task experiments")
    
    # Legacy sample tasks (simplified subset)
    SAMPLE_TASKS = {
        "P1": {
            "description": "Planning visit waypoints within time windows under spatial constraints",
            "goals": [
                {"id": "goal_1", "description": "Visit all required campus locations"},
                {"id": "goal_2", "description": "Minimize total travel time"},
                {"id": "goal_3", "description": "Respect all time window constraints"}
            ],
            "constraints": [
                {"id": "constraint_1", "description": "Each location has specific time windows"},
                {"id": "constraint_2", "description": "Travel time between locations"},
                {"id": "constraint_3", "description": "Total tour must complete within time limit"}
            ],
            "resources": {
                "locations": ["library", "cafeteria", "gym", "student_center"],
                "time_windows": {
                    "library": {"open": "08:00", "close": "22:00"},
                    "cafeteria": {"open": "07:00", "close": "21:00"},
                    "gym": {"open": "06:00", "close": "23:00"},
                    "student_center": {"open": "08:00", "close": "24:00"}
                },
                "max_tour_time": 480,
                "start_location": "entrance"
            }
        }
    }
    
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
        "realm_bench_integration": False,
        "evaluation_framework": "Legacy Sample Tasks"
    }
    
    for experiment_type in experiments:
        for task_id in tasks:
            if task_id not in SAMPLE_TASKS:
                logger.warning(f"Task {task_id} not found in sample tasks, skipping")
                continue
            
            task_config = SAMPLE_TASKS[task_id]
            
            for seed in seeds:
                logger.info(f"Running {experiment_type}/{task_id}/seed_{seed}")
                
                result = run_single_legacy_experiment(
                    experiment_type,
                    task_id,
                    task_config,
                    seed
                )
                
                all_results.append(result)
                summary["results"].append(result)
                
                if result.get("success", False):
                    summary["successful_runs"] += 1
                else:
                    summary["failed_runs"] += 1
    
    # Save results
    results_file = output_dir / "legacy_detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    summary_file = output_dir / "legacy_experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Legacy results saved to {output_dir}")
    return summary


async def run_single_direct_experiment_async(
    task_id: str,
    task_config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Run a single experiment directly with the orchestrator setup (async version)."""
    
    logger.info(f"Running direct baseline experiment on task {task_id} with seed {seed}")
    start_time = time.time()
    
    try:
        # Check if orchestrator components are available
        if not ORCHESTRATOR_AVAILABLE:
            raise RuntimeError("Orchestrator components not available")
        
        # Use GPT-5 configuration (the one that's working)
        model_config = {
            "provider": "openai",
            "name": "gpt-5-mini",
            "max_tokens": 4096,
            "timeout": 120.0,
            "max_retries": 3
        }
        
        # Create model client
        model_client = create_model_client(model_config)
        
        # Create orchestrator configuration
        orchestrator_config = {
            "agents": {
                "max_agents": 8,  # Slightly reduced for stability
                "max_steps": 100,  # Increased for complex tasks
                "timeout_seconds": 120  # Reduced timeout for faster testing
            },
            "tools": {
                "code_interpreter_enabled": True
            }
        }
        
        # Initialize orchestrator (baseline - no retrieval)
        orchestrator = MASOrchestrator(
            model_client=model_client,
            config=orchestrator_config,
            retrieval_enabled=False
        )
        
        # Add seed to task config
        task_config_with_seed = task_config.copy()
        task_config_with_seed["seed"] = seed
        
        # Generate run ID
        import random
        random.seed(seed)
        run_id = f"baseline_realm_{task_id}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Run task with orchestrator
        result = await orchestrator.execute_task(task_config_with_seed, run_id)
        execution_time = time.time() - start_time
        
        # Add experiment metadata
        result.update({
            "experiment_type": "baseline",
            "task_id": task_id,
            "seed": seed,
            "total_execution_time": execution_time,
            "run_id": run_id
        })
        
        logger.info(f"Direct experiment completed in {execution_time:.2f}s - Success: {result.get('success', False)}")
        return result
        
    except Exception as e:
        logger.error(f"Direct experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        execution_time = time.time() - start_time
        return {
            "experiment_type": "baseline",
            "task_id": task_id,
            "seed": seed,
            "success": False,
            "error": str(e),
            "total_execution_time": execution_time
        }

def run_single_direct_experiment(
    task_id: str,
    task_config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Run a single experiment directly with the orchestrator setup (sync wrapper)."""
    return asyncio.run(run_single_direct_experiment_async(task_id, task_config, seed))


def run_single_legacy_experiment(
    adapter_type: str,
    task_id: str,
    task_config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Run a single legacy experiment with the specified configuration."""
    
    logger.info(f"Running legacy {adapter_type} experiment on task {task_id} with seed {seed}")
    
    try:
        # Create adapter
        if adapter_type == "baseline":
            adapter = create_baseline_adapter()
        elif adapter_type == "rag":
            adapter = create_rag_adapter()
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # Add seed to task config
        task_config_with_seed = task_config.copy()
        task_config_with_seed["seed"] = seed
        
        # Run task
        start_time = time.time()
        result = adapter.run_task(task_id, task_config_with_seed)
        execution_time = time.time() - start_time
        
        # Add experiment metadata
        result.update({
            "experiment_type": adapter_type,
            "task_id": task_id,
            "seed": seed,
            "total_execution_time": execution_time
        })
        
        logger.info(f"Legacy experiment completed in {execution_time:.2f}s - Success: {result.get('success', False)}")
        return result
        
    except Exception as e:
        logger.error(f"Legacy experiment failed: {str(e)}")
        return {
            "experiment_type": adapter_type,
            "task_id": task_id,
            "seed": seed,
            "success": False,
            "error": str(e),
            "total_execution_time": time.time() - start_time if 'start_time' in locals() else 0
        }


def run_experiments(
    experiments: List[str],
    tasks: List[str],
    seeds: List[int],
    output_dir: Path
) -> Dict[str, Any]:
    """Run multiple experiments and save results using REALM-Bench tasks if available."""
    
    # For now, always use the simple approach with real REALM-Bench tasks
    # but the existing working experiment setup
    if REALM_BENCH_AVAILABLE:
        return run_simple_realm_bench_experiments(experiments, tasks, seeds, output_dir)
    else:
        return run_legacy_experiments(experiments, tasks, seeds, output_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run MAS REALM-Bench experiments")
    
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["baseline", "rag"],
        default=["baseline"],
        help="Experiment types to run"
    )
    
    # Determine available tasks based on REALM-Bench availability
    if REALM_BENCH_AVAILABLE:
        available_tasks = list(TASK_DEFINITIONS.keys())
        default_tasks = ["P1", "P3", "P7"]  # Start with a few representative tasks
    else:
        available_tasks = ["P1"]  # Only P1 available in legacy mode
        default_tasks = ["P1"]
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=available_tasks,
        default=default_tasks,
        help=f"Tasks to evaluate. Available: {available_tasks}"
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
        default=Path("out/experiments"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--use-builtin-code-interpreter",
        action="store_true",
        help="Use OpenAI's built-in code interpreter instead of local Jupyter kernel"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also enable debug for our module specifically
        logger.setLevel(logging.DEBUG)
    
    # Set environment variable for built-in code interpreter if requested
    if args.use_builtin_code_interpreter:
        os.environ["USE_BUILTIN_CODE_INTERPRETER"] = "true"
        logger.info("🔧 Using OpenAI built-in code interpreter")
    
    # Display configuration info
    logger.info("Starting MAS REALM-Bench experiments")
    logger.info(f"REALM-Bench integration: {'✅ ACTIVE' if REALM_BENCH_AVAILABLE else '❌ FALLBACK'}")
    logger.info(f"Experiments: {args.experiments}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Seeds: {args.seeds}")
    
    if REALM_BENCH_AVAILABLE:
        logger.info(f"Available REALM-Bench tasks: {len(TASK_DEFINITIONS)} total")
        for task_id in args.tasks:
            task_def = TASK_DEFINITIONS[task_id]
            logger.info(f"  📋 {task_id}: {task_def.name} ({task_def.category.value})")
    
    try:
        summary = run_experiments(
            args.experiments,
            args.tasks,
            args.seeds,
            args.output_dir
        )
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Evaluation Framework: {summary.get('evaluation_framework', 'Unknown')}")
        print(f"REALM-Bench Integration: {'✅ Active' if summary.get('realm_bench_integration', False) else '❌ Legacy'}")
        print(f"Total runs: {summary['total_runs']}")
        print(f"Successful: {summary['successful_runs']}")
        print(f"Failed: {summary['failed_runs']}")
        print(f"Success rate: {summary['successful_runs']/summary['total_runs']*100:.1f}%")
        print(f"Results saved to: {args.output_dir}")
        
        # Enhanced REALM-Bench Analytics (adapted for both modes)
        if summary['results']:
            print("\n" + "="*50)
            print("REALM-BENCH PERFORMANCE ANALYSIS")
            print("="*50)
            
            # Group results by experiment and task
            experiment_summary = {}
            for result in summary['results']:
                exp_type = result.get('experiment_type', result.get('experiment', 'unknown'))
                task_id = result.get('task_id', result.get('task', 'unknown'))
                key = f"{exp_type}_{task_id}"
                
                if key not in experiment_summary:
                    experiment_summary[key] = {
                        'experiment': exp_type,
                        'task': task_id,
                        'runs': [],
                        'successes': 0
                    }
                
                experiment_summary[key]['runs'].append(result)
                if result.get('success', False):
                    experiment_summary[key]['successes'] += 1
            
            # Per-task/experiment breakdown
            print("Per-Experiment/Task Performance:")
            for key, data in experiment_summary.items():
                success_rate = (data['successes'] / len(data['runs'])) * 100
                task_info = ""
                
                if REALM_BENCH_AVAILABLE and data['task'] in TASK_DEFINITIONS:
                    task_def = TASK_DEFINITIONS[data['task']]
                    task_info = f" ({task_def.name})"
                
                print(f"  {data['experiment']}/{data['task']}{task_info}:")
                print(f"    Success Rate: {success_rate:.1f}% ({data['successes']}/{len(data['runs'])} runs)")
                
                # Calculate average metrics for successful runs
                successful_runs = [r for r in data['runs'] if r.get('success', False)]
                if successful_runs:
                    avg_execution_time = sum(r.get('total_execution_time', 0) for r in successful_runs) / len(successful_runs)
                    print(f"    Average Execution Time: {avg_execution_time:.2f}s")
                    
                    # Show achieved goals/constraints for REALM-Bench runs
                    if REALM_BENCH_AVAILABLE and successful_runs[0].get('achieved_goals'):
                        achieved_goals = successful_runs[0]['achieved_goals']
                        satisfied_constraints = successful_runs[0].get('satisfied_constraints', [])
                        
                        if data['task'] in TASK_DEFINITIONS:
                            task_def = TASK_DEFINITIONS[data['task']]
                            print(f"    Goals: {len(achieved_goals)}/{len(task_def.goals)} achieved")
                            print(f"    Constraints: {len(satisfied_constraints)}/{len(task_def.constraints)} satisfied")
                print()
            
            # Show specific details for successful runs
            if REALM_BENCH_AVAILABLE:
                print("Successful Run Details (REALM-Bench):")
                successful_results = [r for r in summary['results'] if r.get('success', False)]
                for result in successful_results[:5]:  # Show first 5 to avoid spam
                    task_id = result.get('task_id', result.get('task'))
                    exp_type = result.get('experiment_type', result.get('experiment'))
                    seed = result.get('seed', 'unknown')
                    
                    print(f"  {exp_type}/{task_id}/seed_{seed}:")
                    
                    achieved_goals = result.get('achieved_goals', [])
                    satisfied_constraints = result.get('satisfied_constraints', [])
                    
                    if task_id in TASK_DEFINITIONS:
                        task_def = TASK_DEFINITIONS[task_id]
                        
                        print(f"    Goals Achieved ({len(achieved_goals)}/{len(task_def.goals)}):")
                        for goal_id in achieved_goals:
                            goal = next((g for g in task_def.goals if g.goal_id == goal_id), None)
                            if goal:
                                print(f"      📈 {goal_id}: {goal.description}")
                        
                        print(f"    Constraints Satisfied ({len(satisfied_constraints)}/{len(task_def.constraints)}):")
                        for constraint_id in satisfied_constraints:
                            constraint = next((c for c in task_def.constraints if c.constraint_id == constraint_id), None)
                            if constraint:
                                print(f"      ✅ {constraint_id}: {constraint.description}")
                    print()
                
                if len(successful_results) > 5:
                    print(f"    ... and {len(successful_results) - 5} more successful runs")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())