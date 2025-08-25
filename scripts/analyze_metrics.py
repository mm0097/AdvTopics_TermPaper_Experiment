#!/usr/bin/env python3
"""
Advanced Metrics Analysis for REALM-Bench MAS Experiments

This script implements the metrics defined in METRICS_IMPLEMENTATION_PROPOSAL.md:
1. Plan Parallelism Score - measures structural quality of task decomposition
2. Plan Redundancy Score - detects semantic overlap in parallel tasks  
3. Agent-Task Fit Score - evaluates delegation accuracy using semantic similarity

All metrics are computed post-hoc from existing experimental data.
"""

import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import sentence transformers for embeddings (optional)
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
    logger.info("✅ Sentence transformers available for semantic analysis")
except ImportError:
    HAS_EMBEDDINGS = False
    logger.warning("⚠️ Sentence transformers not available - using simplified similarity metrics")

@dataclass
class AdvancedMetrics:
    """Container for computed advanced metrics."""
    plan_parallelism_score: float
    plan_redundancy_score: float
    agent_task_fit_score: float
    critical_path_length: int
    total_tasks: int
    
    # Additional metrics
    planning_efficiency: float  # planning_latency / total_tasks
    token_efficiency: float    # total_tokens / total_tasks
    execution_balance: float   # std dev of execution times

class MetricsAnalyzer:
    """Analyzer for computing advanced MAS orchestrator metrics."""
    
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings and HAS_EMBEDDINGS
        
        if self.use_embeddings:
            logger.info("🤖 Loading embedding model for semantic analysis...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
        else:
            self.embedding_model = None
            logger.info("📝 Using text-based similarity analysis")
        
        # Agent capability descriptions for semantic matching
        self.capability_descriptions = {
            "general": "Handles general-purpose tasks or those not covered by a specialist.",
            "route_estimation": "Estimates the time or cost of a potential action or route.",
            "vehicle_assignment": "Assigns available resources or assets to specific tasks.",
            "fuel_check": "Monitors the status or levels of critical resources.",
            "navigation": "Provides guidance on executing a sequence of steps.",
            "passenger_handling": "Manages interactions with users or external entities.",
            "confirmation_and_replanning": "Performs final validation and verification of a plan.",
            "data_collector": "Gathers and consolidates information from various sources.",
            "travel_time_estimator": "Calculates the expected duration of a task or travel.",
            "route_planner": "Creates an ordered sequence of actions or movements.",
            "validator": "Checks work against a set of rules or constraints.",
            "scheduler": "Organizes tasks or events into a timeline.",
            "optimizer": "Improves a plan or solution to better meet a goal."
        }
    
    def analyze_experiment_results(self, results_file: Path, group_by: str = None) -> Dict[str, Any]:
        """
        Analyze experiments in a results file, optionally grouped by configuration.
        
        Args:
            results_file: Path to JSON results file
            group_by: How to group experiments ('scenario', 'rag_setting', 'scenario_rag', or None for no grouping)
        """
        logger.info(f"🔍 Analyzing results from {results_file}")
        
        with open(results_file, 'r') as f:
            experiments = json.load(f)
        
        if not isinstance(experiments, list):
            experiments = [experiments]
        
        if group_by:
            return self._analyze_grouped_experiments(experiments, group_by)
        else:
            return self._analyze_all_experiments(experiments)
    
    def _analyze_all_experiments(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze all experiments together (original behavior)."""
        all_metrics = []
        analysis_summary = {
            "total_experiments": len(experiments),
            "successful_experiments": 0,
            "failed_experiments": 0,
            "metrics": [],
            "correlations": {},
            "insights": []
        }
        
        for exp in experiments:
            if exp.get("success", False):
                analysis_summary["successful_experiments"] += 1
            else:
                analysis_summary["failed_experiments"] += 1
            
            # Analyze both successful and failed experiments for metrics
            try:
                metrics = self.compute_metrics_for_experiment(exp)
                all_metrics.append(metrics)
                analysis_summary["metrics"].append({
                    "run_id": exp.get("run_id", "unknown"),
                    "success": exp.get("success", False),
                    "metrics": metrics.__dict__
                })
            except Exception as e:
                logger.error(f"❌ Failed to analyze experiment {exp.get('run_id', 'unknown')}: {e}")
        
        if all_metrics:
            # Compute correlations and insights
            analysis_summary["correlations"] = self.compute_correlations(all_metrics, experiments)
            analysis_summary["insights"] = self.generate_insights(all_metrics, experiments)
        
        logger.info(f"✅ Analysis complete: {len(all_metrics)} experiments analyzed")
        return analysis_summary
    
    def _analyze_grouped_experiments(self, experiments: List[Dict[str, Any]], group_by: str) -> Dict[str, Any]:
        """Analyze experiments grouped by configuration."""
        logger.info(f"📊 Grouping experiments by: {group_by}")
        
        # Group experiments
        groups = self._group_experiments(experiments, group_by)
        
        grouped_analysis = {
            "grouping_strategy": group_by,
            "total_experiments": len(experiments),
            "groups": {},
            "comparison": {}
        }
        
        # Analyze each group separately
        for group_name, group_experiments in groups.items():
            logger.info(f"🔍 Analyzing group: {group_name} ({len(group_experiments)} experiments)")
            
            group_analysis = self._analyze_all_experiments(group_experiments)
            grouped_analysis["groups"][group_name] = group_analysis
        
        # Generate comparison insights
        if len(groups) > 1:
            grouped_analysis["comparison"] = self._compare_groups(groups)
        
        logger.info(f"✅ Grouped analysis complete: {len(groups)} groups analyzed")
        return grouped_analysis
    
    def _group_experiments(self, experiments: List[Dict[str, Any]], group_by: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group experiments by the specified criteria."""
        groups = defaultdict(list)
        
        for exp in experiments:
            if group_by == "scenario":
                # Extract scenario from run_id (e.g., "P1_seed_42_rag" -> "P1")
                run_id = exp.get("run_id", "unknown")
                scenario = run_id.split("_")[0] if "_" in run_id else run_id
                groups[scenario].append(exp)
                
            elif group_by == "rag_setting":
                # Extract RAG setting from run_id or retrieval_enabled flag
                run_id = exp.get("run_id", "unknown")
                retrieval_enabled = exp.get("retrieval_enabled", False)
                
                # Check for explicit RAG indicators
                has_rag_in_id = "rag" in run_id.lower() and "norag" not in run_id.lower()
                
                if has_rag_in_id or retrieval_enabled:
                    groups["RAG_enabled"].append(exp)
                else:
                    groups["RAG_disabled"].append(exp)
                    
            elif group_by == "scenario_rag":
                # Group by both scenario and RAG setting
                run_id = exp.get("run_id", "unknown")
                scenario = run_id.split("_")[0] if "_" in run_id else run_id
                retrieval_enabled = exp.get("retrieval_enabled", False)
                
                # Check for explicit RAG indicators
                has_rag_in_id = "rag" in run_id.lower() and "norag" not in run_id.lower()
                
                rag_suffix = "RAG" if (has_rag_in_id or retrieval_enabled) else "noRAG"
                group_key = f"{scenario}_{rag_suffix}"
                groups[group_key].append(exp)
                
            else:
                logger.warning(f"Unknown grouping strategy: {group_by}")
                groups["all"].append(exp)
        
        return dict(groups)
    
    def _compare_groups(self, groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate comparison insights between groups."""
        comparison = {
            "group_summary": {},
            "metric_comparisons": {},
            "insights": []
        }
        
        # Compute summary metrics for each group
        for group_name, group_experiments in groups.items():
            successful = sum(1 for exp in group_experiments if exp.get("success", False))
            total = len(group_experiments)
            success_rate = successful / total if total > 0 else 0.0
            
            # Compute average metrics for successful experiments
            successful_experiments = [exp for exp in group_experiments if exp.get("success", False)]
            
            if successful_experiments:
                avg_execution_time = np.mean([exp.get("execution_time", 0) for exp in successful_experiments])
                avg_tokens = np.mean([exp.get("token_usage", {}).get("total", 0) for exp in successful_experiments])
                avg_planning_time = np.mean([exp.get("planning_latency_seconds", 0) for exp in successful_experiments])
            else:
                avg_execution_time = avg_tokens = avg_planning_time = 0.0
            
            comparison["group_summary"][group_name] = {
                "total_experiments": total,
                "successful_experiments": successful,
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "avg_tokens_used": avg_tokens,
                "avg_planning_time": avg_planning_time
            }
        
        # Generate comparison insights
        group_names = list(groups.keys())
        
        # Compare success rates
        success_rates = [comparison["group_summary"][name]["success_rate"] for name in group_names]
        if len(success_rates) > 1:
            best_group = group_names[np.argmax(success_rates)]
            comparison["insights"].append(f"🏆 Highest success rate: {best_group} ({max(success_rates):.1%})")
        
        # Compare execution times (for successful experiments only)
        exec_times = [comparison["group_summary"][name]["avg_execution_time"] 
                     for name in group_names if comparison["group_summary"][name]["avg_execution_time"] > 0]
        if len(exec_times) > 1:
            valid_groups = [name for name in group_names if comparison["group_summary"][name]["avg_execution_time"] > 0]
            fastest_group = valid_groups[np.argmin(exec_times)]
            comparison["insights"].append(f"⚡ Fastest execution: {fastest_group} ({min(exec_times):.1f}s avg)")
        
        # Compare token usage
        token_usage = [comparison["group_summary"][name]["avg_tokens_used"] 
                      for name in group_names if comparison["group_summary"][name]["avg_tokens_used"] > 0]
        if len(token_usage) > 1:
            valid_groups = [name for name in group_names if comparison["group_summary"][name]["avg_tokens_used"] > 0]
            most_efficient = valid_groups[np.argmin(token_usage)]
            comparison["insights"].append(f"💰 Most token-efficient: {most_efficient} ({min(token_usage):.0f} tokens avg)")
        
        return comparison
    
    def compute_metrics_for_experiment(self, experiment: Dict[str, Any]) -> AdvancedMetrics:
        """Compute advanced metrics for a single experiment."""
        schedule = experiment.get("schedule", [])
        
        if not schedule:
            logger.warning("⚠️ No schedule found in experiment")
            return self._create_empty_metrics()
        
        # 1. Plan Parallelism Score
        parallelism_score, critical_path_length = self.compute_plan_parallelism_score(schedule)
        
        # 2. Plan Redundancy Score  
        redundancy_score = self.compute_plan_redundancy_score(schedule)
        
        # 3. Agent-Task Fit Score
        fit_score = self.compute_agent_task_fit_score(schedule)
        
        # Additional efficiency metrics
        planning_latency = experiment.get("planning_latency_seconds", 0)
        total_tasks = len(schedule)
        planning_efficiency = planning_latency / max(total_tasks, 1)
        
        total_tokens = experiment.get("token_usage", {}).get("total", 0)
        token_efficiency = total_tokens / max(total_tasks, 1)
        
        execution_times = experiment.get("resource_usage", {}).get("execution_times", [])
        execution_balance = np.std(execution_times) if execution_times else 0.0
        
        return AdvancedMetrics(
            plan_parallelism_score=parallelism_score,
            plan_redundancy_score=redundancy_score,
            agent_task_fit_score=fit_score,
            critical_path_length=critical_path_length,
            total_tasks=total_tasks,
            planning_efficiency=planning_efficiency,
            token_efficiency=token_efficiency,
            execution_balance=execution_balance
        )
    
    def compute_plan_parallelism_score(self, schedule: List[Dict[str, Any]]) -> Tuple[float, int]:
        """
        Compute Plan Parallelism Score using dependency graph analysis.
        
        Score = 1 - (Critical Path Length / Total Tasks)
        A score of 1.0 indicates maximum parallelism (all tasks independent)
        A score of 0.0 indicates purely sequential execution
        """
        if not schedule:
            return 0.0, 0
        
        # Build dependency graph
        G = nx.DiGraph()
        
        for task in schedule:
            task_id = task["task_id"]
            G.add_node(task_id)
            
            # Add edges for dependencies
            for dep in task.get("dependencies", []):
                if dep:  # Ensure dependency is not empty
                    G.add_edge(dep, task_id)
        
        # Find critical path using longest path in DAG
        try:
            # Get topological order to find longest path
            topo_order = list(nx.topological_sort(G))
            
            # Compute longest path (critical path)
            distances = {node: 0 for node in G.nodes()}
            
            for node in topo_order:
                for successor in G.successors(node):
                    distances[successor] = max(distances[successor], distances[node] + 1)
            
            critical_path_length = max(distances.values()) + 1 if distances else 1
            
        except nx.NetworkXError:
            # Handle cycles or other graph issues
            logger.warning("⚠️ Dependency graph has cycles or other issues")
            critical_path_length = len(schedule)  # Worst case: all sequential
        
        total_tasks = len(schedule)
        parallelism_score = 1.0 - (critical_path_length / total_tasks)
        
        logger.debug(f"📊 Parallelism: {parallelism_score:.3f} (critical path: {critical_path_length}/{total_tasks})")
        return parallelism_score, critical_path_length
    
    def compute_plan_redundancy_score(self, schedule: List[Dict[str, Any]]) -> float:
        """
        Compute Plan Redundancy Score by analyzing semantic similarity across ALL tasks.
        
        Higher scores indicate more redundancy (bad).
        Lower scores indicate distinct, meaningful task decomposition (good).
        """
        if not schedule:
            return 0.0
        
        if len(schedule) < 2:
            return 0.0  # Can't have redundancy with single task
        
        total_similarity = 0.0
        comparison_count = 0
        
        # Compare all pairs of tasks in the entire plan
        for i in range(len(schedule)):
            for j in range(i + 1, len(schedule)):
                task1_desc = schedule[i].get("description", "")
                task2_desc = schedule[j].get("description", "")
                
                similarity = self._compute_semantic_similarity(task1_desc, task2_desc)
                total_similarity += similarity
                comparison_count += 1
        
        redundancy_score = total_similarity / max(comparison_count, 1)
        logger.debug(f"📊 Redundancy: {redundancy_score:.3f} (avg similarity across {comparison_count} task pairs)")
        return redundancy_score
    
    def compute_agent_task_fit_score(self, schedule: List[Dict[str, Any]]) -> float:
        """
        Compute Agent-Task Fit Score using semantic similarity between task descriptions
        and agent capability descriptions.
        
        Higher scores indicate better task-capability matching.
        """
        if not schedule:
            return 0.0
        
        total_fit = 0.0
        
        for task in schedule:
            task_desc = task.get("description", "")
            capability = task.get("capability", "general")
            
            # Get capability description
            capability_desc = self.capability_descriptions.get(capability, "General purpose capability")
            
            # Compute semantic similarity
            fit_score = self._compute_semantic_similarity(task_desc, capability_desc)
            total_fit += fit_score
        
        avg_fit = total_fit / len(schedule)
        logger.debug(f"📊 Agent-Task Fit: {avg_fit:.3f} (avg across {len(schedule)} tasks)")
        return avg_fit
    
    def _group_tasks_by_level(self, schedule: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group tasks by their execution level based on dependencies."""
        # Create task lookup
        task_dict = {task["task_id"]: task for task in schedule}
        
        # Compute levels using topological sort
        levels = []
        remaining_tasks = set(task["task_id"] for task in schedule)
        
        while remaining_tasks:
            # Find tasks with no remaining dependencies
            current_level = []
            for task_id in list(remaining_tasks):
                task = task_dict[task_id]
                dependencies = task.get("dependencies", [])
                
                # Check if all dependencies are satisfied (not in remaining tasks)
                if all(dep not in remaining_tasks for dep in dependencies if dep):
                    current_level.append(task)
            
            if not current_level:
                # Handle circular dependencies by taking all remaining
                current_level = [task_dict[tid] for tid in remaining_tasks]
            
            levels.append(current_level)
            for task in current_level:
                remaining_tasks.remove(task["task_id"])
        
        return levels
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        if self.use_embeddings:
            # Use sentence transformers for semantic similarity
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        else:
            # Simple text-based similarity (word overlap)
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
    
    def _create_empty_metrics(self) -> AdvancedMetrics:
        """Create empty metrics for failed analyses."""
        return AdvancedMetrics(
            plan_parallelism_score=0.0,
            plan_redundancy_score=0.0,
            agent_task_fit_score=0.0,
            critical_path_length=0,
            total_tasks=0,
            planning_efficiency=0.0,
            token_efficiency=0.0,
            execution_balance=0.0
        )
    
    def compute_correlations(self, metrics_list: List[AdvancedMetrics], experiments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute correlations between metrics and performance indicators."""
        if not metrics_list or len(metrics_list) < 2:
            return {}
        
        # Extract metric arrays - only for successful experiments
        successful_experiments = [exp for exp in experiments if exp.get("success")]
        successful_metrics = [metrics_list[i] for i, exp in enumerate(experiments) if exp.get("success")]
        
        if len(successful_experiments) < 2 or len(successful_metrics) < 2:
            return {}
        
        # Extract metric arrays
        parallelism_scores = [m.plan_parallelism_score for m in successful_metrics]
        redundancy_scores = [m.plan_redundancy_score for m in successful_metrics]
        fit_scores = [m.agent_task_fit_score for m in successful_metrics]
        
        # Extract performance indicators from successful experiments
        execution_times = [exp.get("execution_time", exp.get("total_execution_time", 0)) for exp in successful_experiments]
        success_rates = [exp.get("final_state", {}).get("success_rate", 1.0) for exp in successful_experiments]  # Default to 1.0 for successful
        token_counts = [exp.get("token_usage", {}).get("total", 0) for exp in successful_experiments]
        
        correlations = {}
        
        # Only compute correlations if we have matching array sizes and sufficient data
        if len(parallelism_scores) == len(execution_times) and len(parallelism_scores) >= 2:
            try:
                if len(set(parallelism_scores)) > 1 and len(set(execution_times)) > 1:  # Check for variance
                    correlations["parallelism_vs_execution_time"] = np.corrcoef(parallelism_scores, execution_times)[0, 1]
                if len(set(parallelism_scores)) > 1 and len(set(success_rates)) > 1:
                    correlations["parallelism_vs_success_rate"] = np.corrcoef(parallelism_scores, success_rates)[0, 1]
                if len(set(redundancy_scores)) > 1 and len(set(execution_times)) > 1:
                    correlations["redundancy_vs_execution_time"] = np.corrcoef(redundancy_scores, execution_times)[0, 1]
                if len(set(redundancy_scores)) > 1 and len(set(token_counts)) > 1:
                    correlations["redundancy_vs_tokens"] = np.corrcoef(redundancy_scores, token_counts)[0, 1]
                if len(set(fit_scores)) > 1 and len(set(success_rates)) > 1:
                    correlations["fit_vs_success_rate"] = np.corrcoef(fit_scores, success_rates)[0, 1]
            except Exception as e:
                logger.warning(f"⚠️ Could not compute correlations: {e}")
        
        return correlations
    
    def generate_insights(self, metrics_list: List[AdvancedMetrics], experiments: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from the computed metrics."""
        insights = []
        
        if not metrics_list:
            return ["No metrics available for analysis"]
        
        # Average metrics
        avg_parallelism = np.mean([m.plan_parallelism_score for m in metrics_list])
        avg_redundancy = np.mean([m.plan_redundancy_score for m in metrics_list])
        avg_fit = np.mean([m.agent_task_fit_score for m in metrics_list])
        
        insights.append(f"Average Plan Parallelism Score: {avg_parallelism:.3f}")
        insights.append(f"Average Plan Redundancy Score: {avg_redundancy:.3f}")
        insights.append(f"Average Agent-Task Fit Score: {avg_fit:.3f}")
        
        # Qualitative insights
        if avg_parallelism > 0.7:
            insights.append("✅ High parallelism: Plans are well-structured for concurrent execution")
        elif avg_parallelism < 0.3:
            insights.append("⚠️ Low parallelism: Plans are mostly sequential, limiting efficiency")
        
        if avg_redundancy > 0.7:
            insights.append("⚠️ High redundancy: Parallel tasks have significant semantic overlap")
        elif avg_redundancy < 0.3:
            insights.append("✅ Low redundancy: Parallel tasks are distinct and meaningful")
        
        if avg_fit > 0.7:
            insights.append("✅ Good agent-task matching: Tasks are well-suited to assigned capabilities")
        elif avg_fit < 0.4:
            insights.append("⚠️ Poor agent-task matching: Tasks may not align well with capabilities")
        
        return insights

def main():
    """Main entry point for metrics analysis."""
    parser = argparse.ArgumentParser(description="Analyze REALM-Bench MAS experiment metrics")
    parser.add_argument(
        "--results-file", 
        type=Path,
        default=Path("out/experiments_fixed/realm_bench_detailed_results.json"),
        help="Path to experiment results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path, 
        default=Path("out/metrics_analysis"),
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--group-by",
        choices=["scenario", "rag_setting", "scenario_rag"],
        help="Group experiments by: 'scenario' (P1, P3, etc.), 'rag_setting' (RAG vs no-RAG), or 'scenario_rag' (P1_RAG, P1_noRAG, etc.)"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable semantic embeddings (use simple text similarity)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MetricsAnalyzer(use_embeddings=not args.no_embeddings)
    
    # Analyze results
    logger.info(f"🔬 Starting metrics analysis...")
    analysis = analyzer.analyze_experiment_results(args.results_file, group_by=args.group_by)
    
    # Save analysis results
    if args.group_by:
        output_file = args.output_dir / f"advanced_metrics_analysis_{args.group_by}.json"
    else:
        output_file = args.output_dir / "advanced_metrics_analysis.json"
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"📊 Analysis saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("ADVANCED METRICS ANALYSIS SUMMARY")
    print("="*60)
    
    if args.group_by:
        print(f"Grouping strategy: {analysis['grouping_strategy']}")
        print(f"Total experiments: {analysis['total_experiments']}")
        print(f"Groups analyzed: {len(analysis['groups'])}")
        
        # Print group summaries
        for group_name, group_analysis in analysis['groups'].items():
            print(f"\n📊 GROUP: {group_name}")
            print(f"  • Experiments: {group_analysis['total_experiments']}")
            print(f"  • Successful: {group_analysis['successful_experiments']}")
            print(f"  • Success rate: {group_analysis['successful_experiments']/group_analysis['total_experiments']:.1%}")
        
        # Print comparison insights
        if "comparison" in analysis and analysis["comparison"].get("insights"):
            print(f"\n🔍 COMPARISON INSIGHTS:")
            for insight in analysis["comparison"]["insights"]:
                print(f"  • {insight}")
        
    else:
        print(f"Total experiments: {analysis['total_experiments']}")
        print(f"Successful: {analysis['successful_experiments']}")
        print(f"Failed: {analysis['failed_experiments']}")
        print("\nKey Insights:")
        for insight in analysis['insights']:
            print(f"  • {insight}")
        
        if analysis['correlations']:
            print("\nCorrelations:")
            for metric, correlation in analysis['correlations'].items():
                print(f"  • {metric}: {correlation:.3f}")
    
    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()