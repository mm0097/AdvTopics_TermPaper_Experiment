"""
Agent Interaction Logger for Research Documentation
Captures all agent input-output pairs in structured formats for analysis.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class AgentInteraction:
    """Represents a single agent interaction."""
    timestamp: str
    run_id: str
    agent_id: str
    capability: str
    subtask_id: str
    subtask_description: str
    input_prompt: str
    output_response: str
    tool_calls: List[Dict[str, Any]]
    execution_time: float
    tokens_used: Dict[str, int]
    status: str
    error: Optional[str] = None
    memory_usage: Optional[Dict[str, float]] = None

class InteractionLogger:
    """Logs agent interactions for research documentation."""
    
    def __init__(self, output_dir: str = "logs/agent_interactions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interactions: List[AgentInteraction] = []
        
    def log_interaction(
        self,
        run_id: str,
        agent_id: str,
        capability: str,
        subtask_id: str,
        subtask_description: str,
        input_prompt: str,
        output_response: str,
        tool_calls: List[Dict[str, Any]] = None,
        execution_time: float = 0.0,
        tokens_used: Dict[str, int] = None,
        status: str = "completed",
        error: Optional[str] = None,
        memory_usage: Optional[Dict[str, float]] = None
    ):
        """Log a single agent interaction."""
        
        interaction = AgentInteraction(
            timestamp=datetime.now().isoformat(),
            run_id=run_id,
            agent_id=agent_id,
            capability=capability,
            subtask_id=subtask_id,
            subtask_description=subtask_description,
            input_prompt=input_prompt,
            output_response=output_response,
            tool_calls=tool_calls or [],
            execution_time=execution_time,
            tokens_used=tokens_used or {},
            status=status,
            error=error,
            memory_usage=memory_usage
        )
        
        self.interactions.append(interaction)
        logger.info(f"📝 Logged interaction: {agent_id} -> {subtask_id}")
        
    def save_session(self, run_id: str, format: str = "both"):
        """Save all interactions for a session."""
        session_interactions = [i for i in self.interactions if i.run_id == run_id]
        
        if not session_interactions:
            logger.warning(f"No interactions found for run_id: {run_id}")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{run_id}_{timestamp}"
        
        if format in ["json", "both"]:
            self._save_json(session_interactions, base_filename)
            
        if format in ["markdown", "both"]:
            self._save_markdown(session_interactions, base_filename)
            
    def _save_json(self, interactions: List[AgentInteraction], base_filename: str):
        """Save interactions as structured JSON."""
        json_file = self.output_dir / f"{base_filename}_interactions.json"
        
        data = {
            "metadata": {
                "session_id": interactions[0].run_id,
                "total_interactions": len(interactions),
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": sum(i.execution_time for i in interactions),
                "total_tokens": self._aggregate_tokens(interactions)
            },
            "interactions": [asdict(interaction) for interaction in interactions]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 Saved JSON log: {json_file}")
        
    def _save_markdown(self, interactions: List[AgentInteraction], base_filename: str):
        """Save interactions as readable Markdown."""
        md_file = self.output_dir / f"{base_filename}_interactions.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Agent Interactions Log\n\n")
            f.write(f"**Session ID:** {interactions[0].run_id}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Interactions:** {len(interactions)}\n")
            f.write(f"**Total Execution Time:** {sum(i.execution_time for i in interactions):.2f}s\n")
            f.write(f"**Total Tokens:** {self._aggregate_tokens(interactions)}\n\n")
            
            # Table of Contents
            f.write("## Table of Contents\n\n")
            for i, interaction in enumerate(interactions, 1):
                f.write(f"{i}. [{interaction.subtask_id}: {interaction.capability}](#{interaction.subtask_id.lower().replace('_', '-')})\n")
            f.write("\n")
            
            # Interactions
            for i, interaction in enumerate(interactions, 1):
                f.write(f"## {i}. {interaction.subtask_id}\n\n")
                f.write(f"**Agent:** {interaction.agent_id}\n")
                f.write(f"**Capability:** {interaction.capability}\n")
                f.write(f"**Status:** {interaction.status}\n")
                f.write(f"**Execution Time:** {interaction.execution_time:.2f}s\n")
                f.write(f"**Tokens Used:** {interaction.tokens_used}\n")
                f.write(f"**Timestamp:** {interaction.timestamp}\n\n")
                
                f.write(f"### Subtask Description\n")
                f.write(f"{interaction.subtask_description}\n\n")
                
                f.write(f"### Input Prompt\n")
                f.write(f"```\n{interaction.input_prompt}\n```\n\n")
                
                f.write(f"### Agent Response\n")
                f.write(f"```\n{interaction.output_response}\n```\n\n")
                
                if interaction.tool_calls:
                    f.write(f"### Tool Calls\n")
                    for j, tool_call in enumerate(interaction.tool_calls, 1):
                        f.write(f"#### Tool Call {j}\n")
                        f.write(f"```json\n{json.dumps(tool_call, indent=2)}\n```\n\n")
                
                if interaction.error:
                    f.write(f"### Error\n")
                    f.write(f"```\n{interaction.error}\n```\n\n")
                    
                f.write("---\n\n")
                
        logger.info(f"📄 Saved Markdown log: {md_file}")
        
    def _aggregate_tokens(self, interactions: List[AgentInteraction]) -> Dict[str, int]:
        """Aggregate token usage across interactions."""
        total_tokens = {}
        
        for interaction in interactions:
            for key, value in interaction.tokens_used.items():
                total_tokens[key] = total_tokens.get(key, 0) + value
                
        return total_tokens
        
    def get_session_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary statistics for a session."""
        session_interactions = [i for i in self.interactions if i.run_id == run_id]
        
        if not session_interactions:
            return {}
            
        successful = [i for i in session_interactions if i.status == "completed"]
        failed = [i for i in session_interactions if i.status == "failed"]
        
        return {
            "run_id": run_id,
            "total_interactions": len(session_interactions),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(session_interactions) if session_interactions else 0,
            "total_execution_time": sum(i.execution_time for i in session_interactions),
            "average_execution_time": sum(i.execution_time for i in session_interactions) / len(session_interactions) if session_interactions else 0,
            "capabilities_used": list(set(i.capability for i in session_interactions)),
            "total_tokens": self._aggregate_tokens(session_interactions),
            "tool_usage": self._analyze_tool_usage(session_interactions)
        }
        
    def _analyze_tool_usage(self, interactions: List[AgentInteraction]) -> Dict[str, int]:
        """Analyze tool usage across interactions."""
        tool_usage = {}
        
        for interaction in interactions:
            for tool_call in interaction.tool_calls:
                if isinstance(tool_call, dict):
                    if 'function' in tool_call:
                        tool_name = tool_call['function'].get('name', 'unknown')
                    elif 'name' in tool_call:
                        tool_name = tool_call['name']
                    else:
                        tool_name = 'unknown'
                elif hasattr(tool_call, 'function'):
                    tool_name = tool_call.function.name
                else:
                    tool_name = 'unknown'
                    
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                
        return tool_usage

# Global logger instance
_interaction_logger = None

def get_interaction_logger() -> InteractionLogger:
    """Get the global interaction logger instance."""
    global _interaction_logger
    if _interaction_logger is None:
        _interaction_logger = InteractionLogger()
    return _interaction_logger

def log_agent_interaction(**kwargs):
    """Convenience function to log an agent interaction."""
    logger = get_interaction_logger()
    logger.log_interaction(**kwargs)