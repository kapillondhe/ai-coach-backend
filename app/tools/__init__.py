"""In-process coaching tools (fitness-domain logic) exposed to the coach agent.

These used to live in the separate ai-coach-mcp-server and were reached over MCP;
they now run in the backend process, so chat no longer depends on another deployed
service. MCP is only used for genuinely external tool providers (COROS).

Tool docstrings (incl. Args) are parsed into the tool description and parameter docs
the model sees, so keep them precise.
"""

from pydantic_ai.toolsets import FunctionToolset

from app.tools.knowledge_base import search_knowledge_base
from app.tools.nutrition import calculate_protein_intake
from app.tools.zones import calculate_heart_rate_zones, calculate_power_zones, calculate_swim_pace_zones

coaching_toolset = FunctionToolset(
    [
        calculate_protein_intake,
        search_knowledge_base,
        calculate_heart_rate_zones,
        calculate_power_zones,
        calculate_swim_pace_zones,
    ],
    id="coaching",
)
