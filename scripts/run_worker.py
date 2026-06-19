import asyncio
import concurrent.futures
import logging
import os

from dotenv import load_dotenv
from temporalio.worker import Worker

from activities.tool_activities import (
    ToolActivities,
    dynamic_tool_activity,
    mcp_list_tools,
)
from shared.config import TEMPORAL_TASK_QUEUE, get_temporal_client
from shared.mcp_client_manager import MCPClientManager
from workflows.agent_goal_workflow import AgentGoalWorkflow


async def main():
    # Load environment variables
    load_dotenv(override=True)

    # Print Bedrock configuration info
    bedrock_model = os.environ.get(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    guardrail_id = os.environ.get("BEDROCK_GUARDRAIL_ID", "")
    guardrail_version = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "")
    guardrails_status = (
        f"on (id={guardrail_id})" if (guardrail_id and guardrail_version) else "off"
    )
    print(
        f"Worker will use Bedrock model: {bedrock_model} "
        f"(region={aws_region}, guardrails={guardrails_status})"
    )

    # Create shared MCP client manager
    mcp_client_manager = MCPClientManager()

    # Create the client
    client = await get_temporal_client()

    # Initialize the activities class with injected manager
    activities = ToolActivities(mcp_client_manager)

    print("Worker ready to process tasks!")
    logging.basicConfig(level=logging.INFO)

    # Run the worker with proper cleanup
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=100
        ) as activity_executor:
            worker = Worker(
                client,
                task_queue=TEMPORAL_TASK_QUEUE,
                workflows=[AgentGoalWorkflow],
                activities=[
                    activities.agent_validatePrompt,
                    activities.agent_toolPlanner,
                    activities.get_wf_env_vars,
                    activities.mcp_tool_activity,
                    dynamic_tool_activity,
                    mcp_list_tools,
                ],
                activity_executor=activity_executor,
            )

            print(f"Starting worker, connecting to task queue: {TEMPORAL_TASK_QUEUE}")
            await worker.run()
    finally:
        # Cleanup MCP connections when worker shuts down
        await mcp_client_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
