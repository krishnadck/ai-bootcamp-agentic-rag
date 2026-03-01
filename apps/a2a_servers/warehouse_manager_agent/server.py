import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_executor import WarehouseManagerAgentExecutor
from agent import WarehouseManagerAgent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HOST = "localhost"
PORT = 8020

app_name = "warehouse_manager_agent"
app_description = "A warehouse manager agent that can check the availability of items in the warehouse and reserve them"

capabilities = AgentCapabilities(streaming=True)
skill1 = AgentSkill(
    id="01_check_warehouse_availability",
    name="check_warehouse_availability",
    description="Check the availability of items in the warehouse",
    tags=["warehouse", "availability"],
    examples=["what is the availability of the item with id 123?", "what is the availability of the item with id 456?"]
)

skill2 = AgentSkill(
    id="02_reserve_warehouse_items",
    name="reserve_warehouse_items",
    description="Reserve items in the warehouse",
    tags=["warehouse", "reservation"],
    examples=["reserve 10 units of the item with id 123", "reserve 5 units of the item with id 456"]
)

skills = [skill1, skill2]

agent_card = AgentCard(
    name=app_name,
    description=app_description,
    url=f"http://{HOST}:{PORT}/",
    version="1.0.0",
    author="Krishna Kumar D c",
    email="krishnakumar.dc@gmail.com",
    tags=["warehouse", "availability", "reservation"],
    default_input_modes=["text"],
    default_output_modes=["text"],
    skills=skills,
    capabilities=capabilities,
)

adk_agent = WarehouseManagerAgent().get_agent()

runner = Runner(
    agent=adk_agent,
    app_name=app_name,
    session_service=InMemorySessionService(),
    memory_service=InMemoryMemoryService(),
    artifact_service=InMemoryArtifactService(),
)

agent_executor = WarehouseManagerAgentExecutor(runner=runner, app_name=app_name)

request_handler = DefaultRequestHandler(
    agent_executor=agent_executor,
    task_store=InMemoryTaskStore(),
)

server = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
)

app = server.build()

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
