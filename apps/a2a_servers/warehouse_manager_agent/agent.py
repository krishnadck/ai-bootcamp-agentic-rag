import os
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from tools import check_warehouse_availability, reserve_warehouse_items


class WarehouseManagerAgent:
    def __init__(self):
        self.model = LiteLlm(model="gpt-4.1-mini",
                temperature=0.0,
                api_key=os.getenv("OPENAI_API_KEY"),
                )
        
        self.agent = Agent(
            name="warehouse_manager_agent",
            model=self.model,
            description="A warehouse agent that checks the availability of items in the warehouse and reserves them",
            tools=[check_warehouse_availability, reserve_warehouse_items],
            instruction="""
            You are a part of the shopping assistant that can manage available inventory in the warehouses.
            """
        )

    def get_agent(self):
        return self.agent
    