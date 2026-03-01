"""A2A AgentExecutor implementation for the Warehouse Manager Agent."""

import logging
from collections.abc import AsyncGenerator

from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import Session
from google.genai import types

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState, TextPart, UnsupportedOperationError
from a2a.utils.errors import ServerError

from converters import a2a_parts_to_genai, genai_parts_to_a2a

logger = logging.getLogger(__name__)

_DEFAULT_USER_ID = "krishnak"


class WarehouseManagerAgentExecutor(AgentExecutor):
    """Executes Warehouse Manager agent tasks within the A2A server framework."""

    def __init__(self, runner: Runner, app_name: str) -> None:
        self.runner = runner
        self.app_name = app_name

    # ── ADK helpers ───────────────────────────────────────────────────────────

    def _run_agent(
        self,
        session_id: str,
        user_id: str,
        new_message: types.Content,
    ) -> AsyncGenerator[Event, None]:
        """Start an async ADK run and return the event stream."""
        return self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
        )

    async def _upsert_session(self, session_id: str, user_id: str) -> Session:
        """Return an existing session or create a new one."""
        session = await self.runner.session_service.get_session(
            session_id=session_id,
            user_id=user_id,
            app_name=self.app_name,
        )
        if session:
            return session
        return await self.runner.session_service.create_session(
            session_id=session_id,
            user_id=user_id,
            app_name=self.app_name,
        )

    # ── Event processing ──────────────────────────────────────────────────────

    def _extract_parts(self, event: Event) -> list[Part]:
        """Safely extract A2A parts from an ADK event, returning [] on empty content."""
        if event.content and event.content.parts:
            return genai_parts_to_a2a(event.content.parts)
        return []

    async def _handle_event(self, event: Event, updater: TaskUpdater) -> bool:
        """Process a single ADK event and update task state.

        Returns True when the final response has been sent and the loop should stop.
        """
        if event.is_final_response():
            parts = self._extract_parts(event)
            logger.debug("Final response parts: %s", parts)
            await updater.add_artifact(parts)
            await updater.complete()
            return True

        if event.get_function_calls():
            logger.debug("Tool calls in progress: %s", event.get_function_calls())
            return False

        parts = self._extract_parts(event)
        if parts:
            logger.debug("Intermediate response parts: %s", parts)
            updater.update_status(
                state=TaskState.working,
                message=updater.new_agent_message(parts=parts),
            )
        return False

    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        user_id: str,
        updater: TaskUpdater,
    ) -> None:
        """Run the agent for one turn and stream events to the TaskUpdater."""
        session = await self._upsert_session(session_id, user_id)
        async for event in self._run_agent(session.id, user_id, new_message):
            done = await self._handle_event(event, updater)
            if done:
                break

    # ── AgentExecutor interface ───────────────────────────────────────────────

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Entry point called by the A2A framework for every inbound message."""
        user_id: str = context.message.metadata.get("user_id", _DEFAULT_USER_ID)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )

        if not context.current_task:
            await updater.submit()

        await updater.start_work()
        
        print("context.message.parts: ", context.message.parts)

        try:
            await self._process_request(
                new_message=types.UserContent(parts=a2a_parts_to_genai(context.message.parts)),
                session_id=context.context_id,
                user_id=user_id,
                updater=updater,
            )
        except Exception as e:
            logger.error("Error processing request: %s", e, exc_info=True)
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[Part(root=TextPart(text="Internal server error while processing the request."))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel is not supported for this agent."""
        raise ServerError(error=UnsupportedOperationError())
