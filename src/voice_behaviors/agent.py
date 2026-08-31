"""The voice agent itself."""

import json
from collections.abc import AsyncIterator

from livekit.agents import Agent, FunctionTool, ModelSettings
from livekit.agents.llm import ChatChunk, ChatContext

from .config import DEMO_CALL_DATA


def _build_initial_chat_ctx() -> ChatContext:
    """Seed first-turn context as a user message.

    Mirrors how per-call data is injected without mutating the system prompt, so
    the cached prompt prefix stays byte-stable across calls. LiveKit replays
    `chat_ctx` as history before the first real turn, so the LLM still sees it.
    """
    chat_ctx = ChatContext.empty()
    call_block = json.dumps(DEMO_CALL_DATA, indent=2)
    chat_ctx.add_message(
        role="user",
        content=f"## Current Call Information\n```json\n{call_block}\n```",
    )
    return chat_ctx


class Assistant(Agent):
    def __init__(self, *, instructions: str, greeting: str) -> None:
        super().__init__(
            instructions=instructions,
            chat_ctx=_build_initial_chat_ctx(),
        )
        self._greeting = greeting

    async def on_enter(self) -> None:
        """Speak the opening line as soon as the session connects."""
        if self._greeting:
            self.session.say(self._greeting, allow_interruptions=False)

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[str | ChatChunk]:
        """Representative custom `llm_node`.

        Production runs a streaming filter over the model's output here. This is a
        transparent pass-through of the SDK's default node, so the emitted span
        shape is unchanged -- the point is that overriding `llm_node` and
        re-yielding chunks must still be traced correctly.
        """
        async for chunk in Agent.default.llm_node(
            self, chat_ctx, tools, model_settings
        ):
            yield chunk
