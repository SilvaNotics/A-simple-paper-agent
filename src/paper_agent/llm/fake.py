# -*- coding: utf-8 -*-
"""离线假模型：无网络、无密钥时也能跑通整个图（`PAPER_AGENT_FAKE_LLM=1`）。

实现要点：
- `create_agent` 在构建时会调用 `bind_tools` / `with_structured_output`，
  所以假模型必须实现这两个方法（普通 FakeChatModel 会抛 NotImplementedError）；
- 假模型不会发起工具调用，直接返回一段占位文本 → agent 一步结束，
  随后各 `run_*` 函数走各自的兜底路径（模板 / 确定性检索）。
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda


class FakeToolCallingModel(BaseChatModel):
    """支持 bind_tools / 结构化输出的确定性假模型。"""

    reply: str = "[fake-llm] 离线模式占位回复"

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingModel":
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        def _build(_: Any) -> Any:
            try:
                return schema()
            except Exception:  # noqa: BLE001 - 必填字段缺失时交给调用方兜底
                return None

        return RunnableLambda(_build)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])
