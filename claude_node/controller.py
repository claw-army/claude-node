#!/usr/bin/env python3
"""
Claude CLI Node Controller — 生产实现
经过完整实测验证，基于 Claude Code v2.1.81+

协议：stream-json 双向通信（无 SDK 依赖）
输入格式：{"type":"user","message":{"role":"user","content":[{"type":"text","text":"..."}]}}
等待标志：stdout 出现 {"type":"result"} 即为本轮完成
"""

import subprocess
import json
import threading
import time
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from .exceptions import ClaudeSendConflictError
from .runtime import check_claude_available


# ── 消息模型 ─────────────────────────────────────────────────────────────────

@dataclass
class ClaudeMessage:
    """解析后的单条 Claude 输出消息"""
    type: str
    subtype: str = ""
    raw: dict = field(default_factory=dict)

    # ── 类型判断 ─────────────────────────────────────

    @property
    def is_init(self) -> bool:
        return self.type == "system" and self.subtype == "init"

    @property
    def is_result(self) -> bool:
        return self.type == "result"

    @property
    def is_result_ok(self) -> bool:
        return self.type == "result" and self.subtype == "success"

    @property
    def is_result_error(self) -> bool:
        return self.type == "result" and self.subtype == "error"

    # 已知 API 错误文本前缀（经实测：rate limit / auth 失败均返回 subtype=success）
    _ERROR_PREFIXES = ("API Error:", "Not logged in", "Rate limit", "Error:", "Authentication")

    @property
    def is_api_error(self) -> bool:
        """result_text 包含已知 API 错误（即使 subtype=success）"""
        if not self.is_result:
            return False
        t = self.result_text
        return any(t.startswith(p) for p in self._ERROR_PREFIXES)

    @property
    def truly_succeeded(self) -> bool:
        """真正成功：result ok 且 result_text 不是已知错误"""
        return self.is_result_ok and not self.is_api_error

    @property
    def is_task_event(self) -> bool:
        return self.type == "system" and self.subtype in (
            "task_started", "task_progress", "task_notification"
        )

    @property
    def is_assistant(self) -> bool:
        return self.type == "assistant"

    @property
    def is_tool_result(self) -> bool:
        """CLI 自动生成的工具执行结果"""
        if self.type != "user":
            return False
        content = self.raw.get("message", {}).get("content", [])
        if not isinstance(content, list):
            return False
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )

    # ── 内容提取 ─────────────────────────────────────

    @property
    def result_text(self) -> str:
        """本轮最终回复文本（仅 result 消息有效）"""
        return self.raw.get("result", "")

    @property
    def session_id(self) -> str:
        return self.raw.get("session_id", "")

    @property
    def assistant_texts(self) -> list[str]:
        """assistant 消息中的所有 text block"""
        if self.type != "assistant":
            return []
        texts = []
        for block in self.raw.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block["text"])
        return texts

    @property
    def tool_calls(self) -> list[dict]:
        """assistant 消息中的所有 tool_use block"""
        if self.type != "assistant":
            return []
        calls = []
        for block in self.raw.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(block)
        return calls

    @property
    def tool_results(self) -> list[dict]:
        """user 消息中的所有 tool_result block（CLI 自动生成）"""
        if self.type != "user":
            return []
        results = []
        content = self.raw.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results.append(block)
        return results

    @property
    def cost_usd(self) -> float:
        """本轮费用（仅 result 消息有效）"""
        return self.raw.get("total_cost_usd", 0.0)

    @property
    def num_turns(self) -> int:
        """本轮内部 LLM 调用次数（仅 result 消息有效）"""
        return self.raw.get("num_turns", 0)

    def __repr__(self) -> str:
        if self.is_init:
            return f"<ClaudeMessage INIT session={self.session_id[:8]}>"
        if self.is_result:
            return f"<ClaudeMessage RESULT ok={self.is_result_ok} '{self.result_text[:40]}'>"
        if self.is_assistant:
            texts = self.assistant_texts
            calls = self.tool_calls
            parts = [f"text:{t[:30]!r}" for t in texts] + [f"tool:{c['name']}" for c in calls]
            return f"<ClaudeMessage ASSISTANT {', '.join(parts) or 'thinking'}>"
        if self.is_task_event:
            return f"<ClaudeMessage TASK/{self.subtype} {self.raw.get('description', '')[:40]}>"
        return f"<ClaudeMessage {self.type}/{self.subtype}>"


# ── Controller ────────────────────────────────────────────────────────────────

class ClaudeController:
    """
    Claude CLI 进程控制器

    启动一个 Claude CLI 持久进程并通过 stream-json 协议双向通信。
    进程存活期间保持完整的 agent loop、多轮上下文、工具链、子代理、skills。

    基本用法：
        with ClaudeController(skip_permissions=True) as ctrl:
            result = ctrl.send("列出当前目录的文件")
            print(result.result_text)

    带回调：
        def on_msg(msg):
            if msg.is_assistant:
                print(msg.assistant_texts)

        ctrl = ClaudeController(on_message=on_msg, skip_permissions=True)
        ctrl.start()
        result = ctrl.send("分析代码架构", timeout=120)
        ctrl.stop()
    """

    def __init__(
        self,
        system_prompt: str = "",
        append_system_prompt: str = "",
        tools: list[str] = None,
        allowed_tools: list[str] = None,
        disallowed_tools: list[str] = None,
        permission_mode: str = None,
        skip_permissions: bool = False,
        bare: bool = False,
        resume: str = None,
        continue_session: bool = False,
        fork_session: bool = False,
        model: str = None,
        cwd: str = None,
        add_dirs: list[str] = None,
        on_message: Callable[["ClaudeMessage"], None] = None,
        transcript_path: str = None,
    ):
        self.cwd = cwd
        self.on_message = on_message

        self._cmd = [
            "claude",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
        ]
        if bare:
            # ⚠️  --bare 会破坏 OAuth 认证（keychain 永远不读）
            # OAuth 用户请改用 tools=[...] 参数替代
            # 仅在 ANTHROPIC_API_KEY 环境变量明确设置时可用
            self._cmd.append("--bare")
        if skip_permissions:
            self._cmd.append("--dangerously-skip-permissions")
        if system_prompt:
            self._cmd += ["--system-prompt", system_prompt]
        if append_system_prompt:
            self._cmd += ["--append-system-prompt", append_system_prompt]
        if tools:
            self._cmd += ["--tools", ",".join(tools)]
        if allowed_tools:
            self._cmd += ["--allowedTools", ",".join(allowed_tools)]
        if disallowed_tools:
            self._cmd += ["--disallowedTools", ",".join(disallowed_tools)]
        if permission_mode:
            self._cmd += ["--permission-mode", permission_mode]
        if model:
            self._cmd += ["--model", model]
        if resume:
            self._cmd += ["--resume", resume]
            if fork_session:
                self._cmd.append("--fork-session")
        elif continue_session:
            self._cmd.append("--continue")
            if fork_session:
                self._cmd.append("--fork-session")
        if add_dirs:
            for d in add_dirs:
                self._cmd += ["--add-dir", d]

        self._proc: Optional[subprocess.Popen] = None
        self._out_buf: list[str] = []
        self._err_buf: list[str] = []
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._session_id: Optional[str] = None
        self._transcript_path: Optional[str] = transcript_path
        self._transcript_file = None

        # Store kwargs for fork() — allows recreating controller with same config
        self._ctrl_kwargs = {
            "system_prompt": system_prompt,
            "append_system_prompt": append_system_prompt,
            "tools": tools,
            "allowed_tools": allowed_tools,
            "disallowed_tools": disallowed_tools,
            "permission_mode": permission_mode,
            "skip_permissions": skip_permissions,
            "bare": bare,
            "fork_session": fork_session,
            "model": model,
            "cwd": cwd,
            "add_dirs": add_dirs,
            "on_message": on_message,
        }

    # ── 生命周期 ─────────────────────────────────────

    def start(self, wait_init_timeout: float = 0) -> bool:
        """
        启动 Claude CLI 进程。

        启动前先调用 check_claude_available() 检查 binary 是否存在。
        如果 binary 不存在，抛 ClaudeBinaryNotFound。
        wait_init_timeout > 0 时，阻塞直到收到 system/init 消息。
        返回 True 表示启动成功。
        """
        check_claude_available()  # raises ClaudeBinaryNotFound if missing
        if self._transcript_path:
            self._transcript_file = open(self._transcript_path, "a", encoding="utf-8")
        env = {**os.environ, "TERM": "dumb"}
        if env.get("ANTHROPIC_AUTH_TOKEN") and not env.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = env["ANTHROPIC_AUTH_TOKEN"]
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=self.cwd,
        )
        threading.Thread(
            target=self._reader,
            args=(self._proc.stdout, self._out_buf),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._reader,
            args=(self._proc.stderr, self._err_buf),
            daemon=True,
        ).start()

        if self._proc.poll() is not None:
            return False

        # Claude CLI with stdin PIPE requires at least one input message before producing
        # any output (including init). Send an empty bootstrap message first.
        bootstrap_start_index = len(self._out_buf)
        self._write("")

        if wait_init_timeout > 0:
            if not self._wait_for_init(wait_init_timeout):
                return False

            # MiniMax/Claude CLI may emit a deferred "ready" result for the empty bootstrap
            # message. Drain that result here so the first real send() is not polluted by it.
            bootstrap_timeout = min(wait_init_timeout, 15.0)
            bootstrap_result = self._wait_result(bootstrap_start_index, bootstrap_timeout)
            if bootstrap_result and bootstrap_result.session_id:
                self._session_id = bootstrap_result.session_id
            return True

        return True

    def stop(self, timeout: float = 5.0):
        """终止 Claude CLI 进程"""
        if self._transcript_file:
            self._transcript_file.close()
            self._transcript_file = None
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    # ── 发送与接收 ───────────────────────────────────

    def send(self, text: str, timeout: float = 60.0) -> Optional[ClaudeMessage]:
        """
        发送一条用户消息，阻塞直到收到 result 消息。

        返回 result 消息（ClaudeMessage），超时返回 None。
        result.result_text 是 Claude 最终的回复文本。

        如果另一个 send() 正在等待 result，此调用立即抛 ClaudeSendConflictError。
        """
        if not self.alive:
            raise RuntimeError(f"Claude process (PID {self.pid}) is not running")

        if not self._send_lock.acquire(blocking=False):
            raise ClaudeSendConflictError("send already in flight")
        start_index = len(self._out_buf)
        self._write(text)
        return self.wait_for_result(start_index=start_index, timeout=timeout)

    def send_nowait(self, text: str):
        """发送消息，不等待结果（配合 wait_for_result 使用）"""
        if not self.alive:
            raise RuntimeError(f"Claude process (PID {self.pid}) is not running")
        if not self._send_lock.acquire(blocking=False):
            raise ClaudeSendConflictError("send already in flight")
        self._write(text)

    def wait_for_result(
        self,
        timeout: float = 60.0,
        start_index: int = 0,
    ) -> Optional[ClaudeMessage]:
        """
        等待 result 消息（配合 send_nowait 使用），完成后释放发送锁。

        当 send_nowait() 持有锁时，wait_for_result() 在所有退出路径
        （正常返回 / 超时 / 异常）都释放该锁。
        """
        try:
            return self._wait_result(start_index, timeout)
        finally:
            self._send_lock.release()

    def wait_for_tool_use(
        self,
        tool_name: str,
        timeout: float = 30.0,
        start_index: int = 0,
    ) -> Optional[dict]:
        """等待 Claude 调用特定工具，返回 tool_use block"""
        start = time.time()
        seen = start_index
        while time.time() - start < timeout:
            for line in self._out_buf[seen:]:
                msg = self._parse(line)
                if msg and msg.is_assistant:
                    for call in msg.tool_calls:
                        if call.get("name") == tool_name:
                            return call
            seen = len(self._out_buf)
            time.sleep(0.1)
        return None

    # ── 状态查询 ─────────────────────────────────────

    def get_messages(self) -> list[ClaudeMessage]:
        """返回到目前为止收到的所有解析后的消息"""
        messages = []
        for line in list(self._out_buf):
            msg = self._parse(line)
            if msg:
                messages.append(msg)
        return messages

    def get_init_message(self) -> Optional[ClaudeMessage]:
        """返回 system/init 消息（如果已收到）"""
        for msg in self.get_messages():
            if msg.is_init:
                return msg
        return None

    def get_available_tools(self) -> list[str]:
        """从 init 消息获取可用工具列表"""
        init = self.get_init_message()
        return init.raw.get("tools", []) if init else []

    def get_available_skills(self) -> list[str]:
        """从 init 消息获取可用 skill 列表"""
        init = self.get_init_message()
        return init.raw.get("slash_commands", []) if init else []

    def get_stderr(self) -> list[str]:
        """返回 stderr 内容（用于调试）"""
        return list(self._err_buf)

    def get_transcript_path(self) -> Optional[str]:
        """返回当前 transcript 文件路径，未设置时返回 None"""
        return self._transcript_path

    def get_tool_errors(self, start_index: int = 0) -> list[dict]:
        """
        返回指定消息区间内所有失败的 tool_result block。
        用于检测权限拒绝、文件不存在等工具执行失败。

        典型用法：
            idx = len(ctrl._out_buf)
            result = ctrl.send("任务")
            errors = ctrl.get_tool_errors(idx)
        """
        errors = []
        for line in self._out_buf[start_index:]:
            msg = self._parse(line)
            if msg and msg.is_tool_result:
                for block in msg.tool_results:
                    if block.get("is_error"):
                        errors.append(block)
        return errors

    def send_checked(
        self,
        text: str,
        timeout: float = 60.0,
    ) -> Tuple[Optional["ClaudeMessage"], list[dict]]:
        """
        send() 的增强版：同时返回 (result, tool_errors)。
        调用方可以同时检查 API 错误和工具执行失败。

        示例：
            result, tool_errors = ctrl.send_checked("写入文件")
            if not result.truly_succeeded:
                print("API 错误:", result.result_text)
            if tool_errors:
                print("工具失败:", tool_errors)
        """
        start_index = len(self._out_buf)
        result = self.send(text, timeout=timeout)
        tool_errors = self.get_tool_errors(start_index)
        return result, tool_errors

    # ── Session forking ─────────────────────────────────

    def fork(self) -> "ClaudeController":
        """
        创建一个从当前会话分支的新 ClaudeController。

        新控制器配置为从当前控制器的 session 恢复（resume），
        从而继承截至 fork 点的完整对话历史。

        原控制器不受影响。新控制器返回时处于停止状态，
        调用方需自行调用 start()。

        如果当前控制器没有 session_id（尚未 start 或 start 失败），抛出 RuntimeError。
        """
        if not self._session_id:
            raise RuntimeError(
                "cannot fork: no session_id (controller not started or session not established)"
            )
        forked = ClaudeController(resume=self._session_id, **self._ctrl_kwargs)
        return forked

    # ── 内部实现 ─────────────────────────────────────

    def _write(self, text: str):
        obj = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        }
        self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _wait_result(self, start_index: int, timeout: float) -> Optional[ClaudeMessage]:
        deadline = time.time() + timeout
        seen = start_index
        while time.time() < deadline:
            for line in self._out_buf[seen:]:
                msg = self._parse(line)
                if msg and msg.is_result:
                    if msg.session_id:
                        self._session_id = msg.session_id
                    return msg
            seen = len(self._out_buf)
            remaining = deadline - time.time()
            if remaining > 0:
                time.sleep(min(0.1, remaining))
        return None

    def _wait_for_init(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        seen = 0
        while time.time() < deadline:
            for line in self._out_buf[seen:]:
                msg = self._parse(line)
                if msg and msg.is_init:
                    if msg.session_id:
                        self._session_id = msg.session_id
                    return True
            seen = len(self._out_buf)
            time.sleep(0.05)
        return False

    def _reader(self, pipe, buf: list):
        for raw in iter(pipe.readline, ""):
            line = raw.rstrip("\n")
            if not line:
                continue
            with self._lock:
                buf.append(line)
            if pipe is self._proc.stdout:
                if self._transcript_file:
                    self._transcript_file.write(line + "\n")
                    self._transcript_file.flush()
                msg = self._parse(line)
                if msg and self.on_message:
                    try:
                        self.on_message(msg)
                    except Exception:
                        pass

    @staticmethod
    def _parse(line: str) -> Optional[ClaudeMessage]:
        try:
            obj = json.loads(line)
            return ClaudeMessage(
                type=obj.get("type", ""),
                subtype=obj.get("subtype", ""),
                raw=obj,
            )
        except (json.JSONDecodeError, Exception):
            return None

    # ── Context Manager ──────────────────────────────

    def __enter__(self) -> "ClaudeController":
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __repr__(self) -> str:
        status = "alive" if self.alive else "stopped"
        return f"<ClaudeController pid={self.pid} {status} session={self._session_id and self._session_id[:8]}>"
