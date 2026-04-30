"""
deep_research/core/llm.py
Unified LLM interface: Claude API, OpenAI API, Gemini API,
plus CLI-based backends (claude, gemini, codex, or any custom command).

Provider strings:
  "claude"      — Anthropic Python SDK (needs ANTHROPIC_API_KEY)
  "openai"      — OpenAI Python SDK   (needs OPENAI_API_KEY)
  "gemini"      — Google genai SDK    (needs GEMINI_API_KEY)
  "claude-cli"  — shells out to `claude --print` (Claude Code CLI, no key needed)
  "gemini-cli"  — shells out to `gemini`         (Gemini CLI, no key needed)
  "codex"       — shells out to `codex`          (OpenAI Codex CLI)
  "cli:<cmd>"   — shells out to any custom command, e.g. "cli:ollama run llama3"
"""
from __future__ import annotations
import subprocess
import shutil
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import cfg


class LLMError(Exception):
    pass


# CLI backends that are known + their default invocation
_CLI_COMMANDS: dict[str, list[str]] = {
    "claude-cli": ["claude", "--print"],
    "gemini-cli": ["gemini"],
    "codex":      ["codex"],
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def complete(
    prompt: str,
    system: str = "You are a precise research assistant.",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    config=None,
) -> str:
    c = config or cfg
    p, m = c.resolve_model(provider)
    if model:
        m = model

    if p == "claude":
        return _claude(prompt, system, m, max_tokens, temperature, c)
    elif p == "openai":
        return _openai(prompt, system, m, max_tokens, temperature, c)
    elif p == "gemini":
        return _gemini(prompt, system, m, max_tokens, temperature, c)
    elif p in _CLI_COMMANDS:
        return _cli(prompt, system, _CLI_COMMANDS[p], m if m else None)
    elif p.startswith("cli:"):
        cmd = p[4:].split()
        return _cli(prompt, system, cmd, None)
    else:
        raise LLMError(
            f"Unknown provider: '{p}'. "
            "Use 'claude', 'openai', 'gemini', 'claude-cli', 'gemini-cli', 'codex', or 'cli:<command>'."
        )


# ── API backends ──────────────────────────────────────────────────────────────

def _claude(prompt, system, model, max_tokens, temperature, c) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=c.anthropic_api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _openai(prompt, system, model, max_tokens, temperature, c) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=c.openai_api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def _gemini(prompt, system, model, max_tokens, temperature, c) -> str:
    from genai import Client
    from genai.types import GenerateContentConfig
    
    client = Client(api_key=c.gemini_api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return resp.text


# ── CLI subprocess backend ────────────────────────────────────────────────────

def _cli(prompt: str, system: str, cmd: list[str], model: Optional[str]) -> str:
    """
    Send prompt to any CLI tool via stdin, return stdout.

    The system prompt is prepended to the user prompt separated by two newlines,
    since most CLIs don't have a separate system argument.
    """
    full_input = f"{system}\n\n{prompt}" if system else prompt
    input_data = full_input

    # Some CLIs accept a model flag and use arguments instead of stdin
    run_cmd = list(cmd)
    if cmd[0] == "claude":
        run_cmd = ["claude", "-p"]
        if system:
            run_cmd.extend(["--system-prompt", system])
        if model:
            run_cmd.extend(["--model", model])
        run_cmd.append(prompt)
        input_data = None
    elif cmd[0] == "gemini":
        run_cmd = ["gemini", "-p", full_input]
        if model:
            run_cmd.extend(["--model", model])
        input_data = None
    else:
        if model and cmd[0] in ("claude",):
            run_cmd += ["--model", model]

    exe = shutil.which(run_cmd[0])
    if exe is None:
        raise LLMError(
            f"CLI not found: '{run_cmd[0]}'. "
            "Make sure it is installed and on your PATH."
        )
    run_cmd[0] = exe

    try:
        result = subprocess.run(
            run_cmd,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError:
        raise LLMError(
            f"CLI not found: '{cmd[0]}'. "
            "Make sure it is installed and on your PATH."
        )
    except subprocess.TimeoutExpired:
        raise LLMError(f"CLI '{cmd[0]}' timed out after 180s.")

    if result.returncode != 0:
        err_msg = result.stderr.strip() or result.stdout.strip()
        raise LLMError(f"CLI '{cmd[0]}' exited {result.returncode}: {err_msg}")

    output = result.stdout.strip()
    if not output:
        raise LLMError(f"CLI '{cmd[0]}' returned empty output.")
    return output
