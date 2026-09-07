"""Render a saved trace payload (the same dict TraceLogger writes to JSON) as a standalone HTML page.

One HTML file is written next to each JSON trace (none.json -> none.html, injection_task_1.json ->
injection_task_1.html). It surfaces the IPIGuard-specific components -- the pre-plan, the committed DAG,
the runtime tool-call gating (executed / deferred), and the DAG event timeline -- alongside the raw
message stream, so a run can be read at a glance without opening the JSON.
"""

from __future__ import annotations

import html
import json
from typing import Any

_CSS = """
:root { --sys:#64748b; --usr:#2563eb; --ast:#16a34a; --tool:#d97706; --dag:#7c3aed; --plan:#0891b2; --blk:#b91c1c; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:"Helvetica Neue", Arial, sans-serif; background:#fff; display:flex; justify-content:center; padding:24px; }
.figure { width:960px; max-width:100%; }
.title { font-size:15px; font-weight:700; color:#111; margin-bottom:4px; }
.title span { font-weight:400; color:#666; font-size:13px; }
.subnote { font-size:11.5px; color:#94a3b8; margin-bottom:14px; }
.ok { color:#16a34a; font-weight:700; } .bad { color:#b91c1c; font-weight:700; }
.msg { border:1.5px solid; border-radius:8px; padding:10px 14px; margin-bottom:10px; font-size:12.5px; line-height:1.45; color:#1f2937; }
.role { display:inline-block; font-size:10.5px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; color:#fff; border-radius:4px; padding:2px 8px; margin-bottom:6px; }
.sys { border-color:var(--sys); background:#f8fafc; } .sys .role { background:var(--sys); }
.usr { border-color:var(--usr); background:#eff6ff; } .usr .role { background:var(--usr); }
.ast { border-color:var(--ast); background:#f0fdf4; } .ast .role { background:var(--ast); }
.tool { border-color:var(--tool); background:#fffbeb; margin-left:32px; } .tool .role { background:var(--tool); }
.dagcard { border-color:var(--dag); background:#f5f3ff; } .dagcard .role { background:var(--dag); }
.plancard { border-color:var(--plan); background:#ecfeff; } .plancard .role { background:var(--plan); }
.box { border-radius:6px; padding:6px 10px; margin-top:6px; display:block; white-space:pre-wrap; overflow-x:auto; font-family:"SF Mono", Menlo, Consolas, monospace; font-size:11.5px; }
.dagbox { background:#ede9fe; border:1px solid #c4b5fd; color:#4c1d95; }
.planbox { background:#cffafe; border:1px solid #67e8f9; color:#155e75; }
.call { background:#dcfce7; border:1px solid #86efac; color:#14532d; }
.result { background:#fef3c7; border:1px solid #fcd34d; color:#78350f; }
.errresult { background:#fee2e2; border:1px solid #fca5a5; color:#7f1d1d; }
.deferred { background:#fee2e2; border:1px solid #fca5a5; color:#7f1d1d; }
.executed { background:#dcfce7; border:1px solid #86efac; color:#14532d; }
.evline { font-size:11px; color:#6d28d9; padding:1px 0; white-space:pre-wrap; }
.flag { font-size:11.5px; color:#475569; margin-top:8px; }
.small { font-size:11px; color:#64748b; }
"""


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _pretty(x: Any) -> str:
    try:
        return json.dumps(x, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(x)


def _content_to_text(content: Any) -> str:
    """messages content is a list of {type, content} blocks (0.1.35+), or a str, or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                parts.append(str(blk.get("content", "")))
            else:
                parts.append(str(blk))
        return "\n".join(p for p in parts if p)
    return str(content)


def _flag(val: Any) -> str:
    if val is True:
        return '<span class="ok">✓</span>'
    if val is False:
        return '<span class="bad">✗</span>'
    return f'<span class="small">{_esc(val)}</span>'


def _render_tool_calls(tool_calls: list) -> str:
    out = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or tc.get("function_name") or tc.get("name") or "?"
            args = tc.get("args", tc.get("arguments", {}))
        else:
            fn = getattr(tc, "function", "?")
            args = getattr(tc, "args", {})
        out.append(f'<code class="box call">→ {_esc(fn)}({_esc(_pretty(args))})</code>')
    return "".join(out)


def _render_messages(messages: list) -> str:
    parts = []
    for m in messages or []:
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        cls = {"system": "sys", "user": "usr", "assistant": "ast", "tool": "tool"}.get(role, "sys")
        text = _content_to_text(m.get("content") if isinstance(m, dict) else None)
        inner = [f'<span class="role">{_esc(role)}</span>']
        if role == "tool":
            # Put the tool's returned data (or error) into the result box, not a plain div.
            err = m.get("error") if isinstance(m, dict) else None
            fn = ""
            tc = m.get("tool_call") if isinstance(m, dict) else None
            if isinstance(tc, dict):
                fn = tc.get("function") or tc.get("function_name") or ""
            elif tc is not None:
                fn = getattr(tc, "function", "")
            if fn:
                inner.append(f'<div class="small">{_esc(fn)} returned:</div>')
            if err:
                inner.append(f'<code class="box errresult">error: {_esc(err)}</code>')
            inner.append(f'<code class="box result">{_esc(text) if text.strip() else "(empty)"}</code>')
        else:
            if text.strip():
                inner.append(f"<div>{_esc(text)}</div>")
            if role == "assistant":
                tcs = m.get("tool_calls") if isinstance(m, dict) else None
                if tcs:
                    inner.append(_render_tool_calls(tcs))
        parts.append(f'<div class="msg {cls}">{"".join(inner)}</div>')
    return "".join(parts)


def _render_preplan(data: dict) -> str:
    pre = data.get("pre_plan")
    if not pre:
        return ""
    text = pre if isinstance(pre, str) else _pretty(pre)
    return (
        '<div class="msg plancard"><span class="role">IPIGuard · pre-plan</span>'
        '<div class="small">Task-understanding pass run before planning: explicit/implicit requirements, '
        'known info, and "missing information" it will try to gather with tools.</div>'
        f'<code class="box planbox">{_esc(text)}</code></div>'
    )


def _render_plan(data: dict) -> str:
    """Top card: only the committed plan (initial DAG). Runtime activity goes below the messages."""
    initial = data.get("initial_dag")
    if not initial:
        return ""
    return (
        '<div class="msg dagcard"><span class="role">IPIGuard · committed plan (initial DAG)</span>'
        '<div class="small">The tool-dependency plan fixed BEFORE any tool runs. Runtime-discovered '
        'calls are gated against it below (read-only execute, state-changing are deferred).</div>'
        f'<code class="box dagbox">{_esc(_pretty(initial))}</code></div>'
    )


def _render_runtime(data: dict) -> str:
    """Runtime trace section (after the message stream): expanded DAG, gated runtime calls, events."""
    initial = data.get("initial_dag")
    expanded = data.get("expanded_dag")
    events = data.get("dag_events")
    new_calls = data.get("new_tool_calls")
    if not any([expanded and expanded != initial, events, new_calls]):
        return ""

    body = ['<div class="msg dagcard"><span class="role">IPIGuard · runtime trace</span>']
    if new_calls:
        rows = []
        for c in new_calls:
            if not isinstance(c, dict):
                continue
            fn = c.get("function_name") or c.get("function") or "?"
            status = c.get("status", "executed" if c.get("whitelisted") else "deferred")
            cls = "executed" if status == "executed" else "deferred"
            args = c.get("args", {})
            rows.append(
                f'<code class="box {cls}">[{_esc(status)}] {_esc(fn)}({_esc(_pretty(args))})</code>'
            )
        if rows:
            body.append('<div class="small">Runtime-discovered tool calls (gating):</div>')
            body.extend(rows)
    if expanded and expanded != initial:
        body.append('<div class="small" style="margin-top:6px">Expanded DAG (after arg resolution):</div>')
        body.append(f'<code class="box dagbox">{_esc(_pretty(expanded))}</code>')
    if events:
        ev_lines = []
        for e in events:
            if isinstance(e, dict):
                ev = e.get("event", "?")
                fn = e.get("function", "")
                extra = e.get("status") or e.get("arg") or ""
                ev_lines.append(_esc(f"• {ev}  {fn}  {extra}".rstrip()))
        if ev_lines:
            body.append('<div class="small" style="margin-top:6px">DAG event timeline:</div>')
            body.append('<code class="box dagbox">' + "\n".join(ev_lines) + "</code>")
    body.append("</div>")
    return "".join(body)


def render_trace_html(data: dict) -> str:
    suite = data.get("suite_name", "?")
    utid = data.get("user_task_id", "?")
    itid = data.get("injection_task_id")
    attack = data.get("attack_type") or "none"
    pipeline = data.get("pipeline_name", "?")
    util = data.get("utility")
    sec = data.get("security")
    dur = data.get("duration")
    itoks = data.get("input_tokens")
    otoks = data.get("output_tokens")
    err = data.get("error")

    sub_bits = []
    if dur is not None:
        try:
            sub_bits.append(f"{float(dur):.2f}s")
        except (TypeError, ValueError):
            pass
    if itoks is not None or otoks is not None:
        sub_bits.append(f"tokens in {itoks} / out {otoks}")
    if itid:
        sub_bits.append(f"injection: {itid}")
    if err:
        sub_bits.append(f"error: {err}")
    subnote = " · ".join(_esc(b) for b in sub_bits)

    title = (
        f"Execution Trace <span>— suite: {_esc(suite)} · {_esc(utid)} · attack: {_esc(attack)} · "
        f"{_esc(pipeline)} · utility {_flag(util)} security {_flag(sec)}</span>"
    )

    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
        f"<title>Trace — {_esc(suite)} / {_esc(utid)} / {_esc(attack)}</title>"
        f"<style>{_CSS}</style></head><body><div class=\"figure\">"
        f'<div class="title">{title}</div>'
        f'<div class="subnote">{subnote}</div>'
        f"{_render_preplan(data)}"
        f"{_render_plan(data)}"
        f"{_render_messages(data.get('messages', []))}"
        f"{_render_runtime(data)}"
        f'<div class="flag">utility {_flag(util)} · security {_flag(sec)}</div>'
        "</div></body></html>"
    )
