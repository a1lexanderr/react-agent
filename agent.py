# -*- coding: utf-8 -*-
"""ИИ-агент (ReAct) для работы со свежими вопросами через OpenRouter.

Включает:
- ReAct-цикл с вызовом инструментов из tools.py
- Детектор зацикливания (seen tool calls)
- Леджер затрат (токены и стоимость в долларах)
- Мягкую остановку (soft finish) по шаблону FINAL: <ответ>
- Логирование полного трейса в папку traces/
- Проверку ответа (нормализация)
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from tools import REGISTRY

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_KEY, "Не найден OPENROUTER_API_KEY в файле .env!"

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "HTTP-Referer": "https://postypashki.ru",
    "X-Title": "agents-course-homework"
}


MODELS = {
    "cheap": "openai/gpt-4o-mini",
    "mid": "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6"
}

TRACES_DIR = Path("traces")
TRACES_DIR.mkdir(exist_ok=True)


AGENT_SYSTEM = (
    "You are an autonomous AI research agent answering questions about events from 2025 and 2026. "
    "Rules you must strictly follow:\n"
    "1. Never answer from memory alone. Always verify facts using your search tools.\n"
    "2. For general overview and Wikipedia knowledge, use 'web_search'.\n"
    "3. If an article is long (like '2026 in computing' or country overviews), use 'page_find' with title and keywords.\n"
    "4. For news, recent startup releases, and non-Wikipedia sources, use 'ddg_search'.\n"
    "5. For calculations or date differences, use 'calculator' or 'python_exec'.\n"
    "6. Do not repeat the exact same tool call with the same arguments.\n"
    "7. When you have enough information, finish your reasoning and output the final answer as the very last line in the exact format:\n"
    "FINAL: <short unambiguous answer>"
)


def call_openrouter(body: dict, attempts: int = 5) -> dict:
    """Вызов OpenRouter с повторами при сетевых сбоях и ошибках 429/5xx."""
    last_err = "нет ответа"
    for attempt in range(attempts):
        try:
            r = requests.post(CHAT_URL, json=body, headers=HEADERS, timeout=90)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}: {r.text[:120]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last_err = f"RequestException: {e}"
        time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"Сбой OpenRouter: {last_err}")


def normalize(text: Any) -> str:
    """Нормализует текст для честной проверки совпадения."""
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(expected: str, actual: str) -> bool:
    """Проверяет, содержится ли нормализованный эталон в ответе агента."""
    exp_norm = normalize(expected)
    act_norm = normalize(actual)
    return bool(exp_norm) and exp_norm in act_norm


def run_agent(
    question: str,
    model_name: str = MODELS["cheap"],
    allowed_tools: Optional[List[str]] = None,
    max_steps: int = 7,
    expected_answer: Optional[str] = None,
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """Запускает агента на одном вопросе."""
    start_time = time.time()
    

    tool_schemas = []
    if allowed_tools:
        for t_name in allowed_tools:
            if t_name in REGISTRY:
                tool_schemas.append(REGISTRY[t_name][1])


    system_prompt = AGENT_SYSTEM if allowed_tools else (
        "Answer the question with a short, unambiguous answer only. "
        "Finish your answer on the last line with: FINAL: <short answer>"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    total_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    seen_calls = set()
    final_answer = ""
    stop_reason = "max_steps"


    for step in range(max_steps):
        body = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "usage": {"include": True}
        }
        if tool_schemas:
            body["tools"] = tool_schemas

        try:
            resp = call_openrouter(body)
        except RuntimeError as e:
            final_answer = f"error: {e}"
            stop_reason = "api_error"
            break


        usage = resp.get("usage") or {}
        total_cost += usage.get("cost") or 0.0
        prompt_tokens += usage.get("prompt_tokens") or 0
        completion_tokens += usage.get("completion_tokens") or 0

        msg = resp["choices"][0]["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []


        if not tool_calls:
            content = msg.get("content") or ""
            m = re.search(r"FINAL:\s*(.+)", content, re.I)
            final_answer = m.group(1).strip() if m else content.strip()
            stop_reason = "completed"
            break


        if not allowed_tools:
            content = msg.get("content") or ""
            final_answer = content.strip()
            stop_reason = "no_tools"
            break


        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args_raw = tc["function"]["arguments"]


            call_key = (fn_name, fn_args_raw)
            if call_key in seen_calls:
                tool_output = "Цикл: вы уже вызывали этот инструмент с такими аргументами. Смените стратегию или ответьте FINAL:"
            else:
                seen_calls.add(call_key)
                if fn_name in REGISTRY:
                    func = REGISTRY[fn_name][0]
                    try:
                        args = json.loads(fn_args_raw)
                        tool_output = str(func(**args))
                    except Exception as e:
                        tool_output = f"Ошибка выполнения инструмента: {e}"
                else:
                    tool_output = f"Неизвестный инструмент: {fn_name}"


            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_output[:3000]
            })


    if not final_answer:
        last_content = messages[-1].get("content") or ""
        m = re.search(r"FINAL:\s*(.+)", last_content, re.I)
        final_answer = m.group(1).strip() if m else last_content.strip()

    duration = round(time.time() - start_time, 2)
    correct = is_correct(expected_answer, final_answer) if expected_answer else False

    result = {
        "id": trace_id,
        "question": question,
        "expected": expected_answer,
        "answer": final_answer,
        "is_correct": correct,
        "steps": len([m for m in messages if m.get("role") == "tool"]) + 1,
        "cost": round(total_cost, 5),
        "tokens_in": prompt_tokens,
        "tokens_out": completion_tokens,
        "duration_sec": duration,
        "stop_reason": stop_reason,
        "messages": messages
    }


    if trace_id:
        trace_file = TRACES_DIR / f"{trace_id}.json"
        trace_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


if __name__ == "__main__":
    print("Проверка работы агента на одном тестовом вопросе...")
    test_q = "What AI browser based on Chromium did Perplexity AI release on July 9, 2025?"
    test_ans = "Comet"
    
    res = run_agent(
        question=test_q,
        model_name=MODELS["cheap"],
        allowed_tools=["web_search", "page_find", "ddg_search"],
        max_steps=5,
        expected_answer=test_ans,
        trace_id="test_run"
    )
    
    print("\n--- Результат прогона ---")
    print(f"Вопрос: {res['question']}")
    print(f"Эталон: {res['expected']}")
    print(f"Ответ агента: {res['answer']}")
    print(f"Верно: {'ДА' if res['is_correct'] else 'НЕТ'}")
    print(f"Шагов: {res['steps']}")
    print(f"Потрачено: ${res['cost']}")
    print(f"Время: {res['duration_sec']} сек")
    print(f"Трейс сохранен в: traces/{res['id']}.json")