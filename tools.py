# -*- coding: utf-8 -*-
"""Инструменты (скиллы) для ИИ-агента и их тесты на 3 случая:
1. Нормальный вход
2. Пустой результат
3. Ошибка (не роняет скрипт, возвращается текстом)
"""
import ast
import operator
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Tuple

import requests
from pydantic import BaseModel, Field
from ddgs import DDGS

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "agents-course-homework/1.0 (educational project)"}


class SearchArgs(BaseModel):
    query: str = Field(description="Короткий поисковый запрос по Википедии: имя, термин, событие")

class PageFindArgs(BaseModel):
    title: str = Field(description="Точный заголовок статьи Википедии (например, '2026 in computing')")
    keywords: str = Field(description="2-4 ключевых слова для поиска строк внутри статьи через пробел")

class DDGSearchArgs(BaseModel):
    query: str = Field(description="Поисковый запрос в интернет (новости, блоги, свежие факты)")

class CalculatorArgs(BaseModel):
    expression: str = Field(description="Математическое выражение для вычисления (например, '24 * 60')")

class PythonExecArgs(BaseModel):
    code: str = Field(description="Код на Python; вывод должен быть напечатан через print()")


def wiki_request(params: dict, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        try:
            r = requests.get(WIKI_API, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=12)
            if r.status_code == 200:
                return r.json().get("query", {})
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return {}


def web_search(query: str) -> str:
    """Ищет статью в англоязычной Википедии и возвращает вступление."""
    if not query or not query.strip():
        return "Пустой поисковый запрос"
    try:
        data = wiki_request({"action": "query", "list": "search", "srsearch": query, "srlimit": 3})
        hits = data.get("search", [])
        if not hits:
            return "Ничего не найдено"

        title = hits[0]["title"]
        page_data = wiki_request({
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "titles": title,
            "redirects": 1
        })
        pages = page_data.get("pages", {})
        extract = next(iter(pages.values())).get("extract", "") if pages else ""
        text = " ".join(extract.split())[:3000]
        
        others = ", ".join(h["title"] for h in hits[1:])
        out = f"[{title}] {text}"
        if others:
            out += f" | Другие статьи: {others}"
        return out if text else "Статья найдена, но текст пуст"
    except Exception as e:
        return f"Ошибка поиска в Википедии: {e}"


def page_find(title: str, keywords: str) -> str:
    """Ищет строки с ключевыми словами в полном тексте длинной статьи Википедии."""
    if not title or not title.strip():
        return "Не указан заголовок статьи"
    if not keywords or not keywords.strip():
        return "Не указаны ключевые слова"
    try:
        page_data = wiki_request({
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "redirects": 1
        })
        pages = page_data.get("pages", {})
        if not pages or "-1" in pages:
            return "Страница не найдена"

        text = next(iter(pages.values())).get("extract", "")
        if not text:
            return "Текст статьи пуст"

        words = [w.lower() for w in re.findall(r"\w+", keywords) if len(w) > 2]
        if not words:
            return "Ключевые слова слишком короткие"

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        threshold = max(1, (len(words) + 1) // 2)
        hits = [l for l in lines if sum(w in l.lower() for w in words) >= threshold]

        if not hits:
            return "Ключевые слова не найдены на странице"
        return "\n".join(hits[:8])[:3500]
    except Exception as e:
        return f"Ошибка чтения статьи: {e}"


def ddg_search(query: str) -> str:
    """Ищет информацию в интернете через DuckDuckGo."""
    if DDGS is None:
        return "Ошибка: библиотека ddgs не установлена"
    if not query or not query.strip():
        return "Пустой поисковый запрос"
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "Ничего не найдено"
        formatted = [f"[{r.get('title')}] {r.get('body')} ({r.get('href')})" for r in results]
        return "\n\n".join(formatted)[:3500]
    except Exception as e:
        return f"Ошибка поиска DuckDuckGo: {e}"


def calculator(expression: str) -> str:
    """Безопасный калькулятор базовых арифметических выражений."""
    if not expression or not expression.strip():
        return "Пустое выражение"
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("Неподдерживаемая операция")

    try:
        expr = expression.strip().replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree.body)
        return str(round(result, 6) if isinstance(result, float) else result)
    except ZeroDivisionError:
        return "Ошибка: деление на ноль"
    except Exception as e:
        return f"Ошибка вычисления: {e}"


def python_exec(code: str) -> str:
    """Выполняет код Python в изолированном процессе и возвращает stdout."""
    if not code or not code.strip():
        return "Пустой код"
    try:
        res = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        out = (res.stdout + res.stderr).strip()
        return out[:3000] if out else "Код выполнился, но вывод пустой"
    except subprocess.TimeoutExpired:
        return "Ошибка: превышен таймаут выполнения (5 сек)"
    except Exception as e:
        return f"Ошибка запуска Python: {e}"


REGISTRY: Dict[str, Tuple[Callable, dict]] = {}

def register_tool(func: Callable, args_model: type[BaseModel], description: str):
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": args_model.model_json_schema()
        }
    }
    REGISTRY[func.__name__] = (func, schema)

register_tool(web_search, SearchArgs, "Ищет статью в Википедии и возвращает краткое содержание вступления.")
register_tool(page_find, PageFindArgs, "Ищет строки по ключевым словам в теле длинной статьи Википедии (например, '2026 in computing').")
register_tool(ddg_search, DDGSearchArgs, "Поиск в интернете через DuckDuckGo (новости, компании, внешние источники).")
register_tool(calculator, CalculatorArgs, "Калькулятор для математических операций (поддерживает +, -, *, /, **, %).")
register_tool(python_exec, PythonExecArgs, "Выполняет скрипт Python и возвращает то, что напечатано в print().")


def run_tests():
    print("Запуск тестов инструментов...")

    assert calculator("2 + 2 * 2") == "6", "calc normal fail"
    assert calculator("") == "Пустое выражение", "calc empty fail"
    assert "деление на ноль" in calculator("1 / 0"), "calc error fail"

    assert python_exec("print(2 ** 8)") == "256", "py normal fail"
    assert python_exec("x = 10") == "Код выполнился, но вывод пустой", "py empty fail"
    assert "ZeroDivisionError" in python_exec("1 / 0"), "py error fail"

    res = web_search("Python (programming language)")
    assert "[Python (programming language)]" in res, "wiki normal fail"
    assert web_search("   ") == "Пустой поисковый запрос", "wiki empty fail"
    assert web_search("qzzxxw123456789nonsense") == "Ничего не найдено", "wiki missing fail"

    res_pf = page_find("Python (programming language)", "Guido Rossum")
    assert "Guido" in res_pf or "Rossum" in res_pf, "page_find normal fail"
    assert page_find("Python (programming language)", "   ") == "Не указаны ключевые слова", "page_find empty fail"
    assert page_find("NonExistentArticle1234567890", "test") == "Страница не найдена", "page_find error fail"

    if DDGS is not None:
        ddg_res = ddg_search("Python programming language")
        assert len(ddg_res) > 30, f"ddg normal fail: {ddg_res}"
        assert ddg_search("") == "Пустой поисковый запрос", "ddg empty fail"
        assert "Ошибка" in ddg_search(None) or "Пустой" in ddg_search(None), "ddg error fail"

    print("Все тесты инструментов успешно пройдены (5/5)!")

if __name__ == "__main__":
    run_tests()