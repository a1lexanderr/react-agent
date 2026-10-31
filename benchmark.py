# -*- coding: utf-8 -*-
"""Скрипт бенчмарка для домашки:
1. Прогоняет 5 обязательных конфигураций по data/fresh.jsonl
2. Логирует трейсы каждого вопроса в traces/
3. Формирует итоговую таблицу метрик
4. Строит график money_chart.png
"""
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from agent import run_agent, MODELS

DATA_FILE = Path("data/fresh.jsonl")
IMG_DIR = Path("img")
IMG_DIR.mkdir(exist_ok=True)


CONFIGURATIONS = [
    {
        "id": "strong_no_tools",
        "name": "Сильная (Sonnet), без инструментов",
        "model": MODELS["strong"],
        "tools": None,
        "max_steps": 1
    },
    {
        "id": "cheap_no_tools",
        "name": "Дешёвая (GPT-4o-mini), без инструментов",
        "model": MODELS["cheap"],
        "tools": None,
        "max_steps": 1
    },
    {
        "id": "cheap_search_only",
        "name": "Дешёвая, только поиск (web_search)",
        "model": MODELS["cheap"],
        "tools": ["web_search"],
        "max_steps": 5
    },
    {
        "id": "cheap_best_agent",
        "name": "Дешёвая, лучший агент (Wiki + DDG + PageFind)",
        "model": MODELS["cheap"],
        "tools": ["web_search", "page_find", "ddg_search", "calculator", "python_exec"],
        "max_steps": 7
    },
    {
        "id": "mid_best_agent",
        "name": "Средняя (Haiku), лучший агент",
        "model": MODELS["mid"],
        "tools": ["web_search", "page_find", "ddg_search", "calculator", "python_exec"],
        "max_steps": 7
    }
]


def load_dataset():
    assert DATA_FILE.exists(), f"Файл {DATA_FILE} не найден!"
    rows = []
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_benchmark():
    dataset = load_dataset()
    print(f"Загружено {len(dataset)} вопросов из {DATA_FILE}\n")

    summary_rows = []

    for cfg in CONFIGURATIONS:
        print(f"==================================================")
        print(f"Запуск конфигурации: {cfg['name']}")
        print(f"==================================================")

        correct_count = 0
        total_cost = 0.0
        total_steps = 0
        total_time = 0.0

        for idx, row in enumerate(dataset, 1):
            q_id = row.get("id", f"q{idx}")
            trace_id = f"{q_id}_{cfg['id']}"

            res = run_agent(
                question=row["question"],
                model_name=cfg["model"],
                allowed_tools=cfg["tools"],
                max_steps=cfg["max_steps"],
                expected_answer=row["answer"],
                trace_id=trace_id
            )

            is_ok = res["is_correct"]
            correct_count += int(is_ok)
            total_cost += res["cost"]
            total_steps += res["steps"]
            total_time += res["duration_sec"]

            status = "✓ OK" if is_ok else "✗ FAIL"
            print(f"[{idx:02d}/{len(dataset)}] {status} | {row['question'][:65]}... -> {res['answer'][:25]} (${res['cost']:.4f})")

        total_tasks = len(dataset)
        accuracy = (correct_count / total_tasks) * 100
        cost_per_task = total_cost / total_tasks
        cost_per_correct = (total_cost / correct_count) if correct_count > 0 else 0.0
        avg_steps = total_steps / total_tasks
        avg_time = total_time / total_tasks

        summary_rows.append({
            "Конфигурация": cfg["name"],
            "Доля верных (%)": round(accuracy, 1),
            "Цена задачи ($)": round(cost_per_task, 5),
            "Цена верного ответа ($)": round(cost_per_correct, 5),
            "Среднее число шагов": round(avg_steps, 2),
            "Среднее время (с)": round(avg_time, 2),
            "_accuracy_raw": accuracy,
            "_cost_task_raw": cost_per_task
        })
        print(f"\nИтог по {cfg['name']}: Верно {accuracy:.1f}%, Потрачено всего: ${total_cost:.4f}\n")

    df = pd.DataFrame(summary_rows)

    print("\n================== ИТОГОВЫЙ ОТЧЕТ (REPORT) ==================")
    report_cols = ["Конфигурация", "Доля верных (%)", "Цена задачи ($)", "Цена верного ответа ($)", "Среднее число шагов", "Среднее время (с)"]
    print(df[report_cols].to_markdown(index=False))


    Path("report.md").write_text(df[report_cols].to_markdown(index=False), encoding="utf-8")
    print("\nТаблица сохранена в report.md")

    plot_chart(df)


def plot_chart(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    colors = ["#C43C3C", "#787882", "#00838F", "#5436A3", "#F09000"]

    for i, row in df.iterrows():
        plt.scatter(
            row["Цена задачи ($)"], 
            row["Доля верных (%)"], 
            color=colors[i % len(colors)], 
            s=120, 
            zorder=3,
            label=row["Конфигурация"]
        )
        plt.annotate(
            row["Конфигурация"].split("(")[0].strip(),
            (row["Цена задачи ($)"], row["Доля верных (%)"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9
        )

    plt.title("Money Chart: Качество vs Стоимость", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Цена задачи, $", fontsize=11)
    plt.ylabel("Доля верных ответов, %", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(0, 105)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    plt.savefig("money_chart.png", dpi=200)
    plt.savefig(IMG_DIR / "money_chart.png", dpi=200)
    print("График сохранен в money_chart.png и img/money_chart.png")


if __name__ == "__main__":
    run_benchmark()