# -*- coding: utf-8 -*-
"""Скрипт сборки отчета и графика из уже сохраненных трейсов.
НЕ делает сетевых запросов, ничего не стоит.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

TRACES_DIR = Path("traces")
IMG_DIR = Path("img")
IMG_DIR.mkdir(exist_ok=True)

CONFIGS = [
    ("strong_no_tools", "Сильная (Sonnet), без инструментов"),
    ("cheap_no_tools", "Дешёвая (GPT-4o-mini), без инструментов"),
    ("cheap_search_only", "Дешёвая, только поиск (web_search)"),
    ("cheap_best_agent", "Дешёвая, лучший агент (Wiki + DDG + PageFind)"),
    ("mid_best_agent", "Средняя (Haiku), лучший агент"),
]

def generate_report():
    assert TRACES_DIR.exists(), "Папка traces/ не найдена!"
    
    rows = []
    
    for cfg_id, cfg_name in CONFIGS:
        trace_files = list(TRACES_DIR.glob(f"*_{cfg_id}.json"))
        if not trace_files:
            print(f"Предупреждение: трейсы для {cfg_id} не найдены")
            continue
            
        total_tasks = len(trace_files)
        correct_count = 0
        total_cost = 0.0
        total_steps = 0
        total_time = 0.0
        
        for tf in trace_files:
            data = json.loads(tf.read_text(encoding="utf-8"))
            if data.get("is_correct"):
                correct_count += 1
            total_cost += data.get("cost", 0.0)
            total_steps += data.get("steps", 1)
            total_time += data.get("duration_sec", 0.0)
            
        accuracy = (correct_count / total_tasks) * 100
        cost_per_task = total_cost / total_tasks
        cost_per_correct = (total_cost / correct_count) if correct_count > 0 else 0.0
        avg_steps = total_steps / total_tasks
        avg_time = total_time / total_tasks
        
        rows.append({
            "name": cfg_name,
            "accuracy": round(accuracy, 1),
            "cost_task": round(cost_per_task, 5),
            "cost_correct": round(cost_per_correct, 5),
            "avg_steps": round(avg_steps, 2),
            "avg_time": round(avg_time, 2)
        })

    # Формируем красивую Markdown-таблицу вручную (без tabulate!)
    md = []
    md.append("| Конфигурация | Доля верных (%) | Цена задачи ($) | Цена верного ответа ($) | Среднее число шагов | Среднее время (с) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        md.append(f"| {r['name']} | {r['accuracy']}% | ${r['cost_task']:.5f} | ${r['cost_correct']:.5f} | {r['avg_steps']} | {r['avg_time']}s |")
        
    report_text = "\n".join(md)
    print("\n================== ИТОГОВЫЙ ОТЧЕТ (REPORT) ==================\n")
    print(report_text)
    
    Path("report.md").write_text(report_text, encoding="utf-8")
    print("\nТаблица сохранена в report.md")

    plt.figure(figsize=(10, 6))
    colors = ["#C43C3C", "#787882", "#00838F", "#5436A3", "#F09000"]

    for i, r in enumerate(rows):
        plt.scatter(
            r["cost_task"], 
            r["accuracy"], 
            color=colors[i % len(colors)], 
            s=130, 
            zorder=3,
            label=r["name"]
        )
        plt.annotate(
            r["name"].split("(")[0].strip(),
            (r["cost_task"], r["accuracy"]),
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
    generate_report()