"""Console helpers for Stage II paper-format data preparation."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


def resolve_num_workers(value: int) -> int:
    """Map prep ``num_workers`` (0 = auto) to a concrete worker count."""
    if value < 0:
        raise ValueError(f"num_workers must be >= 0, got {value}")
    if value == 0:
        return max(1, min(8, os.cpu_count() or 1))
    return value


class Stage2PrepReporter:
    """Rich-backed progress and summary output for ``prepare_stage2_paper``."""

    def __init__(self, *, use_rich: bool = True) -> None:
        self.use_rich = use_rich and sys.stdout.isatty()
        self.console = Console()

    def section(self, title: str) -> None:
        if self.use_rich:
            self.console.rule(f"[bold]{title}")
        else:
            print(f"\n=== {title} ===")

    def info(self, message: str) -> None:
        if self.use_rich:
            self.console.print(message)
        else:
            print(message)

    def warn(self, message: str) -> None:
        if self.use_rich:
            self.console.print(f"[yellow]{message}[/yellow]")
        else:
            print(f"WARNING: {message}")

    def print_plan(self, rows: list[tuple[str, str]], *, title: str = "Stage II Paper Prep") -> None:
        if self.use_rich:
            table = Table(title=title, show_header=True, header_style="bold")
            table.add_column("Setting", style="cyan")
            table.add_column("Value")
            for key, value in rows:
                table.add_row(key, value)
            self.console.print(table)
            return

        print(f"=== {title} ===")
        for key, value in rows:
            print(f"  {key}: {value}")

    def print_stats(self, rows: list[tuple[str, str]], *, title: str = "Summary") -> None:
        self.print_plan(rows, title=title)

    def print_table(self, columns: list[str], rows: list[list[str]], *, title: str) -> None:
        """Print an arbitrary-width table (``print_plan`` is the 2-column case)."""
        if self.use_rich:
            table = Table(title=title, show_header=True, header_style="bold")
            for index, column in enumerate(columns):
                table.add_column(column, style="cyan" if index == 0 else "")
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
            return

        print(f"=== {title} ===")
        print("  " + " | ".join(columns))
        for row in rows:
            print("  " + " | ".join(row))

    @contextmanager
    def track(self, description: str, total: int | None = None) -> Iterator[Any]:
        if not self.use_rich:
            if total:
                from tqdm import tqdm

                bar = tqdm(total=total, desc=description, unit="item")
                try:
                    yield bar
                finally:
                    bar.close()
            else:
                self.info(description)
                yield None
            return

        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
        ]
        if total is not None:
            columns.extend([TaskProgressColumn(), MofNCompleteColumn()])
        columns.append(TimeElapsedColumn())

        with Progress(*columns, console=self.console, transient=False) as progress:
            task_id = progress.add_task(description, total=total)
            yield _RichTask(progress, task_id)

    @contextmanager
    def tasks(self) -> Iterator["_TaskGroup"]:
        if not self.use_rich:
            yield _FallbackTaskGroup(self)
            return

        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]
        with Progress(*columns, console=self.console, transient=False) as progress:
            yield _RichTaskGroup(progress)

    def done(self, message: str) -> None:
        if self.use_rich:
            self.console.print(f"[green]{message}[/green]")
        else:
            print(message)


class _TaskGroup:
    def add(self, description: str, *, total: int | None = None) -> Any:
        raise NotImplementedError

    def advance(self, task: Any, amount: int = 1) -> None:
        raise NotImplementedError

    def set_total(self, task: Any, total: int) -> None:
        raise NotImplementedError


class _RichTaskGroup(_TaskGroup):
    def __init__(self, progress: Progress) -> None:
        self._progress = progress

    def add(self, description: str, *, total: int | None = None) -> TaskID:
        return self._progress.add_task(description, total=total)

    def advance(self, task: TaskID, amount: int = 1) -> None:
        self._progress.update(task, advance=amount)

    def set_total(self, task: TaskID, total: int) -> None:
        self._progress.update(task, total=total)


class _FallbackTaskGroup(_TaskGroup):
    def __init__(self, reporter: Stage2PrepReporter) -> None:
        self._reporter = reporter
        self._tasks: dict[str, Any] = {}

    def add(self, description: str, *, total: int | None = None) -> str:
        if total:
            self._tasks[description] = self._new_bar(description, total)
        else:
            self._reporter.info(description)
            self._tasks[description] = None
        return description

    def advance(self, task: str, amount: int = 1) -> None:
        bar = self._tasks.get(task)
        if bar is not None:
            bar.update(amount)

    def set_total(self, task: str, total: int) -> None:
        bar = self._tasks.get(task)
        if bar is None:
            # Totals discovered after the task was added still get a bar.
            self._tasks[task] = self._new_bar(task, total)
            return
        bar.total = total
        bar.refresh()

    @staticmethod
    def _new_bar(description: str, total: int) -> Any:
        from tqdm import tqdm

        return tqdm(total=total, desc=description, unit="item")


class _RichTask:
    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def update(self, advance: int = 1, **fields: Any) -> None:
        self._progress.update(self._task_id, advance=advance, **fields)

    def set_total(self, total: int) -> None:
        self._progress.update(self._task_id, total=total)
