"""
Асинхронний сортувальник:
- Рекурсивно читає всі файли у вихідній папці
- Копіює в підпапки цільової папки за розширенням (no_ext — без розширення)
- Паралельність через asyncio + черга; копіювання — aioshutil.copy2
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import aioshutil


def setup_logging(verbose: bool, log_file: Optional[Path]) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


log = logging.getLogger(__name__)


def is_within(child: Path, parent: Path) -> bool:
    """Перевіряє, чи лежить child усередині parent."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ext_bucket(p: Path) -> str:
    """Назва підпапки за розширенням (без крапки). Порожнє -> 'no_ext'."""
    ext = p.suffix[1:].lower()
    return ext if ext else "no_ext"


async def copy_file(src: Path, out_root: Path) -> None:
    """
    Копіює один файл у підпапку за його розширенням.
    """
    bucket = ext_bucket(src)
    dst_dir = out_root / bucket
    dst_file = dst_dir / src.name

    # Захист від копіювання самого в себе
    if src.resolve() == dst_file.resolve():
        log.debug("Пропуск (джерело == призначення): %s", src)
        return

    try:
        await asyncio.to_thread(dst_dir.mkdir, parents=True, exist_ok=True)
        # Асинхронне копіювання
        await aioshutil.copy2(src, dst_file)
        log.info("Скопійовано: %s ➜ %s", src, dst_file)
    except PermissionError:
        log.error("Немає дозволу на копіювання: %s", src, exc_info=True)
    except Exception as e:
        log.exception("Помилка при копіюванні %s: %s", src, e)


async def _consumer(
    name: int, q: asyncio.Queue[Optional[Path]], out_root: Path
) -> None:
    """Воркер: бере з черги шлях і копіює файл."""
    while True:
        item = await q.get()
        if item is None:  # сигнал завершення
            q.task_done()
            log.debug("Воркер #%d завершує роботу", name)
            break
        try:
            await copy_file(item, out_root)
        finally:
            q.task_done()


async def read_folder(
    src_root: Path, out_root: Path, workers: int = 8, queue_size: int = 2000
) -> None:
    """
    Рекурсивно читає всі файли у src_root (через os.walk) і копіює їх у out_root,
    розкладаючи по підпапках за розширенням. Пропускає out_root і його піддерева,
    якщо вони лежать всередині src_root.
    """
    q: asyncio.Queue[Optional[Path]] = asyncio.Queue(maxsize=queue_size)

    # Стартуємо воркерів
    consumers = [
        asyncio.create_task(_consumer(i + 1, q, out_root)) for i in range(workers)
    ]

    files_count = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        current = Path(dirpath)

        # Не заходити в сам out_root
        if current.resolve() == out_root.resolve():
            dirnames.clear()
            continue

        # Викидаємо підпапки, що ведуть у out_root (коли out_root всередині src_root)
        dirnames[:] = [d for d in dirnames if not is_within(current / d, out_root)]

        for name in filenames:
            await q.put(current / name)
            files_count += 1
            if files_count % 1000 == 0:
                log.debug("Поставлено у чергу: %d файлів", files_count)

    log.info("Усього знайдено файлів: %d", files_count)

    # Дочікуємо обробку черги
    await q.join()

    # Надсилаємо сигнал завершення воркерам
    for _ in consumers:
        await q.put(None)
    await asyncio.gather(*consumers, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Асинхронне сортування файлів за розширенням."
    )
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        type=Path,
        help="Шлях до вихідної папки.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Шлях до цільової папки (буде створена, якщо відсутня).",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=min(32, (os.cpu_count() or 8)),
        help="Кількість паралельних воркерів копіювання (за замовчуванням: кількість CPU, максимум 32).",
    )
    parser.add_argument(
        "--queue-size",
        "-q",
        type=int,
        default=2000,
        help="Максимальний розмір черги (back-pressure).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Докладний лог (DEBUG).",
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=Path,
        default=None,
        help="Шлях до лог-файлу (додатково до виводу у консоль).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    setup_logging(args.verbose, args.log_file)

    src_root: Path = args.source
    out_root: Path = args.output
    workers: int = max(1, args.workers)
    queue_size: int = max(1, args.queue_size)

    if not src_root.exists() or not src_root.is_dir():
        log.error("Помилка: вихідна папка не існує або не є директорією: %s", src_root)
        return

    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.exception("Неможливо створити цільову папку %s: %s", out_root, e)
        return

    try:
        await read_folder(src_root, out_root, workers=workers, queue_size=queue_size)
    except Exception as e:
        log.exception("Критична помилка під час обробки: %s", e)
        return

    log.info("Готово. Файли відсортовано у «%s».", out_root.resolve())


if __name__ == "__main__":
    asyncio.run(main())
