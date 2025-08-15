"""
Підрахунок частоти слів з URL за допомогою MapReduce + багатопотоковість
"""

import argparse
import string
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import requests


def visualize_top_words(
    word_counts: Dict[str, int], source_url: str, top_n: int = 10
) -> None:
    """Побудувати горизонтальну діаграму топ-N слів."""
    if not word_counts:
        print("Немає даних для візуалізації.")
        return

    top = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels, values = zip(*top)
    labels = list(labels)[::-1]
    values = list(values)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.xlabel("Частота")
    plt.ylabel("Слова")
    plt.title(f"Топ {top_n} слів у тексті\n source: {source_url}")
    plt.tight_layout()
    plt.show()


def get_text(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        return response.text
    except requests.RequestException:
        return None


# Функція для видалення знаків пунктуації
def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def map_function(word: str) -> Tuple[str, int]:
    return word, 1


def shuffle_function(
    mapped_values: Sequence[Tuple[str, int]],
) -> Iterable[Tuple[str, List[int]]]:
    shuffled: Dict[str, List[int]] = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    return shuffled.items()


def reduce_function(key_values: Tuple[str, List[int]]) -> Tuple[str, int]:
    key, values = key_values
    return key, sum(values)


# Виконання MapReduce
def map_reduce(
    text: str,
    search_words: Iterable[str] | None = None,
    max_workers: int | None = None,
) -> Dict[str, int]:
    # 0) нормалізація: нижній регістр + прибрати пунктуацію
    text = remove_punctuation(text.lower())
    words = text.split()

    # Якщо задано список слів — фільтруємо (регістр уже зведено)
    if search_words:
        wanted = set(w.lower() for w in search_words)
        words = [word for word in words if word in wanted]

    # Крок 1: Паралельний Мапінг
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        mapped_values = list(executor.map(map_function, words))

    # Крок 2: Shuffle
    shuffled_values = shuffle_function(mapped_values)

    # Крок 3: Паралельна Редукція
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        reduced_values = list(executor.map(reduce_function, shuffled_values))

    return dict(reduced_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MapReduce аналіз частоти слів з URL",
    )
    parser.add_argument(
        "--url",
        "-u",
        required=True,
        help="URL-адреса з текстом",
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=10,
        help="Скільки топ-слів візуалізувати (default: 10)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Кількість потоків для map/reduce (default: авто)",
    )
    parser.add_argument(
        "--search",
        "-s",
        nargs="*",
        default=None,
        help="Опційно: рахувати лише ці слова (через пробіл)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    text = get_text(args.url)
    if not text:
        print("Помилка: Не вдалося отримати вхідний текст.")
        raise SystemExit(1)

    # Виконання MapReduce на вхідному тексті
    result = map_reduce(text, search_words=args.search, max_workers=args.workers)

    print("Унікальних слів:", len(result))
    if args.search:
        print(
            {w.lower(): result.get(w.lower(), 0) for w in args.search},
        )

    visualize_top_words(result, top_n=max(args.top, 1), source_url=args.url)
