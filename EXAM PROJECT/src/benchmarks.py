import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from csd2_GA import csd2_GA
from csd_GA import csd_GA
from real_GA import real_GA


def lowpass_easy(w):
    if w <= 1.1:
        return 1.0
    elif w >= 1.9:
        return 0.0
    else:
        return None


def lowpass_medium(w):
    if w <= 1.3:
        return 1.0
    elif w >= 1.7:
        return 0.0
    else:
        return None


def lowpass_sharp(w):
    if w <= 1.4:
        return 1.0
    elif w >= 1.6:
        return 0.0
    else:
        return None


def lowpass_narrow(w):
    if w <= 0.7:
        return 1.0
    elif w >= 1.0:
        return 0.0
    else:
        return None


def lowpass_wide(w):
    if w <= 1.8:
        return 1.0
    elif w >= 2.2:
        return 0.0
    else:
        return None


TARGET_MAP = {
    "lowpass_easy": lowpass_easy,
    "lowpass_medium": lowpass_medium,
    "lowpass_sharp": lowpass_sharp,
    "lowpass_narrow": lowpass_narrow,
    "lowpass_wide": lowpass_wide,
}


POPULATION = 100
GENERATIONS = 300
ORDERS = [9, 17, 25, 33]
WORDLENGTHS = [8, 10, 12, 14, 16]
N_DIGITS = 3
N_MAX = 100
TARGETS = [
    lowpass_easy,
    lowpass_medium,
    lowpass_sharp,
    lowpass_wide,
    lowpass_narrow,
]
WORN = 4096
ELITISMS = [True]
EXPERIMENTS = 30

RESULTS_FILE = Path("results_dense_advanced.json")
MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)


# --------------------------------------------------
# JSON helpers
# --------------------------------------------------
def load_results(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results_atomic(path: Path, data):
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def ensure_path(root: dict, *keys):
    node = root
    for key in keys:
        if key not in node:
            node[key] = {}
        node = node[key]
    return node


# --------------------------------------------------
# History utilities
# --------------------------------------------------
def best_so_far_fitness(history):
    out = []
    current_best = float("-inf")
    for ind in history:
        if ind.fitness > current_best:
            current_best = float(ind.fitness)
        out.append(current_best)
    return out


def best_pop_fitness(history):
    return [float(ind.fitness) for ind in history]


def real_coefficients(best_individual):
    return [float(x) for x in best_individual.genome]


def csd_coefficients(best_individual):
    return [float(x) for x in best_individual.coeffs]


# --------------------------------------------------
# Progress helpers
# --------------------------------------------------
def count_total_runs():
    n_real = len(TARGETS) * len(ORDERS) * len(ELITISMS) * EXPERIMENTS
    n_csd = len(TARGETS) * len(ORDERS) * len(ELITISMS) * len(WORDLENGTHS) * EXPERIMENTS
    return n_real + n_csd + n_csd  # real + basic_csd + enhanced_csd


TOTAL_RUNS = count_total_runs()


def log_submit(idx, kind, target_name, order, elitism, rep, wordlength=None):
    msg = (
        f"[{idx}/{TOTAL_RUNS}] "
        f"SUBMIT {kind} | "
        f"target={target_name} | "
        f"order={order} | "
        f"elitism={elitism}"
    )
    if wordlength is not None:
        msg += f" | wordlength={wordlength}"
    msg += f" | rep={rep}/{EXPERIMENTS}"
    print(msg, flush=True)


def log_skip(idx, kind, target_name, order, elitism, rep, wordlength=None):
    msg = (
        f"[{idx}/{TOTAL_RUNS}] "
        f"SKIP   {kind} | "
        f"target={target_name} | "
        f"order={order} | "
        f"elitism={elitism}"
    )
    if wordlength is not None:
        msg += f" | wordlength={wordlength}"
    msg += f" | rep={rep}/{EXPERIMENTS} already present"
    print(msg, flush=True)


def log_done(done_idx, total_submitted, kind, best_fitness, target_name, order, elitism, rep, wordlength=None):
    msg = (
        f"[done {done_idx}/{total_submitted}] "
        f"DONE {kind} | "
        f"target={target_name} | "
        f"order={order} | "
        f"elitism={elitism}"
    )
    if wordlength is not None:
        msg += f" | wordlength={wordlength}"
    msg += f" | rep={rep}/{EXPERIMENTS} | best_fitness={best_fitness:.8f}"
    print(msg, flush=True)


def log_error(done_idx, total_submitted, kind, exc, target_name, order, elitism, rep, wordlength=None):
    msg = (
        f"[done {done_idx}/{total_submitted}] "
        f"ERROR {kind} | "
        f"target={target_name} | "
        f"order={order} | "
        f"elitism={elitism}"
    )
    if wordlength is not None:
        msg += f" | wordlength={wordlength}"
    msg += f" | rep={rep}/{EXPERIMENTS} | {type(exc).__name__}: {exc}"
    print(msg, flush=True)


# --------------------------------------------------
# Task/result helpers
# --------------------------------------------------
def make_rep_key(rep_idx):
    return f"rep{rep_idx}"


def result_exists(results, kind, target_key, order, elitism, rep_idx, wordlength=None):
    order_key = f"order_{order}"
    elitism_key = f"elitism_{elitism}"
    rep_key = make_rep_key(rep_idx)

    if kind == "real_ga":
        branch = (
            results.get(target_key, {})
                   .get(order_key, {})
                   .get(elitism_key, {})
                   .get("real_ga", {})
        )
        return rep_key in branch

    if kind == "basic_csd_ga":
        wl_key = f"wordlength_{wordlength}"
        branch = (
            results.get(target_key, {})
                   .get(order_key, {})
                   .get(elitism_key, {})
                   .get("basic_csd_ga", {})
                   .get(wl_key, {})
        )
        return rep_key in branch

    if kind == "enhanced_csd_ga":
        wl_key = f"wordlength_{wordlength}"
        branch = (
            results.get(target_key, {})
                   .get(order_key, {})
                   .get(elitism_key, {})
                   .get("enhanced_csd_ga", {})
                   .get(wl_key, {})
        )
        return rep_key in branch

    raise ValueError(f"Unknown kind: {kind}")


def store_result(results, payload):
    kind = payload["kind"]
    target_key = payload["target_key"]
    order_key = f"order_{payload['order']}"
    elitism_key = f"elitism_{payload['elitism']}"
    rep_key = make_rep_key(payload["rep_idx"])

    if kind == "real_ga":
        branch = ensure_path(results, target_key, order_key, elitism_key, "real_ga")
    elif kind == "basic_csd_ga":
        wl_key = f"wordlength_{payload['wordlength']}"
        branch = ensure_path(results, target_key, order_key, elitism_key, "basic_csd_ga", wl_key)
    elif kind == "enhanced_csd_ga":
        wl_key = f"wordlength_{payload['wordlength']}"
        branch = ensure_path(results, target_key, order_key, elitism_key, "enhanced_csd_ga", wl_key)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    branch[rep_key] = payload["result"]


# --------------------------------------------------
# Worker
# --------------------------------------------------
def run_task(task):
    """
    Executed inside a worker process.
    Returns only JSON-serializable data.
    """
    kind = task["kind"]
    target_key = task["target_key"]
    target = TARGET_MAP[target_key]
    order = task["order"]
    elitism = task["elitism"]
    rep_idx = task["rep_idx"]
    wordlength = task.get("wordlength")

    # Optional reproducible per-task seed
    seed = hash((kind, target_key, order, elitism, wordlength, rep_idx)) & 0xFFFFFFFF

    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    if kind == "real_ga":
        best, hist = real_GA(
            n_pop=POPULATION,
            generations=GENERATIONS,
            order=order,
            target=target,
            worN=WORN,
            elitism=elitism,
        )
        result = {
            "coefficients": real_coefficients(best),
            "best_so_far_fitness": best_so_far_fitness(hist),
            "best_pop_fitness": best_pop_fitness(hist),
        }

    elif kind == "basic_csd_ga":
        best, hist = csd_GA(
            n_pop=POPULATION,
            generations=GENERATIONS,
            order=order,
            target=target,
            worN=WORN,
            elitism=elitism,
            wordlength=wordlength,
            N_digits=N_DIGITS,
            N_max=N_MAX,
        )
        result = {
            "coefficients": csd_coefficients(best),
            "best_so_far_fitness": best_so_far_fitness(hist),
            "best_pop_fitness": best_pop_fitness(hist),
        }

    elif kind == "enhanced_csd_ga":
        best, hist = csd2_GA(
            n_pop=POPULATION,
            generations=GENERATIONS,
            order=order,
            target=target,
            worN=WORN,
            elitism=elitism,
            wordlength=wordlength,
            N_digits=N_DIGITS,
        )
        result = {
            "coefficients": csd_coefficients(best),
            "best_so_far_fitness": best_so_far_fitness(hist),
            "best_pop_fitness": best_pop_fitness(hist),
        }

    else:
        raise ValueError(f"Unknown kind: {kind}")

    return {
        "kind": kind,
        "target_key": target_key,
        "order": order,
        "elitism": elitism,
        "rep_idx": rep_idx,
        "wordlength": wordlength,
        "best_fitness": float(best.fitness),
        "result": result,
    }


# --------------------------------------------------
# Build task list
# --------------------------------------------------
def build_tasks(results):
    tasks = []
    logical_counter = 0

    for target in TARGETS:
        target_key = target.__name__

        for order in ORDERS:
            for elitism in ELITISMS:
                # Real GA
                for c in range(EXPERIMENTS):
                    rep_idx = c + 1
                    logical_counter += 1

                    if result_exists(results, "real_ga", target_key, order, elitism, rep_idx):
                        log_skip(logical_counter, "real_ga", target_key, order, elitism, rep_idx)
                    else:
                        log_submit(logical_counter, "real_ga", target_key, order, elitism, rep_idx)
                        tasks.append({
                            "kind": "real_ga",
                            "target_key": target_key,
                            "order": order,
                            "elitism": elitism,
                            "rep_idx": rep_idx,
                        })

                # CSD GAs
                for wordlength in WORDLENGTHS:
                    for c in range(EXPERIMENTS):
                        rep_idx = c + 1

                        logical_counter += 1
                        if result_exists(results, "basic_csd_ga", target_key, order, elitism, rep_idx, wordlength):
                            log_skip(
                                logical_counter,
                                "basic_csd_ga",
                                target_key,
                                order,
                                elitism,
                                rep_idx,
                                wordlength,
                            )
                        else:
                            log_submit(
                                logical_counter,
                                "basic_csd_ga",
                                target_key,
                                order,
                                elitism,
                                rep_idx,
                                wordlength,
                            )
                            tasks.append({
                                "kind": "basic_csd_ga",
                                "target_key": target_key,
                                "order": order,
                                "elitism": elitism,
                                "rep_idx": rep_idx,
                                "wordlength": wordlength,
                            })

                        logical_counter += 1
                        if result_exists(results, "enhanced_csd_ga", target_key, order, elitism, rep_idx, wordlength):
                            log_skip(
                                logical_counter,
                                "enhanced_csd_ga",
                                target_key,
                                order,
                                elitism,
                                rep_idx,
                                wordlength,
                            )
                        else:
                            log_submit(
                                logical_counter,
                                "enhanced_csd_ga",
                                target_key,
                                order,
                                elitism,
                                rep_idx,
                                wordlength,
                            )
                            tasks.append({
                                "kind": "enhanced_csd_ga",
                                "target_key": target_key,
                                "order": order,
                                "elitism": elitism,
                                "rep_idx": rep_idx,
                                "wordlength": wordlength,
                            })

    return tasks


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    results = load_results(RESULTS_FILE)

    print(f"Results file: {RESULTS_FILE}")
    print(f"Total logical runs: {TOTAL_RUNS}")
    print(f"Process workers: {MAX_WORKERS}")
    print("Scanning existing results and submitting missing tasks...\n", flush=True)

    tasks = build_tasks(results)

    print(f"\nMissing tasks to execute: {len(tasks)}", flush=True)

    if not tasks:
        print("Nothing to do. Benchmark already complete.", flush=True)
        return

    done_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(run_task, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            done_count += 1

            try:
                payload = future.result()

                # Only the main process mutates results and writes JSON
                store_result(results, payload)
                save_results_atomic(RESULTS_FILE, results)

                log_done(
                    done_count,
                    len(tasks),
                    payload["kind"],
                    payload["best_fitness"],
                    payload["target_key"],
                    payload["order"],
                    payload["elitism"],
                    payload["rep_idx"],
                    payload["wordlength"],
                )

            except Exception as exc:
                log_error(
                    done_count,
                    len(tasks),
                    task["kind"],
                    exc,
                    task["target_key"],
                    task["order"],
                    task["elitism"],
                    task["rep_idx"],
                    task.get("wordlength"),
                )

    print(f"\nBenchmark completed. Results saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()