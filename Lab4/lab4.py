import threading
import time
import random
import csv
from collections import defaultdict

PHILOSOPHER_COUNT = 5
SIMULATION_TIME = 5.0
THINK_TIME_RANGE = (0.5, 2.0)
EAT_TIME_RANGE = (0.5, 1.5)

STATE_THINKING = "THINKING"
STATE_PICK_LEFT = "TAKING_LEFT_FORK"
STATE_PICK_RIGHT = "TAKING_RIGHT_FORK"
STATE_EATING = "EATING"
STATE_PUTTING_FORKS = "PUTTING_FORKS"

log_lock = threading.Lock()
events = []  

start_time = None
stop_event = threading.Event()

current_states = []  
csv_file = None
csv_writer = None

def now_from_start() -> float:
    return time.time() - start_time

def log_state(philosopher_id: int, state: str):
    global csv_writer, current_states
    t = time.time() - start_time
    if t > SIMULATION_TIME:
        return
    with log_lock:
        current_states[philosopher_id] = state
        events.append((t, philosopher_id, state))
        if csv_writer is not None:
            row = [f"{t:.3f}"] + list(current_states)
            csv_writer.writerow(row)

def philosopher_thread(phil_id: int, forks: list[threading.Lock], n: int):
    left_id = phil_id
    right_id = (phil_id + 1) % n
    while not stop_event.is_set():
        # 1) Philosopher is thinking
        log_state(phil_id, STATE_THINKING)
        think_time = random.uniform(*THINK_TIME_RANGE)
        time.sleep(think_time)
        if now_from_start() >= SIMULATION_TIME:
            break
        # 2) Philosopher tries to take forks: always lower-numbered fork first
        first_id = min(left_id, right_id)
        second_id = max(left_id, right_id)
        first_is_left = (first_id == left_id)
        second_is_left = (second_id == left_id)
        # Take first fork
        if first_is_left:
            log_state(phil_id, STATE_PICK_LEFT)
        else:
            log_state(phil_id, STATE_PICK_RIGHT)
        forks[first_id].acquire()
        if now_from_start() >= SIMULATION_TIME:
            forks[first_id].release()
            break
        # Take second fork
        if second_is_left:
            log_state(phil_id, STATE_PICK_LEFT)
        else:
            log_state(phil_id, STATE_PICK_RIGHT)
        forks[second_id].acquire()
        # 3) Eating
        log_state(phil_id, STATE_EATING)
        eat_time = random.uniform(*EAT_TIME_RANGE)
        time.sleep(eat_time)
        # 4) Put forks back
        log_state(phil_id, STATE_PUTTING_FORKS)
        forks[second_id].release()
        forks[first_id].release()
        # If time is over, exit loop after finishing this cycle
        if now_from_start() >= SIMULATION_TIME:
            break

def print_summary(n: int, total_duration: float):
    per_phil_events: dict[int, list[tuple[float, str]]] = {i: [] for i in range(n)}
    for t, pid, state in events:
        per_phil_events[pid].append((t, state))
    stats = {i: defaultdict(float) for i in range(n)}
    for pid in range(n):
        evs = sorted(per_phil_events[pid], key=lambda x: x[0])
        if not evs:
            continue
        for i in range(len(evs) - 1):
            t_cur, state_cur = evs[i]
            t_next, _ = evs[i + 1]
            stats[pid][state_cur] += max(0.0, t_next - t_cur)
        t_last, state_last = evs[-1]
        stats[pid][state_last] += max(0.0, total_duration - t_last)
    print(f"Total simulation duration: {total_duration:.3f} seconds\n")
    for pid in range(n):
        print(f"Philosopher {pid}:")
        st = stats[pid]
        total = sum(st.values())
        for state in [
            STATE_THINKING,
            STATE_PICK_LEFT,
            STATE_PICK_RIGHT,
            STATE_EATING,
            STATE_PUTTING_FORKS,
        ]:
            sec = st[state]
            perc = (sec / total * 100) if total > 0 else 0.0
            print(f"  {state:18s}: {sec:7.3f} s  ({perc:5.1f}%)")
        print()

def main():
    global start_time, csv_file, csv_writer, current_states
    n = PHILOSOPHER_COUNT
    forks = [threading.Lock() for _ in range(n)]
    current_states = [STATE_THINKING for _ in range(n)]
    csv_file = open("dining_log.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    header = ["time"] + [f"philosopher_{i}" for i in range(n)]
    csv_writer.writerow(header)
    threads = []
    start_time = time.time()
    for i in range(n):
        t = threading.Thread(target=philosopher_thread, args=(i, forks, n), daemon=True)
        t.start()
        threads.append(t)
    time.sleep(SIMULATION_TIME)
    stop_event.set()
    for t in threads:
        t.join(timeout=1.0)
    total_duration = min(now_from_start(), SIMULATION_TIME)
    if csv_file is not None:
        csv_file.close()
    print_summary(n, total_duration)

if __name__ == "__main__":
    main()
