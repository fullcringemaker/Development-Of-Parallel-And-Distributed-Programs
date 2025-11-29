import threading
import time
import random
from collections import defaultdict

PHILOSOPHER_COUNT = 5 
SIMULATION_TIME = 20.0 
THINK_TIME_RANGE = (0.5, 2.0) 
EAT_TIME_RANGE   = (0.5, 1.5) 

STATE_THINKING      = "ДУМАЕТ"
STATE_PICK_LEFT     = "БЕРЁТ ЛЕВУЮ ВИЛКУ"
STATE_PICK_RIGHT    = "БЕРЁТ ПРАВУЮ ВИЛКУ"
STATE_EATING        = "ЕСТ"
STATE_PUTTING_FORKS = "КЛАДЁТ ВИЛКИ"

log_lock = threading.Lock()
events = [] 

start_time = None
stop_event = threading.Event()
def now_from_start() -> float:
    return time.time() - start_time

def log_state(philosopher_id: int, state: str):
    t = now_from_start()
    with log_lock:
        events.append((t, philosopher_id, state))
    print(f"{t:8.3f}s | Философ {philosopher_id} -> {state}")

def philosopher_thread(phil_id: int, forks: list[threading.Lock], n: int):
    left_id = phil_id
    right_id = (phil_id + 1) % n
    while not stop_event.is_set():
        #1) Философ думает
        log_state(phil_id, STATE_THINKING)
        think_time = random.uniform(*THINK_TIME_RANGE)
        time.sleep(think_time)
        #2) Философ проголодался и пытается взять вилки. сначала вилку с меньшим номером, затем с большим.
        first_id = min(left_id, right_id)
        second_id = max(left_id, right_id)
        first_is_left = (first_id == left_id)
        second_is_left = (second_id == left_id)
        # Берём первую вилку
        if first_is_left:
            log_state(phil_id, STATE_PICK_LEFT)
        else:
            log_state(phil_id, STATE_PICK_RIGHT)
        forks[first_id].acquire()
        # Берём вторую вилку
        if second_is_left:
            log_state(phil_id, STATE_PICK_LEFT)
        else:
            log_state(phil_id, STATE_PICK_RIGHT)
        forks[second_id].acquire()
        #3) Ест
        log_state(phil_id, STATE_EATING)
        eat_time = random.uniform(*EAT_TIME_RANGE)
        time.sleep(eat_time)
        #4) Кладёт вилки
        log_state(phil_id, STATE_PUTTING_FORKS)
        forks[second_id].release()
        forks[first_id].release()

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
    print(f"Общая длительность симуляции: {total_duration:.3f} секунд\n")
    for pid in range(n):
        print(f"Философ {pid}:")
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
            print(f"  {state:18s}: {sec:7.3f} с  ({perc:5.1f}%)")
        print()

def main():
    global start_time
    n = PHILOSOPHER_COUNT
    forks = [threading.Lock() for _ in range(n)]
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
    total_duration = now_from_start()
    print_summary(n, total_duration)
if __name__ == "__main__":
    main()

