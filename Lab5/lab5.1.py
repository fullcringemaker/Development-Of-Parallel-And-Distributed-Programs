import sys
import time
import threading
import numpy as np

GRID_ROWS = 1500
GRID_COLS = 1500
SEED = 0

def step_single(src, dst):
    up = np.roll(src, 1, axis=0)
    down = np.roll(src, -1, axis=0)
    n = (
        np.roll(up, 1, axis=1) + up + np.roll(up, -1, axis=1) +
        np.roll(src, 1, axis=1) + np.roll(src, -1, axis=1) +
        np.roll(down, 1, axis=1) + down + np.roll(down, -1, axis=1)
    )
    alive = (src == 1)
    dst[:] = ((alive & ((n == 2) | (n == 3))) | ((~alive) & (n == 3))).astype(np.uint8)

def run_single(initial, steps):
    a = initial.copy()
    b = np.empty_like(a)
    t0 = time.perf_counter()
    for _ in range(steps):
        step_single(a, b)
        a[:] = b
    t1 = time.perf_counter()
    return a, (t1 - t0) / steps

def run_threads(initial, steps, n_threads):
    a = initial.copy()
    b = np.empty_like(a)
    starts = [i * GRID_ROWS // n_threads for i in range(n_threads)]
    ends = [(i + 1) * GRID_ROWS // n_threads for i in range(n_threads)]
    top_rows = np.empty((n_threads, GRID_COLS), dtype=np.uint8)
    bottom_rows = np.empty((n_threads, GRID_COLS), dtype=np.uint8)
    barrier_pub = threading.Barrier(n_threads) 
    barrier_done = threading.Barrier(n_threads) 
    barrier_copy = threading.Barrier(n_threads)

    def worker(tid):
        s = starts[tid]
        e = ends[tid]
        prev_tid = (tid - 1) % n_threads
        next_tid = (tid + 1) % n_threads
        block = a[s:e]
        prev_rows = np.empty_like(block)
        next_rows = np.empty_like(block)
        for _ in range(steps):
            top_rows[tid] = a[s]
            bottom_rows[tid] = a[e - 1]
            barrier_pub.wait()
            halo_up = bottom_rows[prev_tid]
            halo_down = top_rows[next_tid]
            prev_rows[0] = halo_up
            prev_rows[1:] = block[:-1]
            next_rows[-1] = halo_down
            next_rows[:-1] = block[1:]
            n = (
                np.roll(prev_rows, 1, axis=1) + prev_rows + np.roll(prev_rows, -1, axis=1) +
                np.roll(block, 1, axis=1) + np.roll(block, -1, axis=1) +
                np.roll(next_rows, 1, axis=1) + next_rows + np.roll(next_rows, -1, axis=1)
            )
            alive = (block == 1)
            b[s:e] = ((alive & ((n == 2) | (n == 3))) | ((~alive) & (n == 3))).astype(np.uint8)
            barrier_done.wait()
            a[s:e] = b[s:e]
            barrier_copy.wait()
            block = a[s:e]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    t1 = time.perf_counter()
    return a, (t1 - t0) / steps

def main():
    steps = int(sys.argv[1])
    n_threads = int(sys.argv[2])
    rng = np.random.default_rng(SEED)
    initial = (rng.random((GRID_ROWS, GRID_COLS)) < 0.5).astype(np.uint8)
    print(f"grid={GRID_ROWS}x{GRID_COLS} steps={steps} threads={n_threads} seed={SEED}")
    print(f"initial_alive={int(initial.sum())}")
    final_single, avg_single = run_single(initial, steps)
    final_threads, avg_threads = run_threads(initial, steps, n_threads)
    mismatch = int(np.count_nonzero(final_single != final_threads))
    alive_single = int(final_single.sum())
    alive_threads = int(final_threads.sum())
    speedup = (avg_single / avg_threads - 1.0) * 100.0
    print(f"single_avg_step_s={avg_single:.6f}")
    print(f"threads_avg_step_s={avg_threads:.6f}")
    print(f"speedupt={speedup:.2f}%")
    print(f"final_alive_single={alive_single}")
    print(f"final_alive_threads={alive_threads}")
    print(f"final_mismatch_cells={mismatch}")

if __name__ == "__main__":
    main()
