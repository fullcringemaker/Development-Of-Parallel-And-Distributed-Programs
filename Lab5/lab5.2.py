import threading
import random
import sys

class ReadWriteLock:
    def __init__(self):
        self._readers = 0
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()

    def acquire_read(self):
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()

    def release_read(self):
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_lock.release()

    def acquire_write(self):
        self._write_lock.acquire()

    def release_write(self):
        self._write_lock.release()

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def contains(self, value):
        current = self.head
        while current:
            if current.value == value:
                return True
            current = current.next
        return False

    def append(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def to_list(self):
        result = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result

def worker(thread_id, count, linked_list, rw_lock):
    for _ in range(count):
        value = random.randint(0, 1000)
        rw_lock.acquire_read()
        exists = linked_list.contains(value)
        rw_lock.release_read()
        if exists:
            continue
        rw_lock.acquire_write()
        if not linked_list.contains(value):
            linked_list.append(value)
            print(f"[Thread {thread_id}] added value {value}")
        rw_lock.release_write()

def main():
    NUMBERS_PER_THREAD = int(sys.argv[1])
    THREADS_COUNT = int(sys.argv[2])
    linked_list = LinkedList()
    rw_lock = ReadWriteLock()
    threads = []
    print("Starting threads...")
    print(f"Threads count: {THREADS_COUNT}")
    print(f"Numbers per thread: {NUMBERS_PER_THREAD}\n")
    for i in range(THREADS_COUNT):
        t = threading.Thread(
            target=worker,
            args=(i, NUMBERS_PER_THREAD, linked_list, rw_lock)
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print("\nAll threads finished.\n")
    values = linked_list.to_list()
    unique_values = set(values)
    print(f"Total values in list: {len(values)}")
    print(f"Unique values:        {len(unique_values)}")
    if len(values) == len(unique_values):
        print("No duplicate values found.")
    else:
        print("Duplicates detected!")
    print("\nFinal list:")
    print(sorted(values))

if __name__ == "__main__":
    main()
