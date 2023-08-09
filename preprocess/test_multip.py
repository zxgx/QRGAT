import pickle
from multiprocessing import Pool
import tracemalloc


def task(pid, idx):
    print("PID: %s start" % pid)
    min_key = 300
    max_key = -1
    for i in range(idx[0], idx[1]):
        max_key = max(data[str(i)][-1], max_key)
        min_key = min(data[str(i)][-1], min_key)
    print("PID: %s, interval: %s, max: %s, min: %s" % (pid, idx, max_key, min_key))
    print("PID: %s end" % pid)
    return [pid]


if __name__ == '__main__':
    tracemalloc.start()
    data = {str(i): [i] * 500000 for i in range(201)}
    bd = pickle.dumps(data)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage {current / 1e6}MB; Peak: {peak / 1e6}MB")
    tracemalloc.stop()

    num_p = 4
    pool = Pool(num_p)
    stride = len(data) // num_p

    tracemalloc.start()
    results = []
    for i in range(num_p):
        start = i * stride
        end = (i+1) * stride if i != num_p-1 else len(data)
        result = pool.apply_async(task, args=(i+1, [start, end]))
        results.append(result)
    pool.close()
    pool.join()
    results = [res.get()[0] for res in results]
    print(results)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage {current / 1e6}MB; Peak: {peak / 1e6}MB")
    tracemalloc.stop()
    print("Done")
