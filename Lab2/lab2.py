from mpi4py import MPI
import sys, math, numpy as np

N = 12288
EPS = 0.00001
MAX_ITERS = 10000

if len(sys.argv) < 2:
    print("Usage: python lab2.py <variant: 1|2>")
    sys.exit(1)
variant = int(sys.argv[1])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rows_per_proc = [N // size + (1 if i < N % size else 0) for i in range(size)]
displs = [0]
for i in range(1, size):
    displs.append(displs[i-1] + rows_per_proc[i-1])
start = displs[rank]
end = start + rows_per_proc[rank]
RECVCOUNTS = np.array(rows_per_proc, dtype=np.int32)
RDISPLS = np.array(displs, dtype=np.int32)

def gather_full(local_vec):
    full = np.empty(N, dtype=np.float64)
    comm.Allgatherv(local_vec, [full, (RECVCOUNTS, RDISPLS)])
    return full

def dist_dot(u, v):
    loc = float(np.dot(u[start:end], v[start:end]))
    return comm.allreduce(loc, op=MPI.SUM)

def dist_norm2(u):
    return math.sqrt(dist_dot(u, u))

def matvec_naive(x):
    yloc = np.empty(end - start, dtype=np.float64)
    for li, i in enumerate(range(start, end)):
        s = 0.0
        for j in range(N):
            s += (2.0 if j == i else 1.0) * x[j]
        yloc[li] = s
    return gather_full(yloc)

x = np.zeros(N, dtype=np.float64)
if variant == 1:
    b = np.full(N, float(N + 1), dtype=np.float64)
elif variant == 2:
    u = np.array([math.sin(2.0 * math.pi * i / N) for i in range(N)], dtype=np.float64)
    b = matvec_naive(u)
else:
    if rank == 0:
        print("Variant must be 1 or 2")
    sys.exit(1)

comm.Barrier()
t0 = MPI.Wtime()

Ax = matvec_naive(x)
r = b - Ax
z = r.copy()
bnorm = dist_norm2(b)
if bnorm == 0.0:
    bnorm = 1.0
rel = dist_norm2(r) / bnorm
iters = 0

while rel > EPS and iters < MAX_ITERS:
    Az = matvec_naive(z)
    rr = dist_dot(r, r)
    denom = dist_dot(Az, z)
    if denom == 0.0:
        break
    alpha = rr / denom
    x = x + alpha * z
    r_new = r - alpha * Az
    rr_new = dist_dot(r_new, r_new)
    beta = rr_new / rr
    z = r_new + beta * z
    r = r_new
    iters += 1
    rel = math.sqrt(rr_new) / bnorm

comm.Barrier()
t1 = MPI.Wtime()

if rank == 0:
    print("variant = {}".format(variant))
    print("procs = {}".format(size))
    print("iterations = {}".format(iters))
    print("relative_residual = {:.10f}".format(rel))
    print("time_sec = {:.3f}".format(t1 - t0))
