"""Time one training step of the flow, to separate "the model is expensive" from "the
GPU is busy or slow".

Three numbers, for the exact architecture and batch size a training run would use:

  gemm     a plain large matmul, to see what the card actually delivers right now. If
           this is far below the device's spec, the GPU is shared or throttled and no
           change to the model will help -- check nvidia-smi.
  eager    forward+backward of the flow as train_flow_*.py runs it today.
  compiled the same under torch.compile, which fuses the spline's hundreds of small
           elementwise kernels. This is where the headroom is: the arithmetic in a step
           is ~1 TFLOP, so anything above ~200 ms is overhead, not maths.

Run:  python bench_flow.py                        (the train_flow_complete.py defaults)
      python bench_flow.py --n_layers 8 --num_bins 6 --batch_size 1048576
"""
import argparse
import time

import torch

from models import NeuralSplineFlow


def timed(fn, n_iter, warmup=3, cuda=False):
    """Mean seconds per call, after warmup, synchronizing if on CUDA."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def bench_gemm(device, cuda, n=8192, n_iter=20):
    """TFLOP/s on a square fp32 matmul -- what this card is giving us today."""
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    dt = timed(lambda: a @ b, n_iter, cuda=cuda)
    return 2 * n ** 3 / dt / 1e12, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dim', type=int, default=6, help="Fit-space dimensions")
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--num_bins', type=int, default=8)
    parser.add_argument('--tail_bound', type=float, default=6.0)
    parser.add_argument('--batch_size', type=int, default=2 ** 19)
    parser.add_argument('--n_iter', type=int, default=10, help="Timed steps per variant")
    parser.add_argument('--device', type=str, default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True,
                        help="Allow TF32 matmul, as the trainers do")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True,
                        help="Also time the torch.compile'd step")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = torch.device(device).type == 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    if cuda:
        name = torch.cuda.get_device_name(device)
        free, total = torch.cuda.mem_get_info(device)
        print(f"{name} | {free / 2**30:.1f} of {total / 2**30:.1f} GiB free "
              f"(anything missing is another process on the card)")
    print(f"device={device} tf32={args.tf32} | dim={args.dim} n_layers={args.n_layers} "
          f"hidden={args.hidden} num_bins={args.num_bins} batch={args.batch_size:,}")

    if cuda:
        tflops, dt = bench_gemm(device, cuda)
        print(f"\n  gemm      8192^3 fp32: {dt * 1000:8.1f} ms  -> {tflops:6.1f} TFLOP/s")

    flow = NeuralSplineFlow(dim=args.dim, n_layers=args.n_layers, hidden=tuple(args.hidden),
                            num_bins=args.num_bins, tail_bound=args.tail_bound).to(device)
    x = torch.randn(args.batch_size, args.dim, device=device)
    w = torch.rand(args.batch_size, device=device)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)

    def step(log_prob):
        def run():
            loss = -(w * log_prob(x)).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return run

    # the arithmetic in one step, for reference: everything above this is overhead
    tr = args.dim - args.dim // 2
    per_row = 2 * (tr * args.hidden[0] + sum(a * b for a, b in zip(args.hidden, args.hidden[1:]))
                   + args.hidden[-1] * tr * (3 * args.num_bins + 1))
    tflop = 3 * per_row * args.n_layers * args.batch_size / 1e12

    variants = [("eager", flow.log_prob)]
    if args.compile:
        try:
            compiled = torch.compile(flow.log_prob, dynamic=False)
            print("\n  compiling (first call builds the graph, ~1-2 min) ...", flush=True)
            step(compiled)()                       # forces the build, outside the timing
            variants.append(("compiled", compiled))
        except Exception as exc:                   # no triton, unsupported backend, ...
            print(f"  torch.compile unavailable: {type(exc).__name__}: {exc}")

    print()
    for tag, log_prob in variants:
        dt = timed(step(log_prob), args.n_iter, cuda=cuda)
        print(f"  {tag:<9} fwd+bwd+step: {dt * 1000:8.1f} ms  -> "
              f"{args.batch_size / dt / 1e6:6.1f} M rows/s, "
              f"{tflop / dt:6.1f} TFLOP/s effective")
    print(f"\n  ({tflop:.2f} TFLOP of actual arithmetic per step)")


if __name__ == '__main__':
    main()
