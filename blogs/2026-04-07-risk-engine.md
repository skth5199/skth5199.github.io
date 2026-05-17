---
layout: post
title: "Building a Real-Time Risk Engine in Pure Python: Architecture, Bottlenecks, and Hard Lessons"
subtitle: "Mostly about the things that broke."
tags: [Quant Dev, Risk Engineering, Python]
---

## Why Pure Python?

Yes, a real-time risk engine in C++ or Rust would be faster. I know. You know. We can skip that part.

The team I inherited had three people. We were pricing a book of ~40,000 positions across rates, FX, credit, and equity derivatives. Our latency target for a full portfolio re-risk was under 30 seconds. The existing system was a tangle of Excel VBA and a legacy C# service that nobody could deploy without a specific engineer's laptop. That engineer had left.

So we picked Python. Not because it was the best language for the job, but because it was the only one that let us ship something in twelve weeks with three people. I want to be honest about that. There was no grand architectural vision. There was a deadline and a CRO who wanted numbers before the Asian open and a C# service we couldn't even compile.

That constraint shaped everything. The interesting engineering question was never "how do we make Python fast enough?" It was "where exactly is Python going to hurt us, and can we design around those specific spots?" Those are very different problems, and I think confusing them is how you end up rewriting everything in Rust eighteen months later with nothing to show for it.

## The Architecture That Survived Production

I want to describe what actually shipped. Not the version I'd build today, not the whiteboard sketch from sprint zero. The thing that ran.

We went through three failed prototypes before landing here. The first one tried to do everything in a single `asyncio` process. The second over-invested in Kafka infrastructure that nobody on the team knew how to operate. The third was actually pretty close but fell apart because we hadn't thought about how to get results out of worker processes without serialising everything through pickle. More on that later.

What shipped had four layers:

```text
Market Data Bus (ZeroMQ PUB/SUB)
          │
          ▼
┌─────────────────────┐
│   Ingestion Layer    │   ← Normalisation, deduplication, throttling
│   (asyncio)          │
└────────┬────────────┘
          │  in-process queue (collections.deque)
          ▼
┌─────────────────────┐
│   Pricing Layer      │   ← NumPy vectorised, per-asset-class workers
│   (ProcessPool)      │
└────────┬────────────┘
          │  shared memory (multiprocessing.shared_memory)
          ▼
┌─────────────────────┐
│  Aggregation Layer   │   ← Greeks roll-up, netting, limit checks
│  (single process)    │
└────────┬────────────┘
          │
          ▼
  Risk Store / API (Redis + FastAPI)
```

### Ingestion

Market data came from two sources: a proprietary feed via ZeroMQ and a fallback REST polling loop for less liquid instruments. The ingestion layer was a single `asyncio` process. Its job was to normalise ticks into an internal schema, deduplicate, and push them into an in-process queue. That's it.

The key decision here was **tick conflation**. Not every tick matters for risk. If EUR/USD moves from 1.0842 to 1.0843 and back to 1.0842 within 50ms, you don't need to reprice 6,000 EUR-denominated swaps three times. We buffered ticks per instrument for a configurable interval (200ms for G10 FX, 2s for illiquid credit names) and emitted only the latest snapshot per window.

```python
class TickConflator:
    def __init__(self, window_ms: float = 200.0):
        self._buffer: dict[str, Tick] = {}
        self._window = window_ms / 1000.0
        self._last_flush = time.monotonic()

    def ingest(self, tick: Tick) -> None:
        self._buffer[tick.instrument_id] = tick  # last-write-wins

    def flush(self) -> list[Tick]:
        now = time.monotonic()
        if now - self._last_flush < self._window:
            return []
        self._last_flush = now
        out = list(self._buffer.values())
        self._buffer.clear()
        return out
```

This is maybe fifteen lines of actual logic. It cut downstream pricing load by roughly 60%. I keep coming back to this whenever I'm tempted to reach for a clever optimisation: in real-time systems, figuring out what work you can skip entirely almost always beats making the same work go faster.

## The Pricing Layer

This is where most "real-time Python" articles wave their hands and say "just use NumPy." They're not wrong exactly, but they're skipping over all the parts that actually matter.

Our pricing layer ran as a pool of worker processes (one per asset class) managed via `multiprocessing`. Each worker received a batch of dirty positions (those affected by a market data change) and returned updated Greeks. The vectorisation strategy was different for each asset class, and that's where the real complexity lived.

### Rates

Repricing a vanilla interest rate swap is straightforward. Discount the projected cashflows, done. The expensive part is *building the curve*. A single tick on the 5Y USD swap rate means you need to re-bootstrap the entire USD discount curve before you can reprice anything that references it.

We pre-allocated NumPy arrays for each curve's node points and used in-place operations to avoid allocation overhead. Curve bootstrapping itself was a thin Cython wrapper around our own monotone convex interpolation. We ended up writing that because `scipy.interpolate` was allocating intermediate arrays on every call and the garbage collector was destroying our tail latency. I'll come back to the GC problem in section 6.

```python
# Curve rebuild: ~2ms for 20 nodes (Cython)
# Curve rebuild: ~18ms for 20 nodes (pure scipy)
# We rebuild 14 curves per tick. That's 224ms saved per cycle.

def reprice_irs_batch(
    positions: np.ndarray,    # (N, cols) structured array
    disc_curve: Curve,
    fwd_curve: Curve,
    as_of: float,
) -> np.ndarray:
    accruals = calc_accruals(positions, fwd_curve, as_of)  # (N, max_cf)
    dfs = disc_curve.discount(positions['cf_dates'])       # (N, max_cf)
    pvs = np.sum(accruals * dfs, axis=1)
    return pvs
```

### FX Options

FX vanillas were priced with Garman-Kohlhagen, which is just Black-Scholes with two rates. Fast enough in NumPy. The bottleneck was the vol surface interpolation. We stored surfaces as flat NumPy arrays (delta-space, expiry-space) and used `scipy.interpolate.RectBivariateSpline`.

Here's the thing that took us an embarrassingly long time to figure out. Our first version rebuilt the spline object on every pricing call. I ran `cProfile` on it one Friday afternoon after noticing the FX worker was consistently the slowest, and 73% of the time was in spline *construction*. Not in Black-Scholes evaluation. Not in the vol lookup. In building the interpolation object that hadn't changed since the last tick.

The fix was a one-line cache check. If the underlying vol matrix hasn't changed, reuse the spline. I almost didn't profile it because I was so sure the bottleneck was in the pricing math itself. I'd have wasted weeks chasing the wrong thing.

### Credit

CDX tranches needed a Gaussian copula. We wrote it in NumPy with Sobol quasi-random sequences for the Monte Carlo. One tranche, 50,000 paths, ~40ms on a single core. Fine.

We had 800 tranche positions. 800 times 40ms is 32 seconds. That blows our entire latency budget on one asset class.

The fix came from staring at the portfolio for a while and realising that the vast majority of those 800 tranches sat on the same handful of underlying indices. So we could run one correlated default simulation per index and then evaluate all the tranches that reference it against that shared simulation output. One Monte Carlo pass, many tranche evaluations. This brought the credit book down to about 3 seconds total.

I don't think any library would've given us that. It required knowing what was actually in the book. This is part of why I'm sceptical of fully generic "plug and play" risk engines, at least for anything beyond vanilla products. The optimisation that made the biggest difference was specific to our portfolio composition.

## Aggregation Under Pressure

Once the pricing workers emit updated Greeks, something has to roll them up. Conceptually simple: sum the deltas, group by currency, check limits. Operationally annoying for reasons I'll get into.

The aggregation layer ran as a single process. It consumed results from workers via shared memory and maintained the full risk state in a set of NumPy structured arrays indexed by position ID.

```python
# Risk state: pre-allocated, updated in-place
risk_state = np.zeros(MAX_POSITIONS, dtype=[
    ('pos_id',    'U32'),
    ('delta',     'f8'),
    ('gamma',     'f8'),
    ('vega',      'f8'),
    ('theta',     'f8'),
    ('pv',        'f8'),
    ('currency', 'U3'),
    ('desk',      'U16'),
    ('book',      'U16'),
    ('updated',  'f8'),
])
```

Someone on the team asked early on why we didn't parallelise the aggregation. Short answer: you can't, not cleanly. Two FX forwards can partially offset each other's delta. A desk-level gamma limit depends on the sum of every position on that desk. Split that across processes and you need a consistency window, and our risk team was (rightly) not interested in "probably correct" limit checks.

Anyway, performance here was never the problem. Aggregating 40,000 positions with NumPy vectorised group-by operations took ~8ms. The bottlenecks were always upstream in pricing.

## The GIL Thing

Every Python performance article mentions the GIL. Most get the implications wrong for this kind of system, or at least wrong for our specific setup.

The GIL didn't matter for the pricing layer. We used `multiprocessing`, not `threading`. Separate OS processes, separate interpreters.

Where it bit us was the ingestion layer. We initially ran the ZeroMQ subscriber, the conflation logic, and the queue producer as threads within a single process. The reasoning was that ZeroMQ manages its own I/O threads internally, so GIL contention should be minimal.

It wasn't. Under load (~5,000 ticks/second during London/NY overlap), we got unpredictable latency spikes. P50 was fine. P99 was terrible. 150ms stalls on what should've been a sub-millisecond operation. I spent two days convinced it was a ZeroMQ configuration issue before someone on the team stuck `sys.getswitchinterval()` logging in and we saw the thread scheduling chaos.

We ripped out the threading and moved to a fully `asyncio` design with `zmq.asyncio`. Single thread, cooperative scheduling. P99 dropped to under 5ms.

Going from multi-threaded to single-threaded made it faster. I realise that sounds wrong, but in CPython, reducing concurrency within a process often improves tail latency because you eliminate the GIL acquisition jitter entirely. It's one of those things I wouldn't have believed if I hadn't measured it myself.

## Memory

Python's memory overhead gets discussed a lot in the abstract. In practice it's less about the general problem and more about very specific things that catch you at the worst times.

### Object overhead

A Python `float` is 24 bytes. A C `double` is 8. With 40,000 positions and 30 risk fields each, the gap between "list of Python objects" and "NumPy structured array" is roughly 35MB versus 10MB. That sounds manageable, but each pricing worker (separate process, because `multiprocessing`) needs its own copy of the relevant position data. Four workers and you're over 200MB of position data before any intermediate arrays.

### Shared memory

Python 3.8's `multiprocessing.shared_memory` was a big deal for us. Before it, inter-process communication meant pickling arrays, which copies the data, runs the serialiser, and triggers GC on the receiving side. We switched to shared memory blocks backed by NumPy arrays:

```python
import multiprocessing.shared_memory as sm
import numpy as np

def create_shared_risk_array(name: str, size: int, dtype) -> tuple:
    nbytes = size * np.dtype(dtype).itemsize
    shm = sm.SharedMemory(name=name, create=True, size=nbytes)
    arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
    arr[:] = 0
    return shm, arr

def attach_shared_risk_array(name: str, size: int, dtype) -> tuple:
    shm = sm.SharedMemory(name=name, create=False)
    arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
    return shm, arr
```

IPC latency went from ~12ms (pickle round-trip for a 40K-row array) to basically zero.

The catch: shared memory segments don't get cleaned up if your process crashes. We found this out when a segfault in a pricing worker left orphaned blocks in `/dev/shm` eating 2GB of RAM. Nobody noticed for six hours, at which point the OOM killer took out the aggregation process during overnight batch. We now run a watchdog that inventories `/dev/shm` and reaps anything stale. It's ugly. I looked into whether there's a cleaner approach and I'm honestly not sure there is, at least not without wrapping everything in a higher-level abstraction that would undo the performance gains.

### Garbage collection

CPython's cyclic garbage collector can introduce stop-the-world pauses. In our pricing workers, we disabled it entirely with `gc.disable()` and relied on reference counting alone.

Yes, if you leak a circular reference, that memory is gone forever. We accepted this because the pricing workers were stateless and got restarted every 4 hours anyway (for reasons I'll get to). The result was real: it eliminated ~15ms GC pauses in our P99.5. But I'm still not fully comfortable with this decision. I wouldn't recommend it without the forced restart mechanism as a safety net, and even with it, there's a nagging feeling I'm missing something.

## Failure Modes

This is the section I actually want people to read. The architecture stuff is fine, but you can piece together variations of it from the docs and a few conference talks. These failures are what shaped our operational model, and I haven't seen most of them written up anywhere.

### The pickle bomb

Before shared memory, IPC used `multiprocessing.Queue`, which pickles everything. One of the other devs added a logging handler to a pricing result dataclass. Reasonable thing to do. But the handler held a reference to a socket, and pickle tried to serialise the socket, and the process just... hung. No error. No timeout. A blocked `put()` call with nothing to indicate anything was wrong.

The risk dashboard showed stale numbers for 40 minutes. Nobody noticed because the staleness indicator was comparing against a threshold we'd set too generously (we were still tuning it). Traders were looking at 40-minute-old Greeks and making decisions on them.

This one still bothers me. We added an explicit serialisation test in CI after that: every object that goes through a queue gets a pickle round-trip in the test suite. But the 40 minutes of stale data happened, and there's no CI test that can undo that.

### The leap second

Our day-count fraction calculations assumed `time.time()` was monotonic. It isn't. During a leap second adjustment (which, depending on your NTP config, can either be a backward time step or a smeared interval), the accrual logic briefly produced negative values for overnight positions. Greeks came out nonsensical. Limit breach alerts fired everywhere.

This was a genuinely hard bug to diagnose because by the time anyone looked at it, the clocks had stabilised and the accruals were fine again. We only figured it out from timestamp gaps in our tick logs. I think we got lucky that someone thought to check the logs at all rather than just assuming it was a blip.

Fix: `time.monotonic()` for all interval measurement, and business dates derived from an explicit schedule. Never from wall-clock arithmetic. Ever.

### The silent overflow

This one's my favourite in a grim sort of way. NumPy integer arrays overflow silently by default. We had a position with notional in JPY (100M JPY is a small trade). It overflowed an `int32` column. The notional wrapped to a large negative number. PV came out with the right magnitude, wrong sign. A desk showed a $900K profit that didn't exist.

```python
# This does NOT raise an error:
arr = np.array([2_147_483_647], dtype=np.int32)
arr[0] += 1
# arr[0] is now -2147483648
```

We caught it in the daily recon, but there was a window of about four hours where the desk was looking at wrong numbers and I can't be sure nobody acted on them. After that, every numeric column in our risk arrays became `float64`. Notionals, contract counts, everything. Even the ones that are "obviously integers." We also added a recon that compares our engine's notionals against the booking system every 60 seconds. The recon logic isn't complicated. What's complicated is explaining to yourself why you didn't have it from the start.

### The restart tax

We restarted pricing workers every 4 hours to contain memory creep (and to give us confidence that the `gc.disable()` choice wouldn't bite us). Each restart meant re-loading market data, rebuilding curves, warming the vol surface cache. About 45 seconds per worker. During that window, positions assigned to the restarting worker went stale.

We mitigated this with rolling restarts and temporary position reassignment. Never more than one worker down at a time. Sounds straightforward, but the orchestration code for it ended up being more lines than the actual pricing logic. I'm not proud of that ratio. I don't know how to improve it either.

## What I'd Change

This system shipped in 2023 and it's still running. Full re-risk cycle is ~22 seconds against a 30-second target. It's been stable enough that management hasn't funded a rewrite, which is either a compliment or a sign that nobody wants to touch it. Probably both.

### Polars for aggregation

When we started, Polars was too early. Today I'd use it for the aggregation layer without question. Lazy eval, multi-threaded group-by, zero-copy NumPy interop. It'd replace a lot of manual structured-array wrangling that nobody enjoys reading or debugging.

### Cython for the curve bootstrapper on day one

We spent eight weeks with a pure-Python bootstrapper that was "fine in dev, terrible in prod." The interpolation hot path is tight numerical loops where Python's per-operation overhead compounds multiplicatively. I'd eat the Cython build complexity upfront next time. Or maybe Mojo, though I haven't used it on anything real yet so I can't say that with confidence.

### Observability from week one, not month four

We bolted on structured logging and Prometheus metrics far too late. I know this sounds like an obvious thing to say, but I keep seeing teams (including mine) push it off because there are "more important" things to build first. If you don't have latency histograms on every inter-process boundary and depth gauges on every queue from day one, you will not know something is slow until it's slow enough to wake someone up at 3am. The observability isn't infrastructure you add later. It's the thing that lets you find problems before they become incidents.

### Free-threaded Python

Python 3.13's experimental no-GIL build is interesting to me. If it stabilises, it would let us collapse the multi-process pricing layer into threads, which eliminates the shared memory complexity, the pickle risks, the restart overhead. I'm watching it. I wouldn't bet production on it yet.

---

I don't have a clean takeaway for this. The system works. It has rough edges I've told you about, and probably some I haven't fully understood yet. If I had to pick one thing, it'd be: spend more time looking at what's actually in your portfolio before you optimise the pricing code. And profile before you guess. I'm still annoyed about the vol surface spline thing.

> I write about risk systems and Python when I have something worth saying. If any of this is relevant to what you're working on, reach out.
