## Q1: False sharing

In the naive variant, adjacent `hit_count[tid]` values normally occupy the same 64-byte cache line. When different OpenMP workers update those values, the cache-coherence protocol must repeatedly invalidate and transfer ownership of the line between cores. This MESI/MOESI coherence traffic serializes updates and consumes memory-bus bandwidth even though the counters are logically independent. The reduction variant gives each worker a private partial result and combines those results after the loop, so it greatly reduces cache-line invalidations.

## Q2: SMT saturation

Speedup should stop scaling linearly once the number of workers reaches the number of physical cores. SMT threads share execution resources such as arithmetic units, load/store units, cache capacity, and memory bandwidth. Additional logical workers can improve utilization when one worker stalls, but they cannot provide another full physical core, so contention and synchronization overhead reduce the incremental speedup.

## Q3: Empirical versus theoretical Amdahl speedup

The program derives `p` from the measured two-thread speedup using:

`p = 2 * (1 - 1 / S_emp(2))`

The theoretical curve assumes zero thread-management overhead, perfect load balance, and unlimited memory bandwidth. The empirical curve diverges as worker count increases because of cache-coherence and memory-bandwidth saturation, plus OpenMP fork/join, scheduling, reduction, and barrier overhead. Record the exact `p` and the observed curves from `results.csv`.

## Q4: Scheduling trade-off

Static scheduling has low scheduling overhead but can leave workers waiting if iteration cost is uneven. Dynamic scheduling balances the tail better, but small chunks such as `dynamic(100)` require more atomic work-queue operations. Large chunks such as `dynamic(10000)` reduce queue contention but can worsen load imbalance. Compare the `scheduling` rows in `results.csv`; the slower dynamic configuration on this machine is the point where queue overhead outweighs its balancing benefit.

