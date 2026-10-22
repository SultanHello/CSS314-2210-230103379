import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Run: javac -d /tmp/task6-classes task6.java && java -cp /tmp/task6-classes task6 */
public class task6 {
    private static final long STUDENT_ID = 230103379L;
    private static final int THREADS = 4;
    private static final int INCREMENTS = 2_000_000;
    private static final int EXPECTED = THREADS * INCREMENTS;
    private static final int ROUNDS = 7;

    private static final class Counter {
        int plain = 0; // Deliberately shared and unsynchronized for the race experiment.
        final AtomicInteger atomic = new AtomicInteger(0);
    }

    private record Result(int actual, long nanos) {}

    private static Result measure(boolean atomic, int increments) throws Exception {
        Counter counter = new Counter();
        CountDownLatch ready = new CountDownLatch(THREADS);
        CountDownLatch start = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread[] threads = new Thread[THREADS];
        for (int id = 0; id < THREADS; id++) {
            threads[id] = new Thread(() -> {
                ready.countDown();
                try {
                    start.await();
                    if (atomic) {
                        for (int i = 0; i < increments; i++) counter.atomic.incrementAndGet();
                    } else {
                        for (int i = 0; i < increments; i++) counter.plain++;
                    }
                } catch (Throwable error) {
                    failure.compareAndSet(null, error);
                }
            }, "counter-" + id);
            threads[id].start();
        }
        ready.await();
        long started = System.nanoTime();
        start.countDown();
        for (Thread thread : threads) thread.join();
        long elapsed = System.nanoTime() - started;
        if (failure.get() != null) throw new IllegalStateException(failure.get());
        // Joining establishes visibility of final writes to the reading main thread.
        int actual = atomic ? counter.atomic.get() : counter.plain;
        if (atomic && actual != THREADS * increments) {
            throw new IllegalStateException("Atomic counter lost updates: " + actual);
        }
        return new Result(actual, elapsed);
    }

    private static double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.printf("OS: %s %s; architecture: %s; Java: %s%n",
                System.getProperty("os.name"), System.getProperty("os.version"),
                System.getProperty("os.arch"), System.getProperty("java.version"));
        System.out.println("Available logical CPUs: " + Runtime.getRuntime().availableProcessors());
        System.out.printf("Threads: %d; attempted increments per thread: %d%n", THREADS, INCREMENTS);
        System.out.println("Expected final value: " + EXPECTED);
        System.out.println("Unsafe: ordinary shared int with counter++; no locks, volatile or atomics.");
        System.out.println("Correct: AtomicInteger.incrementAndGet(), an atomic read-modify-write.");
        System.out.println("Timing: release of ready workers through all joins; excludes thread creation.");
        System.out.println("Three warm-up pairs; seven measured pairs with alternating execution order.");
        for (int i = 0; i < 3; i++) {
            measure(false, 200_000);
            measure(true, 200_000);
        }
        double[] unsafeTimes = new double[ROUNDS];
        double[] atomicTimes = new double[ROUNDS];
        System.out.printf("%n%5s | %13s | %12s | %11s | %13s | %11s%n",
                "Round", "Unsafe actual", "Updates lost", "Unsafe ms", "Atomic actual", "Atomic ms");
        for (int round = 0; round < ROUNDS; round++) {
            Result unsafe;
            Result atomic;
            if (round % 2 == 0) {
                unsafe = measure(false, INCREMENTS);
                atomic = measure(true, INCREMENTS);
            } else {
                atomic = measure(true, INCREMENTS);
                unsafe = measure(false, INCREMENTS);
            }
            unsafeTimes[round] = unsafe.nanos() / 1e6;
            atomicTimes[round] = atomic.nanos() / 1e6;
            double lost = 100.0 * (EXPECTED - unsafe.actual()) / EXPECTED;
            System.out.printf("%5d | %13d | %11.3f%% | %11.3f | %13d | %11.3f%n",
                    round + 1, unsafe.actual(), lost, unsafeTimes[round],
                    atomic.actual(), atomicTimes[round]);
        }
        double unsafe = median(unsafeTimes);
        double atomic = median(atomicTimes);
        System.out.printf("%nMedian unsafe wall-clock time (ms): %.3f%n", unsafe);
        System.out.printf("Median atomic wall-clock time (ms): %.3f%n", atomic);
        System.out.printf("Synchronization time penalty, atomic - unsafe (ms): %+.3f%n", atomic - unsafe);
        System.out.printf("Atomic / unsafe time ratio: %.3fx%n", atomic / unsafe);
        System.out.printf("Relative time penalty: %+.3f%%%n", 100 * (atomic - unsafe) / unsafe);
        System.out.println("Validation: atomic counter equals 8000000 in every measured round.");
        System.out.println("counter++ is a read-modify-write sequence; competing threads overwrite updates.");
        System.out.println("The plain counter has a data race: JIT optimizations and scheduling affect losses");
        System.out.println("and timing; a run with no observed loss would not prove correctness.");
        System.out.println("The penalty compares correct atomic work with incorrect racy work, not an isolated");
        System.out.println("mutex cost. AtomicInteger uses atomic primitives rather than an explicit mutex;");
        System.out.println("contention and cache-coherence traffic contribute to its wall-clock cost.");
    }
}
