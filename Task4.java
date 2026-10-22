import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;

/** Static contiguous blocks versus dynamic scheduling of the specified workload. */
public class Task4 {
    private static final long STUDENT_ID = 230103379L;
    private static final int ITEMS = 1_000;
    private static final int WORKERS = 4;

    private static void process(int item) throws InterruptedException {
        // 0.0001 seconds = 100,000 nanoseconds. Zero-cost items do not sleep.
        int nanos = 100_000 * (item % 20);
        if (nanos > 0) Thread.sleep(nanos / 1_000_000, nanos % 1_000_000);
    }

    private static void run(String strategy, int chunkSize) throws Exception {
        boolean dynamic = chunkSize > 0;
        AtomicInteger nextItem = new AtomicInteger();
        AtomicIntegerArray visits = new AtomicIntegerArray(ITEMS);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        CountDownLatch ready = new CountDownLatch(WORKERS);
        CountDownLatch start = new CountDownLatch(1);
        long[] finished = new long[WORKERS];
        int[] counts = new int[WORKERS];
        long[] requestedNanos = new long[WORKERS];
        Thread[] threads = new Thread[WORKERS];
        long started = System.nanoTime();
        for (int worker = 0; worker < WORKERS; worker++) {
            final int id = worker;
            threads[id] = new Thread(() -> {
                ready.countDown();
                try {
                    start.await();
                    int staticStart = id * ITEMS / WORKERS;
                    int staticEnd = (id + 1) * ITEMS / WORKERS;
                    while (true) {
                        int from = dynamic ? nextItem.getAndAdd(chunkSize) : staticStart;
                        if (from >= ITEMS) break;
                        int to = dynamic ? Math.min(from + chunkSize, ITEMS) : staticEnd;
                        for (int item = from; item < to; item++) {
                            process(item);
                            visits.incrementAndGet(item);
                            counts[id]++;
                            requestedNanos[id] += 100_000L * (item % 20);
                        }
                        if (!dynamic) break;
                    }
                } catch (Throwable error) {
                    failure.compareAndSet(null, error);
                } finally {
                    finished[id] = System.nanoTime();
                }
            }, "worker-" + id);
            threads[id].start();
        }
        ready.await();
        long released = System.nanoTime();
        start.countDown();
        for (Thread thread : threads) thread.join();
        long joined = System.nanoTime();
        if (failure.get() != null) throw new IllegalStateException(failure.get());
        for (int item = 0; item < ITEMS; item++) {
            if (visits.get(item) != 1) throw new IllegalStateException("Item " + item + " not processed exactly once");
        }
        long first = Arrays.stream(finished).min().orElseThrow();
        long last = Arrays.stream(finished).max().orElseThrow();
        System.out.println("\nStrategy: " + strategy);
        System.out.printf("Total execution time, creation through joins (ms): %.3f%n", (joined - started) / 1e6);
        System.out.printf("Execution time, release through last completion (ms): %.3f%n", (last - released) / 1e6);
        System.out.printf("First thread completion after release (ms): %.3f%n", (first - released) / 1e6);
        System.out.printf("Last thread completion after release (ms): %.3f%n", (last - released) / 1e6);
        System.out.printf("Min/max thread completion delta (ms): %.3f%n", (last - first) / 1e6);
        System.out.printf("%6s | %5s | %20s | %15s%n", "Worker", "Items", "Requested sleep (ms)", "Completion (ms)");
        for (int id = 0; id < WORKERS; id++) {
            System.out.printf("%6d | %5d | %20.3f | %15.3f%n", id, counts[id],
                    requestedNanos[id] / 1e6, (finished[id] - released) / 1e6);
        }
        System.out.println("Validation: all 1000 items processed exactly once.");
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("Items: 1000 (indices 0..999); worker threads: 4");
        System.out.println("Workload: sleep(0.0001 * (i % 20)) seconds per item.");
        System.out.println("Dynamic scheduling: shared atomic next-item queue, chunks of 1 and 5.");
        System.out.println("OS sleep precision and scheduling affect observed times.");
        System.out.println("The specified periodic costs make static block loads nearly balanced:");
        System.out.println("requested sleep is 232.5, 242.5, 232.5, 242.5 ms; dynamic need not be faster.");
        run("A: Static contiguous blocks (250 items each)", 0);
        run("B: Dynamic scheduling (chunk size 1)", 1);
        run("B: Dynamic scheduling (chunk size 5)", 5);
    }
}
