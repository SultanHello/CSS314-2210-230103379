import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/** Adjacent counters versus local accumulation and cache-line-separated counters. */
public class Task3 {
    private static final long STUDENT_ID = 230103379L;
    private static final long ITERATIONS = 50_000_000L;
    private static final VarHandle ELEMENT = MethodHandles.arrayElementVarHandle(long[].class);

    private static int hardwareValue(String key) throws Exception {
        Process process = new ProcessBuilder("/usr/sbin/sysctl", "-n", key).start();
        String value = new String(process.getInputStream().readAllBytes()).trim();
        if (process.waitFor() != 0) throw new IllegalStateException("Cannot detect " + key);
        return Integer.parseInt(value);
    }

    private static double measure(String mode, int workers, int stride, long iterations)
            throws Exception {
        long[] shared_counts = new long[workers * stride];
        CountDownLatch ready = new CountDownLatch(workers);
        CountDownLatch start = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread[] threads = new Thread[workers];
        for (int i = 0; i < workers; i++) {
            final int index = i * stride;
            threads[i] = new Thread(() -> {
                ready.countDown();
                try {
                    start.await();
                    if (mode.equals("Local")) {
                        long local = 0;
                        for (long j = 0; j < iterations; j++) local++;
                        shared_counts[index] = local;
                    } else {
                        // Every iteration must reach shared memory; ordinary array accesses
                        // could otherwise be collapsed into a single final store by the JIT.
                        for (long j = 0; j < iterations; j++) {
                            long value = (long) ELEMENT.getVolatile(shared_counts, index);
                            ELEMENT.setVolatile(shared_counts, index, value + 1);
                        }
                    }
                } catch (Throwable error) {
                    failure.compareAndSet(null, error);
                }
            }, "counter-" + i);
            threads[i].start();
        }
        ready.await();
        long started = System.nanoTime();
        start.countDown();
        for (Thread thread : threads) thread.join();
        long elapsed = System.nanoTime() - started;
        if (failure.get() != null) throw new IllegalStateException(failure.get());
        for (int i = 0; i < workers; i++) {
            if (shared_counts[i * stride] != iterations) {
                throw new IllegalStateException("Incorrect count for worker " + i);
            }
        }
        return elapsed / 1_000_000.0;
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        boolean mac = System.getProperty("os.name").equals("Mac OS X");
        int workers = args.length > 0 ? Integer.parseInt(args[0])
                : mac ? hardwareValue("hw.physicalcpu") : 0;
        int cacheLineBytes = args.length > 1 ? Integer.parseInt(args[1])
                : mac ? hardwareValue("hw.cachelinesize") : 0;
        if (workers < 1 || cacheLineBytes < 8) {
            throw new IllegalArgumentException("Supply physical cores and cache line bytes: Task3 <cores> <bytes>");
        }
        // Use the actual cache line size: this Mac has 128-byte lines, not 64.
        int stride = (cacheLineBytes + Long.BYTES - 1) / Long.BYTES;
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("Physical cores / worker threads: " + workers);
        System.out.println("Logical cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Cache line size (bytes): " + cacheLineBytes);
        System.out.println("Increments per thread: " + ITERATIONS);
        System.out.println("Unpadded shared_counts length: " + workers);
        System.out.println("Padded counter spacing (bytes): " + stride * Long.BYTES);
        System.out.println("Shared and padded: identical volatile array reads/writes via VarHandle.");
        System.out.println("Local: private counter with one final array write; JIT may simplify the loop.");
        System.out.println("Timing includes release and joins; excludes thread creation and warm-up.");
        for (String mode : new String[]{"Unpadded Shared", "Local", "Padded"}) {
            measure(mode, workers, mode.equals("Padded") ? stride : 1, 100_000L);
        }
        double shared = measure("Unpadded Shared", workers, 1, ITERATIONS);
        double local = measure("Local", workers, 1, ITERATIONS);
        double padded = measure("Padded", workers, stride, ITERATIONS);
        System.out.printf("%nUnpadded Shared wall-clock time (ms): %.3f%n", shared);
        System.out.printf("Local wall-clock time (ms): %.3f%n", local);
        System.out.printf("Padded wall-clock time (ms): %.3f%n", padded);
        System.out.printf("Execution ratio Unpadded / Local: %.3fx%n", shared / local);
        System.out.printf("Execution ratio Unpadded / Padded: %.3fx%n", shared / padded);
        System.out.println("Validation: every counter equals 50000000 in all three runs.");
        System.out.println("Ratios are observations from one run; scheduling and CPU frequency affect results.");
        System.out.println("Unpadded/Local also includes volatile-access and JIT effects; padded comparison better isolates false sharing.");
    }
}
