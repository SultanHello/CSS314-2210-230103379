import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/** Run: javac -d /tmp/task5-classes task5.java && java -cp /tmp/task5-classes task5 */
public class task5 {
    private static final int[] SIZES = {100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000};
    private static final int SAMPLES = 21;
    private static final double A = 1.75;
    private static final double B = 0.25;
    private static volatile double sink;

    private static void transform(double[] x, double[] y, int from, int to) {
        for (int i = from; i < to; i++) y[i] = A * x[i] + B;
    }

    private static final class Workload {
        final double[] x;
        final double[] serial;
        final double[] parallel;
        final List<Callable<Void>> tasks = new ArrayList<>();
        final ExecutorService pool;

        Workload(int n, int workers, ExecutorService pool) {
            this.pool = pool;
            x = new double[n];
            serial = new double[n];
            parallel = new double[n];
            for (int i = 0; i < n; i++) x[i] = (i % 1009) * 0.125;
            Arrays.fill(serial, Double.NaN);
            Arrays.fill(parallel, Double.NaN);
            for (int worker = 0; worker < workers; worker++) {
                int from = (int) ((long) worker * n / workers);
                int to = (int) ((long) (worker + 1) * n / workers);
                tasks.add(() -> { transform(x, parallel, from, to); return null; });
            }
        }

        void run(boolean multi) throws Exception {
            if (multi) {
                for (Future<Void> result : pool.invokeAll(tasks)) result.get();
            } else {
                transform(x, serial, 0, x.length);
            }
        }

        long time(boolean multi, int repetitions) throws Exception {
            long start = System.nanoTime();
            for (int i = 0; i < repetitions; i++) run(multi);
            long elapsed = System.nanoTime() - start;
            // Observe the output outside the timed region; arrays also escape to pool tasks.
            sink = (multi ? parallel : serial)[x.length / 2];
            return elapsed;
        }

        int calibrate(boolean multi) throws Exception {
            int repetitions = 1;
            while (time(multi, repetitions) < 3_000_000L && repetitions < 65_536) {
                repetitions *= 2;
            }
            return repetitions;
        }

        void validate() {
            for (int i = 0; i < x.length; i++) {
                double expected = A * x[i] + B;
                if (serial[i] != expected || parallel[i] != expected) {
                    throw new IllegalStateException("Incorrect result at index " + i);
                }
            }
        }
    }

    private static double median(double[] values) {
        Arrays.sort(values);
        return values[values.length / 2];
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        int workers = Runtime.getRuntime().availableProcessors();
        System.out.println("Student ID: 230103379");
        System.out.printf("OS: %s %s; architecture: %s%n", System.getProperty("os.name"),
                System.getProperty("os.version"), System.getProperty("os.arch"));
        System.out.printf("Java: %s; JVM: %s%n", System.getProperty("java.version"),
                System.getProperty("java.vm.name"));
        System.out.println("Available logical CPUs / pool threads: " + workers);
        System.out.println("Operation: double y[i] = 1.75 * x[i] + 0.25");
        System.out.println("Serial: calling thread; parallel: fixed pool, one contiguous block per worker.");
        System.out.println("Timing includes task submission, scheduling and waiting; excludes allocation,");
        System.out.println("pool startup, input initialization, verification and JVM warm-up.");
        System.out.println("Each result is the median of 21 batches, normalized per vector operation.");
        System.out.println("Batches calibrated independently to >=3 ms (cap 65536 operations).");
        System.out.println("Serial/parallel measurement order alternates to reduce ordering bias.");
        ExecutorService pool = Executors.newFixedThreadPool(workers);
        boolean[] faster = new boolean[SIZES.length];
        try {
            Workload warmup = new Workload(100_000, workers, pool);
            for (int i = 0; i < 500; i++) {
                warmup.run(false);
                warmup.run(true);
            }
            warmup.validate();
            System.out.printf("%n%10s | %12s | %12s | %8s | %9s | %11s%n",
                    "N", "Serial us", "Parallel us", "Speedup", "Winner", "Batch S/P");
            for (int index = 0; index < SIZES.length; index++) {
                Workload work = new Workload(SIZES[index], workers, pool);
                for (int i = 0; i < 30; i++) { work.run(false); work.run(true); }
                int serialBatch = work.calibrate(false);
                int parallelBatch = work.calibrate(true);
                double[] serialTimes = new double[SAMPLES];
                double[] parallelTimes = new double[SAMPLES];
                for (int sample = 0; sample < SAMPLES; sample++) {
                    if (sample % 2 == 0) {
                        serialTimes[sample] = work.time(false, serialBatch) / (double) serialBatch;
                        parallelTimes[sample] = work.time(true, parallelBatch) / (double) parallelBatch;
                    } else {
                        parallelTimes[sample] = work.time(true, parallelBatch) / (double) parallelBatch;
                        serialTimes[sample] = work.time(false, serialBatch) / (double) serialBatch;
                    }
                }
                work.validate();
                double serial = median(serialTimes);
                double parallel = median(parallelTimes);
                faster[index] = parallel < serial;
                System.out.printf("%10d | %12.3f | %12.3f | %8.3f | %9s | %5d/%-5d%n",
                        SIZES[index], serial / 1000, parallel / 1000, serial / parallel,
                        faster[index] ? "parallel" : "serial", serialBatch, parallelBatch);
            }
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(1, TimeUnit.MINUTES)) pool.shutdownNow();
        }
        System.out.println("\nValidation: both outputs match the expected value at every index for every N.");
        int transitions = 0;
        for (int i = 1; i < SIZES.length; i++) {
            if (!faster[i - 1] && faster[i]) {
                System.out.printf("Observed slower-to-faster crossover bracket: %d < N <= %d.%n",
                        SIZES[i - 1], SIZES[i]);
                transitions++;
            } else if (faster[i - 1] && !faster[i]) {
                System.out.printf("Observed reversal (parallel becomes slower): %d < N <= %d.%n",
                        SIZES[i - 1], SIZES[i]);
            }
        }
        if (transitions == 0) {
            System.out.println("No slower-to-faster crossover observed in the tested sizes.");
            if (faster[0]) System.out.println("Parallel is already faster at the smallest tested N.");
        }
        System.out.println("Brackets refer to adjacent tested sizes, not an exact integer threshold.");
        System.out.println("Scheduling, cache residency, memory bandwidth and CPU frequency can change results;");
        System.out.println("a monotonic crossover is not assumed. Re-run to assess measurement variability.");
    }
}
