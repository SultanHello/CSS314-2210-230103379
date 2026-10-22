import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Parallel summation of the same seeded data as Task1. */
public class Task2 {
    private static final long STUDENT_ID = 230103379L;
    private static final int ELEMENT_COUNT = 20_000_000;

    private static double sumRange(double[] values, int start, int end) {
        double sum = 0.0;
        double correction = 0.0;
        for (int i = start; i < end; i++) {
            double adjusted = values[i] - correction;
            double next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
        return sum;
    }

    private static double parallelSum(double[] values, int workers) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(workers);
        try {
            List<Callable<Double>> tasks = new ArrayList<>();
            for (int worker = 0; worker < workers; worker++) {
                int start = (int) ((long) values.length * worker / workers);
                int end = (int) ((long) values.length * (worker + 1) / workers);
                tasks.add(() -> sumRange(values, start, end));
            }
            double sum = 0.0;
            for (Future<Double> result : pool.invokeAll(tasks)) {
                sum += result.get();
            }
            return sum;
        } finally {
            pool.shutdown();
            // Include thread shutdown in the measured operation.
            pool.close();
        }
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        int cores = Runtime.getRuntime().availableProcessors();
        double[] values = new double[ELEMENT_COUNT];
        Random random = new Random(STUDENT_ID);
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble();
        }
        double reference = sumRange(values, 0, values.length);
        LinkedHashSet<Integer> workerCounts = new LinkedHashSet<>(
                List.of(1, 2, 4, cores, cores * 2));

        System.out.println("Student ID: " + STUDENT_ID);
        System.out.printf("OS: %s %s; architecture: %s%n", System.getProperty("os.name"),
                System.getProperty("os.version"), System.getProperty("os.arch"));
        System.out.println("Logical cores (availableProcessors): " + cores);
        System.out.println("Elements: " + ELEMENT_COUNT + "; type: float64 (Java double)");
        System.out.println("Seed: " + STUDENT_ID + "; distribution: uniform [0, 1)");
        System.out.printf("Reference serial sum: %.17g%n", reference);
        System.out.println("Timing: summation including pool creation, dispatch, reduction and shutdown.");
        System.out.println("Array generation and warm-up excluded; median of 5 measured runs.");
        System.out.println("Amdahl: 1 / (0.05 + 0.95 / workers), P = 0.95.");
        System.out.printf("%7s | %14s | %10s | %18s%n",
                "Cores", "Wall Time (ms)", "Speedup", "Amdahl Upper Bound");
        double baseline = 0.0;
        for (int workers : workerCounts) {
            for (int warmup = 0; warmup < 3; warmup++) {
                parallelSum(values, workers);
            }
            long[] times = new long[5];
            for (int run = 0; run < times.length; run++) {
                long started = System.nanoTime();
                double sum = parallelSum(values, workers);
                times[run] = System.nanoTime() - started;
                if (Math.abs(sum - reference) > Math.abs(reference) * 1e-12) {
                    throw new IllegalStateException("Incorrect sum for " + workers + " workers");
                }
            }
            java.util.Arrays.sort(times);
            double milliseconds = times[times.length / 2] / 1_000_000.0;
            if (workers == 1) baseline = milliseconds;
            System.out.printf("%7d | %14.3f | %10.3f | %18.3f%n", workers,
                    milliseconds, baseline / milliseconds, 1.0 / (0.05 + 0.95 / workers));
        }
        System.out.println("All measured sums match the serial reference within relative tolerance 1e-12.");
        System.out.println("Amdahl values assume a 5% serial fraction; actual hardware and overhead may differ.");
    }
}
