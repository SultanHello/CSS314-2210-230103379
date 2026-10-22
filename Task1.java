import java.util.Locale;
import java.util.Random;

/** Reproducible serial baseline. All array operations run on the main thread. */
public class Task1 {
    private static final long STUDENT_ID = 230103379L;
    private static final int ELEMENT_COUNT = 20_000_000;

    public static void main(String[] args) {
        Locale.setDefault(Locale.ROOT);
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.printf("OS: %s %s%n", System.getProperty("os.name"),
                System.getProperty("os.version"));
        System.out.println("Platform architecture: " + System.getProperty("os.arch"));
        System.out.println("Logical cores (availableProcessors): "
                + Runtime.getRuntime().availableProcessors());
        System.out.println("Seed: " + STUDENT_ID);
        System.out.println("Elements: " + ELEMENT_COUNT);
        System.out.println("Data type: float64 (Java double, 64 bits)");
        System.out.println("Distribution: uniform [0, 1)");
        System.out.println("Execution: single thread (serial loops on main thread)");

        long started = System.nanoTime();
        Random random = new Random(STUDENT_ID);
        double[] values = new double[ELEMENT_COUNT];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble();
        }
        long generated = System.nanoTime();

        // Compensated summation limits floating-point accumulation error.
        double sum = 0.0;
        double correction = 0.0;
        for (double value : values) {
            double adjusted = value - correction;
            double next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
        double mean = sum / values.length;
        double squaredDeviations = 0.0;
        correction = 0.0;
        for (double value : values) {
            double deviation = value - mean;
            double adjusted = deviation * deviation - correction;
            double next = squaredDeviations + adjusted;
            correction = (next - squaredDeviations) - adjusted;
            squaredDeviations = next;
        }
        double standardDeviation = Math.sqrt(squaredDeviations / values.length);
        long finished = System.nanoTime();

        System.out.printf("Sum: %.17g%n", sum);
        System.out.printf("Population standard deviation (ddof=0): %.17g%n", standardDeviation);
        System.out.printf("Serial computation wall-clock time (ms): %.3f%n",
                (finished - generated) / 1_000_000.0);
        System.out.printf("Total wall-clock time, generation + computation (ms): %.3f%n",
                (finished - started) / 1_000_000.0);
    }
}
