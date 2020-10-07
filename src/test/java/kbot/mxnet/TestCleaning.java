package kbot.mxnet;

import java.util.Map;

public class TestCleaning {

	//	@Test
	public void testDispose() {

		// Unfortunately it is LOGGER based 'test'.

		MXNetwork network = new MXNetwork(2, new int[]{ 30, 2 },
				"Relu", "LinearRegression", "adam",
				Map.of("lr", 0.01f));

		System.out.println("Initialized");

		//noinspection UnusedAssignment
		network = null;

		// network not reachable anymore
		//do some other memory intensive work, check if you see the cleaning log message (might be disabled debug)
		int sum = 0;
		for (int i = 1; i <= 3000; i++) {
			int[] a = new int[100000];
			a[1] = 1;
			for (int s : a) sum += s;
			try {
				Thread.sleep(1);
			} catch (InterruptedException ignored) {
			}
		}
		System.out.println("Done");
	}
}
