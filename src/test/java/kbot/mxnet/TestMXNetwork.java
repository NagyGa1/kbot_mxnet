package kbot.mxnet;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Map;

import static java.lang.Math.abs;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestMXNetwork {

	@Test
	public void trainPredict1D() {
		final MXNetwork network = new MXNetwork(2, new int[]{ 30, 1 },
				"Relu", "LinearRegression", "adam",
				Map.of("lr", 0.01f));
		final float[][] features = new float[][]{
				{ 1.0f, 1.1f },
				{ 2.0f, 2.2f },
				{ 6.0f, 3.3f },
				{ 6.0f, 4.4f },
				{ 2.5f, 2.5f },
				{ 5.5f, 4.6f }
		};
		final float[] outputs = new float[]{ 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };

		float mae = 0.0f;
		for (int epoch = 1; epoch < 500; epoch++) {
			network.fit1D(features, outputs);
			final float[] p = network.predict1D_m(features);

			if (epoch % 50 == 0) {
				mae = 0.0f;
				for (int idx = 0; idx < p.length; idx++) {
					mae += abs(p[idx] - outputs[idx]);
				}
				mae /= outputs.length;
				System.out.printf("#%d MAE: %f%n", epoch, mae);
			}
		}
		assertEquals(0.023809558f, mae, 0.001f);
	}

	@Test
	public void trainPredict2D() {
		final MXNetwork network = new MXNetwork(2, new int[]{ 30, 2 },
				"Relu", "LinearRegression", "adam",
				Map.of("lr", 0.01f));
		final float[][] features = new float[][]{
				{ 1.0f, 1.1f },
				{ 2.0f, 2.2f },
				{ 6.0f, 3.3f },
				{ 6.0f, 4.4f },
				{ 2.5f, 2.5f },
				{ 5.5f, 4.6f }
		};
		final float[][] outputs = new float[][]{
				{ 0.0f, 0.5f },
				{ 0.0f, 0.4f },
				{ 1.0f, 1.0f },
				{ 1.0f, 1.0f },
				{ 0.0f, 0.6f },
				{ 1.0f, 1.0f }
		};

		float mae = 0.0f;
		for (int epoch = 1; epoch < 500; epoch++) {
			network.fit2D(features, outputs);
			final float[][] p = network.predict2D_m(features);

			if (epoch % 50 == 0) {
				mae = 0.0f;
				for (int idx = 0; idx < p.length; idx++) {
					mae += abs(p[idx][0] - outputs[idx][0]) + abs(p[idx][1] - outputs[idx][1]);
				}
				mae /= (outputs.length * 2);
				System.out.printf("#%d MAE: %f%n", epoch, mae);
			}
		}
		assertEquals(0.029863559f, mae, 0.006f);
	}
}
