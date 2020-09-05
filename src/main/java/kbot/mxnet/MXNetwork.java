package kbot.mxnet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.util.Map;

public class MXNetwork implements Closeable {

	private static final Logger LOG = LoggerFactory.getLogger(MXNetwork.class);

	static {
		System.loadLibrary("kbot_mxnet");
		nativeInit();
		final String kbot_native_version = kbot_native_version();
		final String kbot_native_compiled_version = "1.0.0";
		LOG.info("KBOT_NATIVE: " + kbot_native_version);
		LOG.info("MXNET: " + mxnet_version());
		if (!kbot_native_version.equals(kbot_native_compiled_version)) {
			final String msg = "KBOT_NATIVE compiled version is " + kbot_native_compiled_version +
					", found " + kbot_native_version;
			LOG.warn(msg);
		}
	}

	/**
	 * Backing c++ object.
	 */
	public long net;

	public MXNetwork(final int inputs, final int[] hiddenLayers,
					 final String activationType,
					 final String outputType,
					 final String optimizerType,
					 final Map<String, Float> optimizerParams
	) {
		final String[] optimizerParamNames = new String[optimizerParams.size()];
		final float[] optimizerParamValues = new float[optimizerParams.size()];
		int idx = 0;
		for (Map.Entry<String, Float> e : optimizerParams.entrySet()) {
			optimizerParamNames[idx] = e.getKey();
			optimizerParamValues[idx] = e.getValue();
			idx++;
		}

		net = constructNet(inputs, hiddenLayers, activationType, outputType, optimizerType,
				optimizerParamNames, optimizerParamValues,
				MXContext.CPU.ndarrayDeviceTypeId, 0);
	}

	public MXNetwork(String fileNamePrefix) {
		this.net = constructNetLoad(fileNamePrefix, MXContext.CPU.ndarrayDeviceTypeId, 0);
	}

	public MXNetwork(MXNetwork other) {
		this.net = other.cloneNet();
	}

	@Override
	public void close() {
		if (net != 0) {
			disposeNet(); // implies net = 0
			assert net == 0;
		}
	}

	@Override
	protected void finalize() {
		if (net != 0) throw new RuntimeException("Memory leak, use close before releasing object.");
	}

	public MXNetwork copy() {
		return new MXNetwork(this);
	}

	static private native void nativeInit();

	static private native String kbot_native_version();

	static private native String mxnet_version();

	/**
	 * @param inputs               number of input features (columns of input array)
	 * @param hiddenLayers         hidden layer dimansions
	 * @param activationType       Relu, Sigmoid, Softrelu, Softsign, Tanh
	 * @param outputType           Softmax, LinearRegression, MAERegression.
	 * @param optimizerType        sgd, rmsprop, adam, adagrad, adagrad, signum, https://mxnet.apache.org/versions/1.6/api/python/docs/api/optimizer/index.html
	 * @param optimizerParamNames  names of optimizer parameters, same order as [optimizerParamValues]
	 * @param optimizerParamValues values of optimizer parameters
	 * @param contextDeviceType    See [MXContext]
	 * @param contextDeviceId      See [MXContext]
	 * @return long pointer to C++ heap MXNetwork object
	 */
	private native long constructNet(final int inputs,
									 final int[] hiddenLayers,
									 final String activationType,
									 final String outputType,
									 final String optimizerType,
									 final String[] optimizerParamNames,
									 final float[] optimizerParamValues,
									 final int contextDeviceType,
									 final int contextDeviceId
	);

	private native long constructNetLoad(final String fileName,
										 final int contextDeviceType,
										 final int contextDeviceId
	);

	private native void disposeNet();

	private native long cloneNet();

	public native void save(final String fileName);

	public native float[] predict1D_m(final float[][] inputs);

	public native float predict1D(final float[] input);

	public native void fit1D(final float[][] intputs, final float[] outputs);

	public native void fit1D_mb(final float[][] intputs, final float[] outputs, final int minibatchSize);

	public native float[][] predict2D_m(final float[][] inputs);

	public native float[] predict2D(final float[] input);

	public native void fit2D(final float[][] intputs, final float[][] outputs);

	public native void fit2D_mb(final float[][] intputs, final float[][] outputs, final int minibatchSize);
}
