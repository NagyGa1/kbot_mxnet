package kbot.mxnet;

/**
 * See ndarray.h enum DeviceType in mxnet.
 */
public enum MXContext {
	CPU(1), GPU(2), CPU_PINNED(3);

	public int ndarrayDeviceTypeId;

	MXContext(int ndarrayDeviceTypeId) {
		this.ndarrayDeviceTypeId = ndarrayDeviceTypeId;
	}
}
