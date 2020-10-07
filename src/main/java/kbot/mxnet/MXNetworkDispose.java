package kbot.mxnet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class MXNetworkDispose implements Runnable {

	private static final Logger LOG = LoggerFactory.getLogger(MXNetworkDispose.class);
	private final long netToClean;

	public MXNetworkDispose(final long netToClean) {
		this.netToClean = netToClean;
	}

	@Override
	public void run() {
		if (LOG.isDebugEnabled()) LOG.debug("Cleaning: " + netToClean);
//		LOG.info("Cleaning: " + netToClean);
		MXNetwork.disposeNet(netToClean);
	}
}
