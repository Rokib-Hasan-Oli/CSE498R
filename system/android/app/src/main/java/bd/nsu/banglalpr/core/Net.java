package bd.nsu.banglalpr.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.FloatBuffer;
import java.util.Collections;

/** One ONNX model run with ONNX Runtime (CPU; XNNPACK on the phone when available). */
public final class Net implements AutoCloseable {
    /** where the .onnx bytes come from: the app's assets on the phone, files on the PC */
    public interface Source { byte[] read(String name) throws Exception; }

    static final OrtEnvironment ENV = OrtEnvironment.getEnvironment();
    public static int threads = 4;
    public static boolean xnnpack = false;
    final OrtSession s;
    final String input;

    public Net(Source src, String name) throws Exception {
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setIntraOpNumThreads(threads);
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        if (xnnpack) {
            try { o.addXnnpack(Collections.singletonMap("intra_op_num_threads", String.valueOf(threads))); }
            catch (Throwable ignored) { /* not built with XNNPACK: plain CPU */ }
        }
        s = ENV.createSession(src.read(name), o);
        input = s.getInputNames().iterator().next();
    }

    /** run on a float tensor of the given shape; returns the first output flattened and its shape */
    public float[] run(float[] data, long[] shape, long[][] outShape) throws OrtException {
        try (OnnxTensor t = OnnxTensor.createTensor(ENV, FloatBuffer.wrap(data), shape);
             OrtSession.Result r = s.run(Collections.singletonMap(input, t))) {
            OnnxTensor o = (OnnxTensor) r.get(0);
            if (outShape != null) outShape[0] = o.getInfo().getShape();
            FloatBuffer fb = o.getFloatBuffer();
            float[] out = new float[fb.remaining()];
            fb.get(out);
            return out;
        }
    }

    @Override public void close() throws OrtException { s.close(); }
}
