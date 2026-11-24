import bd.nsu.banglalpr.core.Img;
import bd.nsu.banglalpr.core.Net;
import bd.nsu.banglalpr.core.Pipeline;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

/** Robustness test: photos and crops of unusual sizes (tiny, huge, thin) must never crash the engine. */
public class Fuzz {
    public static void main(String[] a) throws Exception {
        String v = a[0], onnx = a[1], data = a[2];
        Net.Source src = name -> Files.readAllBytes(Paths.get(onnx, name));
        Pipeline p = new Pipeline(v, src);
        File[] photos = new File(data + "/e2e_test/images").listFiles();
        Random rnd = new Random(1);
        int fails = 0;
        for (int k = 0; k < 120; k++) {
            BufferedImage bi = ImageIO.read(photos[rnd.nextInt(photos.length)]);
            Img im = Img.fromArgb(bi.getRGB(0, 0, bi.getWidth(), bi.getHeight(), null, 0, bi.getWidth()), bi.getWidth(), bi.getHeight());
            double s = new double[]{0.05, 0.1, 0.2, 0.35, 0.5, 1, 1.7, 2.5}[k % 8];
            int nw = Math.max(1, (int) (im.w * s * (0.5 + rnd.nextDouble()))), nh = Math.max(1, (int) (im.h * s * (0.5 + rnd.nextDouble())));
            Img x = im.resizeLinear(nw, nh);
            try {
                p.process(x);
                int cw = 1 + rnd.nextInt(Math.max(1, Math.min(nw, 400))), ch = 1 + rnd.nextInt(Math.max(1, Math.min(nh, 200)));
                p.readCrop(x.crop(0, 0, cw, ch));
            } catch (Throwable e) {
                fails++;
                System.out.println("FAIL photo " + nw + "x" + nh + ": " + e);
                for (StackTraceElement st : e.getStackTrace()) { if (st.getClassName().contains("banglalpr")) System.out.println("    at " + st); }
                if (fails > 3) break;
            }
        }
        System.out.println(v + " fuzz done, failures: " + fails);
    }
}
