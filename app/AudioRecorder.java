import javax.sound.sampled.*;
import java.io.*;

public class AudioRecorder {

    public static void main(String[] args) {
        final int durationSeconds = 5;
        final int sampleRate = 44100;
        final AudioFormat format = new AudioFormat(sampleRate, 16, 1, true, false);

        try {
            DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
            TargetDataLine line = (TargetDataLine) AudioSystem.getLine(info);
            line.open(format);
            line.start();

            System.out.println("Recording...");

            byte[] buffer = new byte[durationSeconds * sampleRate * 2]; // 2 bytes per sample
            int bytesRead = line.read(buffer, 0, buffer.length);

            System.out.println("Recording finished.");

            line.stop();
            line.close();

            // Save recorded audio to file
            AudioInputStream ais = new AudioInputStream(
                    new ByteArrayInputStream(buffer),
                    format,
                    bytesRead / format.getFrameSize());
            String filename = "recorded_sound.wav";
            File file = new File(filename);
            AudioSystem.write(ais, AudioFileFormat.Type.WAVE, file);

            System.out.println("Recording saved as: " + filename);

        } catch (LineUnavailableException | IOException e) {
            e.printStackTrace();
        }
    }
}
