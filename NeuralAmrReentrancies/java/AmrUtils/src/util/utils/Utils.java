package util.utils;

import fig.basic.IOUtils;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 *
 * @author ikonstas
 */
public class Utils {

    public static String[] readLines(String path, int maxLines) {
        ArrayList<String> linesList = null;
        final BufferedReader in = IOUtils.openInEasy(path);
        if (in != null) {
            linesList = new ArrayList<>();
            String line = "";
            int i = 0;
            try {
                while (((line = in.readLine()) != null) && i < maxLines) {
                    linesList.add(line);
                    i++;
                }
                in.close();
            } catch (IOException ioe) {
                System.err.println(String.format("Error reading file %s", path));
            }
        }
        String[] out = new String[linesList.size()];
        return linesList.toArray(out);
    }

    public static String[] readLines(String path) {
        return readLines(path, Integer.MAX_VALUE);
    }

    public static String readFileAsString(String filePath) throws java.io.IOException {
        byte[] buffer = new byte[(int) new File(filePath).length()];
        BufferedInputStream f = null;
        try {
            f = new BufferedInputStream(new FileInputStream(filePath));
            f.read(buffer);
        } finally {
            if (f != null) {
                try {
                    f.close();
                } catch (IOException ignored) {
                }
            }
        }
        return new String(buffer);
    }

    public static String stripExtension(String name) {
        int index = name.lastIndexOf(".");
        return index == -1 ? name : name.substring(0, index);
    }

    public static boolean writeLines(String path, String[] lines) {
        PrintWriter out = IOUtils.openOutEasy(path);
        if (out != null) {
            for (String line : lines) {
                out.println(line);
            }
            out.close();
            return true;
        }
        return false;
    }

    public static boolean write(String path, String text) {
        PrintWriter out = IOUtils.openOutEasy(path);
        if (out != null) {
            out.println(text);
            out.close();
            return true;
        }
        return false;
    }

    public static String[] tokenize(String input) {
        return input.toLowerCase().split("\\s");
    }

    public static String[] removeStopWords(String[] tokens, Set<String> stopWords) {
        List<String> res = new ArrayList<>();
        for (String token : tokens) {
            if (!stopWords.contains(token)) {
                res.add(token);
            }
        }
        return res.toArray(new String[0]);
    }

    public static String[] tokenizeToUnigrams(String sentence, Set<String> stopWords) {
        return stopWords == null ? tokenize(sentence) : removeStopWords(tokenize(sentence), stopWords);
    }

    public static String[] tokenizeToBigrams(String sentence, Set<String> stopWords) {
        List<String> bigrams = new ArrayList<>();
        String[] unigrams = tokenize("<s> " + sentence.replaceAll(", ", "") + " </s>");
        for (int i = 1; i < unigrams.length; i++) {
            bigrams.add(unigrams[i - 1] + "_" + unigrams[i]);
        }
        return bigrams.toArray(new String[0]);
    }

    public static boolean isSentencePunctuation(String s) {
        return s.equals("./.") || s.equals("--/:") || s.equals(".") || s.equals("--");
    }

    public static String toCamelCasing(String name) {
        return name.substring(0, 1).toUpperCase() + name.substring(1);
    }
}
