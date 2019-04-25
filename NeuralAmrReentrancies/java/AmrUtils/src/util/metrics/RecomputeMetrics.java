package util.metrics;

import fig.basic.Fmt;
import fig.basic.Pair;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import uk.co.flamingpenguin.jewel.cli.ArgumentValidationException;
import uk.co.flamingpenguin.jewel.cli.CliFactory;
import uk.co.flamingpenguin.jewel.cli.Option;
import util.corpus.wrappers.AmrAnonymizationAlignment;
import util.corpus.wrappers.AnonymizationAlignment;
import util.utils.Utils;

/**
 *
 * @author ikonstas
 */
public class RecomputeMetrics {

    private final CommandLineArguments cmd;
    private final String outputFolder, inputFolder;
    private final boolean useMultiBleu;
    private String referencesFilename;
    private final GenerationPerformance performance;
    private Map<String, String> referenceExamples, referenceAnonymizedExamples;
    private Map<String, Map<String, AnonymizationAlignment>> anonymizedAlignments;
    private final boolean singleFileOnly, isInputAnonymized, isMultipleFiles, isGridExperiment;
    private List<String> ids;    

    /**
     * Command Line Interface
     */
    public interface CommandLineArguments {

        @Option(description = "Path that contains input folder")
        String getInputFolder();

        @Option(defaultValue = "", description = "Reference file used for evaluation.")
        String getReferenceFilename();

        @Option(defaultValue = "", description = "Reference file (anonymized) used for evaluation.")
        String getReferenceAnonymizedFilename();

        @Option(defaultValue = "", description = "Alignment file that contains mappings from input tokens to their anonymized counterparts (usually for NEs, dates and numerical entities.")
        String getAnonymizedAlignmentsFilename();

        @Option(description = "Anonymized input")
        boolean getIsInputAnonymized();

        @Option(defaultValue = "", description = "File containing the ids in the order of the generated output")
        String getIdsFilename();

        @Option(description = "Folder contains multiple prediction files")
        boolean getIsMultipleFiles();

        @Option(description = "Folder contains multiple folders of prediction files")
        boolean getIsGridExperiment();

    }

    public RecomputeMetrics(CommandLineArguments cmd) {
        this.cmd = cmd;
        this.inputFolder = cmd.getInputFolder();
        this.outputFolder = inputFolder;
        this.useMultiBleu = new File("./multi-bleu.perl").exists();
        this.isInputAnonymized = cmd.getIsInputAnonymized();
        this.isMultipleFiles = cmd.getIsMultipleFiles();
        this.isGridExperiment = cmd.getIsGridExperiment();
        this.singleFileOnly = new File(cmd.getInputFolder()).isFile(); // in case we want to quickly recompute a single file.        
        readReferenceFiles();
        performance = new GenerationPerformance("");
    }

    public void execute() {
        if (referenceExamples != null) {
            if (isMultipleFiles) {
                StringBuilder summary = new StringBuilder(performance.getNameMetricsShortHeader(useMultiBleu));
                List<Pair<String, double[]>> results = new ArrayList<>();
                if (isGridExperiment) {
                    try {
                        
                        Files.walk(Paths.get(inputFolder)).filter(p -> p.toFile().isDirectory()).forEach(p -> results.addAll(processFolder(summary, p.toString())));
                    } catch (IOException ioe) {
                        ioe.printStackTrace(System.err);
                    }
                } else {
                    results.addAll(processFolder(summary, inputFolder));
                }
                StringBuilder maxScores = maxScores(results, performance.getMetricsNames(useMultiBleu));
                System.out.println(maxScores);
                summary.append("\n\nMax scores:-----------\n").append(maxScores);
                Utils.write(outputFolder + "/summary.performance.recomputed", summary.toString());
            } else {
                // Read predicted output
                Map<String, String> predExamples = isInputAnonymized ? null : readFile(inputFolder + (singleFileOnly ? "" : "/decoder-simple.output"));
                // Read anonymized predicted output
                Map<String, String> predAnonExamples = isInputAnonymized ? readFile(inputFolder) : (singleFileOnly ? null : readFile(inputFolder + "/decoder-anonymized-simple.output"));
                if ((predExamples != null || predAnonExamples != null) && (singleFileOnly || (predAnonExamples != null && anonymizedAlignments != null))) {
                    Map<String, String> deAnonymizedPreds = new HashMap<>();
                    processFile(referenceExamples, referenceAnonymizedExamples, predExamples, predAnonExamples, deAnonymizedPreds, anonymizedAlignments);
                    writeOutput(outputFolder + (singleFileOnly ? ".performance.recomputed" : "/decoder.performance.recomputed"));
                }
            }
        }
    }

    private void readReferenceFiles() {
        System.out.println("Reading from " + (singleFileOnly ? "file " : "folder ") + inputFolder);
        // Read reference file ids
        ids = extractIds(cmd.getIdsFilename().equals("") ? cmd.getReferenceFilename() : cmd.getIdsFilename());
        // Read reference file
        referenceExamples = readFileWithIds(cmd.getReferenceFilename());
        // Read anonymized reference file
        referenceAnonymizedExamples = cmd.getReferenceAnonymizedFilename().equals("") ? Collections.EMPTY_MAP : readFileWithIds(cmd.getReferenceAnonymizedFilename());
        // read anonymization alignments
        anonymizedAlignments = (!singleFileOnly || isInputAnonymized) && !cmd.getAnonymizedAlignmentsFilename().equals("") ? readAmrAnonymizedAlignments(cmd.getAnonymizedAlignmentsFilename()) : null;
        if (useMultiBleu && !(referenceExamples == null || referenceExamples.isEmpty())) {
            try {
                File tempFilename = File.createTempFile("references-", ".txt");
                referencesFilename = tempFilename.getAbsolutePath();
                writeOutputInOrder(referenceExamples, referencesFilename);
                tempFilename.deleteOnExit();

            } catch (IOException ex) {
                ex.printStackTrace(System.err);
            }

        }
    }

    public static Map<String, Map<String, AnonymizationAlignment>> readAmrAnonymizedAlignments(String anonymizedAlignmentsFilename) {
        Map<String, Map<String, AnonymizationAlignment>> map = new HashMap<>();
        for (String line : Utils.readLines(anonymizedAlignmentsFilename)) {
            String[] ar = line.split("\t");
            if (ar.length > 2) {
                throw new InputMismatchException("Couldn't read anonymized alignments from: " + anonymizedAlignmentsFilename + " Error in line: " + line);
            }
            String id = ar[0];
            Map<String, AnonymizationAlignment> alignments = new HashMap<>();
            if (ar.length > 1) { // if the example has any anonymized tokens at all
                String[] alignmentsAr = ar[1].split(" # ");
                for (String alignment : alignmentsAr) {
                    AnonymizationAlignment a = new AmrAnonymizationAlignment(alignment);
//                    if(!a.isEmpty()) 
                    {
                        alignments.put(a.getAnonymizedToken(), a);
                    }
                }
            }
            map.put(id, alignments);
        }
        return map;
    }
    
    public static List<String> extractIds(String filename) {
        return new File(filename).exists() ? Arrays.asList(Utils.readLines(filename)).stream().
                map(line -> line.split("\t")[0]).collect(Collectors.toList()) : null;
    }

    public Map<String, String> readFile(String filename) {
        String[] lines = new File(filename).exists() ? Utils.readLines(filename) : null;
        if (lines == null) {
            return null;
        }
        // check if file has ids
        boolean hasIds = lines[0].split("\t").length > 1;
        if (hasIds) {
            return readFileWithIds(lines);
        } else {
            return readFileWithoutIds(lines);
        }
    }

    public static Map<String, String> readFileWithIds(String filename) {
        return readFileWithIds(Utils.readLines(filename));
    }

    public static Map<String, String> readFileWithIds(String[] lines) {
        Map<String, String> out = lines.length > 0 ? Arrays.asList(lines).stream().
                collect(Collectors.toMap((String line) -> {
                    return line.split("\t")[0];
                },
                        (String line) -> {
                            return line.split("\t")[1];
                        })) : null;
        return out;
    }

    public Map<String, String> readFileWithoutIds(String[] lines) {
        Map<String, String> out = new HashMap<>();
        for (int i = 0; i < lines.length; i++) {
            out.put(ids.get(i), lines[i]);
        }
        return out;
    }

    private List<Pair<String, double[]>> processFolder(StringBuilder summary, String inputFolder) {
        List<Pair<String, double[]>> results = new ArrayList<>();
//            Files.list(Paths.get(inputFolder)).filter(filePath -> filePath.endsWith(".txt")).forEach(filename -> {
        Arrays.asList(new File(inputFolder).list()).stream().filter(filePath -> filePath.endsWith(".txt")).forEach(filename -> {
//                String fullFilename = outputFolder + "/" + filename;
            String fullFilename = inputFolder + "/" + filename;
            System.out.println("Processing " + fullFilename);
            // Read predicted output
            Map<String, String> predExamples = isInputAnonymized ? null : readFile(inputFolder + "/" + filename);
            // Read anonymized predicted output
            Map<String, String> predAnonExamples = isInputAnonymized ? readFile(inputFolder + "/" + filename) : null;
            if (predExamples != null || (predAnonExamples != null && anonymizedAlignments != null)) {
                Map<String, String> deAnonymizedPreds = new HashMap<>();
                processFile(referenceExamples, referenceAnonymizedExamples, predExamples, predAnonExamples, deAnonymizedPreds, anonymizedAlignments);
                performance.setExperimentName(fullFilename);
                writeOutput(fullFilename + ".performance.recomputed");
                writeOutputInOrder(deAnonymizedPreds, fullFilename + ".deAnonymized");
                if (useMultiBleu) {
                    String multiBleuStr = computeMultiBleu(fullFilename + ".deAnonymized");
                    double multiBleu = Double.valueOf(multiBleuStr.split("[ ,]")[2]);
                    summary.append(performance.getNameMetricsShort(multiBleuStr));
                    results.add(performance.getMetrics(multiBleu));
                } else {
                    summary.append(performance.getNameMetricsShort());
                    results.add(performance.getMetrics(-1));
                }
                performance.reset();
            }
        }); // for every output file
        return results;
    }

    private void processFile(Map<String, String> referenceExamples, Map<String, String> referenceAnonymizedExamples,
            Map<String, String> predExamples, Map<String, String> predAnonExamples, Map<String, String> deAnonymizedPreds,
            Map<String, Map<String, AnonymizationAlignment>> anonymizedAlignments) {
        referenceExamples.forEach((id, referenceSentence) -> {            
            Hypothesis referenceHypothesis = referenceAnonymizedExamples.isEmpty() ? new Hypothesis(referenceSentence) : new Hypothesis(referenceSentence, referenceAnonymizedExamples.get(id));
            if ((singleFileOnly || isMultipleFiles) && isInputAnonymized) {
                if (predAnonExamples.containsKey(id)) {
                    Hypothesis predHypothesis = new Hypothesis(predAnonExamples.get(id));
                    predHypothesis.deAnonymize(anonymizedAlignments.get(id));
                    deAnonymizedPreds.put(id, predHypothesis.deAnonymizedToString());
                    performance.add(referenceHypothesis, predHypothesis);
                }
            } else if (singleFileOnly) {
                if (predExamples.containsKey(id)) {
                    referenceHypothesis.deAnonymizeFake();
                    Hypothesis predHypothesis = new Hypothesis(predExamples.get(id), predExamples.get(id));
//                            predHypothesis.deAnonymizeFake();
                    deAnonymizedPreds.put(id, predHypothesis.deAnonymizedToString());
                    performance.add(referenceHypothesis, predHypothesis);
                }
            } else if (predExamples.containsKey(id)) {
                if (predAnonExamples != null && predAnonExamples.containsKey(id)) {
                    Hypothesis predHypothesis = new Hypothesis(predExamples.get(id), predAnonExamples.get(id));
                    deAnonymizedPreds.put(id, predHypothesis.deAnonymizedToString());
                    performance.add(referenceHypothesis, predHypothesis);
                } else {
                    Hypothesis predHypothesis = new Hypothesis(predExamples.get(id));
                    predHypothesis.deAnonymizeFake();
                    referenceHypothesis.deAnonymizeFake();
                    performance.add(referenceHypothesis, predHypothesis);
                }
            }
        });
    }

    private StringBuilder maxScores(List<Pair<String, double[]>> results, String[] metricsNames) {
        StringBuilder out = new StringBuilder();
        int numOfResults = results.size();
        int numOfMetrics = metricsNames.length;
        String[] experimentNames = new String[numOfResults];
        double[][] scores = new double[numOfMetrics][numOfResults];        
        for(int i = 0; i < numOfResults; i++) {
            experimentNames[i] = results.get(i).getFirst();
            for(int j = 0; j < numOfMetrics; j++) {
                scores[j][i] = results.get(i).getSecond()[j];
            }
        }
        for(int j = 0; j < numOfMetrics; j++) {
            double[] score = scores[j];
            Pair<Integer, Double> maxScore = max(score);
            out.append(String.format("Max %s: %s\t%s\n", metricsNames[j], Fmt.D((double)maxScore.getSecond()), experimentNames[maxScore.getFirst()]));
        }
        return out;
    }
    
    private Pair<Integer, Double> max(double[] ar) {
        double max = ar[0];
        int maxId = 0;
        for(int i = 1; i < ar.length; i++) {
            if(ar[i] > max) {
                max = ar[i];
                maxId = i;
            }
        }
        return new Pair<>(maxId, max);
    }
    
    private String computeMultiBleu(String predFilename) {
        String bleu = null;
        try {
            String command = String.format("perl ./multi-bleu.perl -lc %s", referencesFilename);
            ProcessBuilder ps = new ProcessBuilder(command.split(" "));
            ps.redirectErrorStream(true);
            ps.redirectInput(new File(predFilename));
            Process pr = ps.start();
            try (BufferedReader in = new BufferedReader(new InputStreamReader(pr.getInputStream()))) {
                String line;
                while ((line = in.readLine()) != null) {
//                    bleu = Double.valueOf(line.split("[ ,]")[2]);
                    bleu = line;
                }
                pr.waitFor();
            }
        } catch (IOException | InterruptedException ex) {
            ex.printStackTrace(System.err);
        }
        return bleu;
    }

    private void writeOutput(String filename) {
        try (PrintWriter output = new PrintWriter(filename)) {
            output.println(performance.output());
            output.flush();
        } catch (IOException ioe) {
            ioe.printStackTrace(System.err);
        }
    }

    private void writeOutputInOrder(Map<String, String> deAnonymizedPreds, String filename) {
        try (PrintWriter out = new PrintWriter(new FileOutputStream(filename))) {
            ids.stream().forEach(id -> out.println(deAnonymizedPreds.get(id)));
        } catch (FileNotFoundException ex) {
            ex.printStackTrace(System.err);
        }
    }

    public static void main(String[] args) {

        try {
            final RecomputeMetrics.CommandLineArguments cmd = CliFactory.parseArguments(RecomputeMetrics.CommandLineArguments.class, args);
            RecomputeMetrics rm = new RecomputeMetrics(cmd);
            rm.execute();

        } catch (ArgumentValidationException ex) {
            System.err.println(ex);
        }
    }
        
}
