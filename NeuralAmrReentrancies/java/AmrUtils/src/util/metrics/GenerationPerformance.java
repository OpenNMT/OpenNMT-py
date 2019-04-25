package util.metrics;

import edu.berkeley.nlp.mt.BatchBleuScorer;
import edu.cmu.meteor.scorer.MeteorConfiguration;

import edu.cmu.meteor.scorer.MeteorScorer;
import edu.cmu.meteor.scorer.MeteorStats;
import fig.basic.Fmt;
import fig.basic.Pair;
import fig.basic.StopWatchSet;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Properties;
import util.utils.Utils;

/**
 * Compute BLEU, and METEOR for predicted generated text, given gold standard
 * text.
 *
 * @author konstas
 */
public class GenerationPerformance {

    public enum Metrics {
        BLEU, BLEU_ANONYMIZED, METEOR, METEOR_ANONYMIZED, POS
    }
    public static final DecimalFormat METRICS_FORMATTER = new DecimalFormat("#.####");
    private String experimentName;
    BatchBleuScorer bleuScorer, bleuAnonymizedScorer;
    MeteorScorer meteorScorer, meteorAnonymizedScorer;
    MeteorStats meteorAggStats, meteorAnonymizedAggStats;
    int total, generated;

    public GenerationPerformance(String experimentName) {
        System.err.println("Loading Performance Metrics...");
        this.experimentName = experimentName;
        bleuScorer = new BatchBleuScorer();
        bleuAnonymizedScorer = new BatchBleuScorer();
        Properties meteorProperties = new Properties();
        meteorProperties.setProperty("paraFile", "lib/models/meteor-1.5/paraphrase-en.gz");
        meteorProperties.setProperty("norm", "false");
        meteorScorer = new MeteorScorer(new MeteorConfiguration(meteorProperties));
        meteorAggStats = new MeteorStats();
        meteorAnonymizedScorer = new MeteorScorer(new MeteorConfiguration(meteorProperties));
        meteorAnonymizedAggStats = new MeteorStats();
        total = 0;
        generated = 0;
    }

    public void reset() {
        bleuScorer = new BatchBleuScorer();
        bleuAnonymizedScorer = new BatchBleuScorer();
        meteorAggStats = new MeteorStats();
        meteorAnonymizedAggStats = new MeteorStats();
        total = 0;
        generated = 0;
    }

    public void add(Hypothesis reference, Hypothesis prediction) {
        if (reference == null) {
            return;
        }
        if (prediction.equals(Hypothesis.newEmptyHypothesis())) {
            total++;
            return;
        }
        String predAnonStr = prediction.anonymizedToString().toLowerCase();
        String predStr = prediction.deAnonymizedToString().toLowerCase();
        String trueAnonStr = reference.anonymizedToString().toLowerCase();
        String trueStr = reference.deAnonymizedToString().toLowerCase();
        // Compute BLEU
        double bleuScore = bleuScorer.evaluateBleu(predStr, trueStr);
        // Compute anonymized BLEU
        double bleuAnonymizedScore = bleuAnonymizedScorer.evaluateBleu(predAnonStr, trueAnonStr);
        // Compute METEOR
        MeteorStats meteorStats = meteorScorer.getMeteorStats(predStr, trueStr);
        meteorAggStats.addStats(meteorStats);
        MeteorStats meteorAnonStats = meteorAnonymizedScorer.getMeteorStats(predAnonStr, trueAnonStr);
        meteorAnonymizedAggStats.addStats(meteorAnonStats);
        // Compute Event Precision, Recall, F-measure
        //            EvalResult subResult = computeFmeasure((GenWidget)trueWidget, (GenWidget)predWidget);
        //            double precision = subResult.precision();
        //            double recall = subResult.recall();
        //            double f1 = subResult.f1();
        //            double wer = computeWer(trueWidget, predWidget);

        prediction.setMetric(Metrics.BLEU, bleuScore);
        prediction.setMetric(Metrics.BLEU_ANONYMIZED, bleuAnonymizedScore);
        prediction.setMetric(Metrics.METEOR, meteorStats.score);
        prediction.setMetric(Metrics.METEOR_ANONYMIZED, meteorAnonStats.score);
        total++;
        generated++;

    }    

    /**
     * Print metrics
     *
     * @return
     */
    public String output() {
        meteorScorer.computeMetrics(meteorAggStats);
        meteorAnonymizedScorer.computeMetrics(meteorAnonymizedAggStats);
        String out = String.format("Generation Experiment: %s\n\n", experimentName);
        out += String.format("Generated %s out of %s examples.\n", generated, total);
        out += "\n----------------------\n";
        out += "Performance statistics";
        out += "\n----------------------\n";
        out += StopWatchSet.getStats().toString();
        out += "\n\n-----------\n";
        out += "BLEU scores";
        out += "\n-----------\n";
        out += bleuScorer.getScore().toString();
        out += "\n\nAnonymized BLEU scores";
        out += "\n-----------\n";
        out += bleuAnonymizedScorer.getScore().toString();
        out += "\n\nMETEOR scores";
        out += "\n-------------";
        out += "\nTest words:\t\t" + meteorAggStats.testLength;
        out += "\nReference words:\t" + meteorAggStats.referenceLength;
        out += "\nChunks:\t\t\t" + meteorAggStats.chunks;
        out += "\nPrecision:\t\t" + meteorAggStats.precision;
        out += "\nRecall:\t\t\t" + meteorAggStats.recall;
        out += "\nf1:\t\t\t" + meteorAggStats.f1;
        out += "\nfMean:\t\t\t" + meteorAggStats.fMean;
        out += "\nFragmentation penalty:\t" + meteorAggStats.fragPenalty;
        out += "\n";
        out += "\nFinal score:\t\t" + Fmt.D(meteorAggStats.score);
        out += "\n-----------\n";
        out += "\n\nAnonymized METEOR scores";
        out += "\n-------------";
        out += "\nTest words:\t\t" + meteorAnonymizedAggStats.testLength;
        out += "\nReference words:\t" + meteorAnonymizedAggStats.referenceLength;
        out += "\nChunks:\t\t\t" + meteorAnonymizedAggStats.chunks;
        out += "\nPrecision:\t\t" + meteorAnonymizedAggStats.precision;
        out += "\nRecall:\t\t\t" + meteorAnonymizedAggStats.recall;
        out += "\nf1:\t\t\t" + meteorAnonymizedAggStats.f1;
        out += "\nfMean:\t\t\t" + meteorAnonymizedAggStats.fMean;
        out += "\nFragmentation penalty:\t" + meteorAnonymizedAggStats.fragPenalty;
        out += "\n";
        out += "\nFinal score:\t\t" + Fmt.D(meteorAnonymizedAggStats.score);

//        out += "\n\nPrecision - Recall - F-measure - Record WER";
//        out += "\n------------------------------";
//        out += "\nTotal Precision: " + Fmt.D(result.precision());
//        out += "\nTotal Recall: " + Fmt.D(result.recall());
//        out += "\nTotal F-measure: " + Fmt.D(result.f1());
//        out += "\nTotal Record WER: " + Fmt.D(totalWer / (float) totalCounts);
        return out;
    }

    public double getAccuracy() {
        meteorScorer.computeMetrics(meteorAggStats);
        return meteorAggStats.score;
    }

    public String getNameMetricsShortHeader(boolean multiBleu) {
        return "Experiment Name\tBLEU\tBLEU anonymized\tMETEOR\tMETEOR anonymized" + (multiBleu ? "\tMULTI-BLEU\n" : "\n");

    }

    public String[] getMetricsNames(boolean multiBleu) {
        return multiBleu ? new String[] {"BLEU", "BLEU anonymized", "METEOR", "METEOR anonymized", "multiBLEU"} 
                         : new String[] {"BLEU", "BLEU anonymized", "METEOR", "METEOR anonymized"};
    }
    
    public Pair<String, double[]> getMetrics(double multiBleu) {
        meteorScorer.computeMetrics(meteorAggStats);
        meteorAnonymizedScorer.computeMetrics(meteorAnonymizedAggStats);
        if (multiBleu == -1) {
            return new Pair<>(experimentName, new double[] {bleuScorer.getScore().getScore(),
                    bleuAnonymizedScorer.getScore().getScore(), meteorAggStats.score, meteorAnonymizedAggStats.score});
        } else {            
            return new Pair<>(experimentName, new double[] {bleuScorer.getScore().getScore(),
                    bleuAnonymizedScorer.getScore().getScore(), meteorAggStats.score, meteorAnonymizedAggStats.score, multiBleu});
        }
    }
    
    public String getNameMetricsShort() {
        return getNameMetricsShort(null);
    }

    public String getNameMetricsShort(String multiBleuScore) {
        meteorScorer.computeMetrics(meteorAggStats);
        meteorAnonymizedScorer.computeMetrics(meteorAnonymizedAggStats);
        if (multiBleuScore == null) {
            return String.format("%s\t%s\t%s\t%s\t%s\n", experimentName, bleuScorer.getScore().getScore(),
                    bleuAnonymizedScorer.getScore().getScore(), meteorAggStats.score, meteorAnonymizedAggStats.score);
        } else {
            return String.format("%s\t%s\t%s\t%s\t%s\t%s\n", experimentName, bleuScorer.getScore().getScore(),
                    bleuAnonymizedScorer.getScore().getScore(), meteorAggStats.score, meteorAnonymizedAggStats.score, multiBleuScore);
        }
    }

    public void writeToFile(String path) {
        Utils.write(path, output());
    }

    public void writeToFile(PrintWriter writer) {
        writer.println(output());
    }

    public static String stripPunctuationFromEnd(String sentence) {
        int lastId = sentence.lastIndexOf(".");
        return lastId == -1 ? sentence : sentence.substring(0, lastId).trim();
    }

    public void setExperimentName(String experimentName) {
        this.experimentName = experimentName;
    }

}
