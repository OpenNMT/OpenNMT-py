package util.corpus;

import fig.basic.IOUtils;
import fig.basic.Indexer;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import util.utils.Database;
import util.utils.HistMap;
import util.utils.Settings;
import util.utils.StemmerPosTagger;
import util.utils.Utils;
import util.corpus.wrappers.AmrAnonymizationAlignment;
import util.corpus.wrappers.AmrLinearizedSentence;
import util.corpus.wrappers.AmrSentence;
import util.corpus.wrappers.AmrWrapper;
import util.corpus.wrappers.Dictionaries;
import util.corpus.wrappers.JamrAlignment;

/**
 *
 * @author konstas
 */
public class AmrCorpus extends AbstractCorpus {

    final String outputFilename;
    final String[] inputPaths;
    final String exportType, linearizeType;

    private final Dictionaries dictionaries;
    private Map<String, Map<String, JamrAlignment>> jamrAlignments;
    private Indexer<String> amrVocabulary, nlVocabulary;
    private Map<String, HistMap<String>> amrAlignedNlVocabulary;
    private boolean augmentLexicon;
    private final boolean deAnonymizeOutput, useNeClusters, lowerCaseOutput, includeReentrances,
            reentrancesRoles, outputBrackets, reshuffleChildren, markLeaves, concatBracketsWithRoles, outputSense;
    private final int amrThres, nlThres;
    private final StemmerPosTagger stemmerPostagger;

    public AmrCorpus(Settings settings, Database db, String corpus) {
        super(settings, db, corpus);
        String basePath = settings.getProperty(corpus + ".down.base");
        int i = 0;
        String[] inputs = settings.getProperty(corpus + ".down.input").split(",");
        inputPaths = new String[inputs.length];
        for (String tok : inputs) {
            inputPaths[i++] = basePath + tok;
        }
        dictionaries = new Dictionaries(settings, true, "amr", false);
        outputFilename = settings.getProperty(corpus + ".down.output");
        exportType = settings.getProperty(corpus + ".export.type");
        linearizeType = settings.getProperty(corpus + ".linearize.type");
        amrThres = Integer.valueOf(settings.getProperty(corpus + ".threshold.amr"));
        nlThres = Integer.valueOf(settings.getProperty(corpus + ".threshold.nl"));
        stemmerPostagger = new StemmerPosTagger();
        deAnonymizeOutput = settings.getProperty(corpus + ".down.deAnonymize").equals("true");
        useNeClusters = settings.getProperty(corpus + ".down.useNeClusters").equals("true");
        includeReentrances = settings.getProperty(corpus + ".down.includeReentrances").equals("true");
        reentrancesRoles = settings.getProperty(corpus + ".down.reentrancesRoles").equals("true");
        outputBrackets = settings.getProperty(corpus + ".down.outputBrackets").equals("true");
        reshuffleChildren = settings.getProperty(corpus + ".down.reshuffleChildren").equals("true");
        markLeaves = settings.getProperty(corpus + ".down.markLeaves").equals("true");
        lowerCaseOutput = settings.getProperty(corpus + ".down.lowercase").equals("true");
        concatBracketsWithRoles = settings.getProperty(corpus + ".down.concatBracketsWithRoles").equals("true");
        outputSense = settings.getProperty(corpus + ".down.outputSense").equals("true");
    }

    @Override
    public void preprocess() {
        List<AmrSentence>[] amrSentences = new ArrayList[inputPaths.length];
        if (exportType.equals("linearize")) {
            preLoadVocabularies();
            preLoadLinearizeDictionaries();
        }

        int i = 0;
        for (String inputPath : inputPaths) {
            amrSentences[i] = new ArrayList<>();
            for (String inputFile : listTxtFiles(inputPath)) {
                System.out.println("Processing file " + inputFile);
                String[] lines = Utils.readLines(inputPath + "/" + inputFile);
                AmrWrapper annotatorFile = new AmrWrapper(getLinearizeType(), isLinearize(), getJamrAlignments(),
                        getDictionaries(), lines);
                annotatorFile.parse();
                amrSentences[i].addAll(annotatorFile.getAmrSentences());
            }
            i++;
        }
        try {
            switch (exportType) {
                case "linearize":
                    writeLinearize(amrSentences);
                    break;
                default: throw new UnsupportedOperationException("Not supported yet.");
            }

        } catch (IOException | UnsupportedOperationException ioe) {
            ioe.printStackTrace(System.err);
        }
    }

    private void preLoadLinearizeDictionaries() {
        jamrAlignments = readJamrAlignments(settings.getProperty(corpus + ".jamr.alignments"));
        amrAlignedNlVocabulary = new HashMap<>();
    }

    private void preLoadVocabularies() {

        amrVocabulary = new Indexer<>();
        amrVocabulary.add("<unk>");
        amrVocabulary.add("<s>");
        amrVocabulary.add("</s>");
        nlVocabulary = new Indexer<>();
        nlVocabulary.add("<unk>");
        nlVocabulary.add("<s>");
        nlVocabulary.add("</s>");

    }

    private void writeLinearize(List<AmrSentence>[] amrSentences) throws IOException {
        int i = 0;
        for (String infix : settings.getProperty(corpus + ".down.input").split(",")) {
            HistMap<String> amrHist = new HistMap<>();
            HistMap<String> nlHist = new HistMap<>();
            Map<String, HistMap<String>> amrNlAlignmentsMap = new HashMap<>();
            augmentLexicon = infix.equals("training");
            boolean useNlOov = !augmentLexicon && !infix.endsWith("test");
            PrintWriter rawIdsAmrWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-raw-amr-ids.txt", outputFilename, infix)));
            PrintWriter linearAmrAnonymizedNlPairWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-%s-linear.txt", outputFilename, infix, linearizeType)));
//            PrintWriter linearAmrAnonymizedStemsPosTagsWriter = new PrintWriter(new FileOutputStream(
//                    String.format("%s%s-%s-stems-posTags-linear.txt", outputFilename, infix, linearizeType)));
            PrintWriter anonymizedAlignmentsWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-anonymized-alignments.txt", outputFilename, infix)));
            PrintWriter jamrAlignmentsWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-jamr-alignments.txt", outputFilename, infix)));
            PrintWriter contentAlignmentsWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-content-alignments.txt", outputFilename, infix)));
            PrintWriter amrNlAlignmentsSeqWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-amr-nl-alignments-seq.txt", outputFilename, infix)));
            PrintWriter propositionAmrAnonymizedNlPairWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-propositions.txt", outputFilename, infix)));
            PrintWriter nlWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-nl.txt", outputFilename, infix)));
            PrintWriter realIdToNumIdMap = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-ids-map.txt", outputFilename, infix)));
            PrintWriter amrVocabWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-amr-vocab.txt", outputFilename, infix)));
            PrintWriter nlVocabWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-nl-vocab.txt", outputFilename, infix)));
//                PrintWriter idEncodedPairWriter = new PrintWriter(new FileOutputStream(
//                        String.format("%s%s-%s-linear.num", outputFilename, infix, linearizeType)));
            System.out.println("Total number of sentences (before alignment heuristics): " + amrSentences[i].size());
            for (AmrSentence amrSentence : amrSentences[i]) // apply alignment heuristics
            {
                ((AmrLinearizedSentence) amrSentence).applyAlignmentHeuristics(amrNlAlignmentsMap, deAnonymizeOutput, includeReentrances, reentrancesRoles);
                if (augmentLexicon) {
                    ((AmrLinearizedSentence) amrSentence).updateVocabularies(amrVocabulary, amrHist, nlVocabulary, nlHist);
                }
		System.out.println(((AmrLinearizedSentence) amrSentence).getAnonymizedSentence());
                // merge jamr with (unsupervised) content alignments if necessary and then store AMR - NL (with counts) vocabulary entries
                Collection<AmrAnonymizationAlignment> mergedAlignments = mergeAlignments((AmrLinearizedSentence) amrSentence, false);
                mergedAlignments.stream().filter(alignment -> !alignment.getRawAmrToken().isEmpty()).forEach(alignment -> {
                    HistMap<String> hist = amrAlignedNlVocabulary.getOrDefault(alignment.getRawAmrToken(), new HistMap<>());
                    if (!alignment.getNlTokens().isEmpty()) {
                        hist.add(alignment.getNlTokens().get(0));
                        amrAlignedNlVocabulary.put(alignment.getRawAmrToken(), hist);
                    }
                });
            }
            Iterator<AmrSentence> it = amrSentences[i].iterator();
            while (it.hasNext()) // remove any instance that has only a single word either on the amr or nl side
            {
                AmrSentence s = it.next();
//                if(s.getAmr().size() == 1 || s.getAnonymizedSentenceSize() == 1 
//                        || (s.getAmr() instanceof AmrLinearize && ((AmrLinearize)s.getAmr()).propositionsSize() == 0))
                if (s.getAmr().isEmpty() || s.getAnonymizedSentenceSize() == 0) {
                    it.remove();
                }
            }
            // apply thresholding
            if (amrThres > 1) {
                amrVocabulary = applyThresholding(amrVocabulary, amrHist, amrThres);
            }
            if (nlThres > 1) {
                nlVocabulary = applyThresholding(nlVocabulary, nlHist, nlThres);
            }
            amrHist.getEntriesSorted().stream().map(e -> String.format("%s\t%s", e.getFirst(), e.getSecond())).forEach(amrVocabWriter::println);
            nlHist.getEntriesSorted().stream().map(e -> String.format("%s\t%s", e.getFirst(), e.getSecond())).forEach(nlVocabWriter::println);
            Collections.shuffle(amrSentences[i], new Random(1234)); // shuffle instances to avoid overfitting
            System.out.println("Total number of sentences (after alignment heuristics): " + amrSentences[i].size());
            for (AmrSentence amrSentence : amrSentences[i]) {
                rawIdsAmrWriter.println(amrSentence.rawIdsToString());
                linearAmrAnonymizedNlPairWriter.println(deAnonymizeOutput
                        ? (outputBrackets
                                ? ((AmrLinearizedSentence) amrSentence).toStringBrackets(reshuffleChildren, markLeaves, outputSense, concatBracketsWithRoles)
                                : amrSentence.toString())
                        : ((AmrLinearizedSentence) amrSentence).toStringAnonymized(outputBrackets, reshuffleChildren, markLeaves, outputSense, concatBracketsWithRoles));
//                linearAmrAnonymizedStemsPosTagsWriter.println(((AmrLinearizedSentence)amrSentence).toStringAmrNlStemsPosTags(stemmerPostagger));
                anonymizedAlignmentsWriter.println(((AmrLinearizedSentence) amrSentence).toStringAnonymizationAlignments(lowerCaseOutput));
                contentAlignmentsWriter.println(((AmrLinearizedSentence) amrSentence).toStringContentAlignments());
                jamrAlignmentsWriter.println(((AmrLinearizedSentence) amrSentence).toStringJamrAlignments());
                amrNlAlignmentsSeqWriter.println(((AmrLinearizedSentence) amrSentence).toStringAmrNlSeqAlignments(mergeAlignments((AmrLinearizedSentence) amrSentence, true)));
                propositionAmrAnonymizedNlPairWriter.println(((AmrLinearizedSentence) amrSentence).toStringPropositionsAnonymizedNl());
                nlWriter.println(((AmrLinearizedSentence) amrSentence).toStringNlOnly());
                realIdToNumIdMap.println(((AmrLinearizedSentence) amrSentence).toStringIdToNumIdOnly());
//                idEncodedPairWriter.println(((AmrLinearizedSentence)amrSentence).toStringLinearizeIndices(amrVocabulary, nlVocabulary, false));
            }
            // cache most frequent nl word for every amr token            
            amrAlignedNlVocabulary.values().stream().forEach(HistMap::setTopFreqElement);
            IOUtils.writeObjFileEasy(String.format("%s%s-io-alignments.obj.gz", outputFilename, infix), amrAlignedNlVocabulary);
            linearAmrAnonymizedNlPairWriter.close();
            rawIdsAmrWriter.close();
//            linearAmrAnonymizedStemsPosTagsWriter.close();
            // cache most frequent nl word for every stem-pos tag pair
//            stemmerPostagger.getStemPosToStrings().values().stream().forEach(hist -> hist.setTopFreqElement());
//            IOUtils.writeObjFileEasy(String.format("%s%s-stemPosTag-alignments.obj.gz", outputFilename, infix), stemmerPostagger.getStemPosToStrings());
//            IOUtils.writeLines(String.format("%s%s-stemPosTag-alignments.txt", outputFilename, infix), Arrays.asList(stemmerPostagger.stemPosToStringsToString().split("\n")));

            // write AMR-NL content-words-only vocabulary in the following format: amr_token|||nl_token
            // cache most frequent nl word for every amr token            
            amrNlAlignmentsMap.values().stream().forEach(HistMap::setTopFreqElement);
            Utils.writeLines(String.format("%s%s-amr-nl-alignments.txt", outputFilename, infix), amrNlAlignmentsMap.entrySet().stream()
                    .filter(entry -> !entry.getKey().equals(""))
                    .map(entry -> entry.getKey().toLowerCase() + "|||" + entry.getValue().getTopFreqElement().toLowerCase())
                    .collect(Collectors.toList()).toArray(new String[0]));

            Utils.writeLines(String.format("%s%s-amr-nl-alignments-hist.txt", outputFilename, infix), amrNlAlignmentsMap.entrySet().stream()
                    .filter(entry -> !entry.getKey().equals(""))
                    .map(entry -> entry.getKey() + "|||" + entry.getValue().toStringOneLine())
                    .collect(Collectors.toList()).toArray(new String[0]));

            // write NL-lemma(NL) content-words-only vocabulary in the following format: amr_token|||nl_token
            Utils.writeLines(String.format("%s%s-nl-lemma-alignments.txt", outputFilename, infix), nlVocabulary.stream()
                    .filter(entry -> !entry.equals(""))
                    .map(entry -> entry + "|||" + stemmerPostagger.process(entry, 1, true).getFirst().get(0))
                    .collect(Collectors.toList()).toArray(new String[0]));

            anonymizedAlignmentsWriter.close();
            contentAlignmentsWriter.close();
            jamrAlignmentsWriter.close();
            propositionAmrAnonymizedNlPairWriter.close();
            amrNlAlignmentsSeqWriter.close();
            nlWriter.close();
            realIdToNumIdMap.close();
            amrVocabWriter.close();
            nlVocabWriter.close();
//            idEncodedPairWriter.close();
            if (augmentLexicon) // save vocabularies if needed
            {
//                try (PrintWriter vocabulariesJsonWriter = new PrintWriter(new FileOutputStream(
//                        String.format("%s%s-%s-linear-vocabularies.json", outputFilename, infix, linearizeType)))) {
//                    vocabulariesJsonWriter.print(encodeVocabulariesToJson(amrVocabulary, nlVocabulary));
//                }                
            }
            System.out.println("AMR vocabulary size: " + amrVocabulary.size());
            System.out.println("NL vocabulary size: " + nlVocabulary.size());
//                System.out.println("AMR Pairs");
//                amrHist.getEntriesSorted().stream().forEach((pair) ->{System.out.println(pair);});
////                System.out.println(amrHist.getEntriesSorted());
//                System.out.println("NL Pairs");
//                nlHist.getEntriesSorted().stream().forEach((pair) ->{System.out.println(pair);});
            i++;
        }
        Utils.writeLines(String.format("%samr-roles-orderIds.txt", outputFilename), dictionaries.getAmrRoles().entrySet().stream()
                .map(entry -> entry.getKey() + "\t" + entry.getValue())
                .collect(Collectors.toList()).toArray(new String[0]));
    }

    private Indexer<String> applyThresholding(Indexer<String> vocabulary, HistMap<String> hist, int thres) {
        if (thres > 1) {
            Indexer<String> out = new Indexer<>();
            vocabulary.stream().forEach(key -> {
                int freq = hist.getFrequency(key);
                if (freq == -1 || freq > thres) {
                    out.add(key);
                }
            });
            return out;
        }
        return vocabulary;
    }

    private Map<String, Map<String, JamrAlignment>> readJamrAlignments(String path) {
        Map<String, Map<String, JamrAlignment>> map = new HashMap<>();
        System.out.println("Reading JAMR alignments from " + path);
        String[] lines = Utils.readLines(path);
        for (int i = 0; i < lines.length; i++) {
            if (lines[i].startsWith("# ::id ")) {
                String id = lines[i].split(" ")[2].trim();
                while (!lines[i].startsWith("# ::tok")) {
                    i++;
                }
                List<String> sentence = Arrays.asList(lines[i].substring(8).split(" "));
                while (!lines[i].startsWith("# ::node")) {
                    i++;
                }
                Map<String, JamrAlignment> alignments = new HashMap<>();
                while (lines[i].startsWith("# ::node")) {
                    JamrAlignment alignment = new JamrAlignment(sentence, lines[i++]);
                    alignments.put(alignment.getNodeId(), alignment);
                }
                map.put(id, alignments);
            }
        }
        return map;
    }

    private Collection<AmrAnonymizationAlignment> mergeAlignments(AmrLinearizedSentence amrSentence, boolean mergeAnonymizationAlignments) {
        Collection<AmrAnonymizationAlignment> contentAlignments = new HashSet<>(amrSentence.getContentAlignments());
        Collection<AmrAnonymizationAlignment> jamrSentAlignments = new HashSet<>(amrSentence.getJamrAlignments());
        contentAlignments.addAll(jamrSentAlignments.stream()
                .filter(jamr -> !contentAlignments.stream()
                        .anyMatch(content -> content.equalsAmr(jamr)))
                .map(alignment -> new JamrAlignment(alignment))
                .collect(Collectors.toSet()));
        if (mergeAnonymizationAlignments) {
            contentAlignments.addAll(amrSentence.getAnonymizationAlignments());
        }
        return contentAlignments;
    }

    private String[] listTxtFiles(String path) {
        return listFilesExt(path, ".txt");
    }

    private String[] listFilesExt(String path, String ext) {
        File f = new File(path);
        if (f.exists() && f.isDirectory()) {
            return f.list((File dir, String name) -> name.endsWith(ext));
        }
        return new String[0];
    }

    public String getExportType() {
        return exportType;
    }

    public Dictionaries getDictionaries() {
        return dictionaries;
    }

    public Map<String, Map<String, JamrAlignment>> getJamrAlignments() {
        return jamrAlignments;
    }

    public boolean isLinearize() {
        return exportType.equals("linearize");
    }

    public String getLinearizeType() {
        return linearizeType;
    }

    public boolean isDeanonymizeOutput() {
        return deAnonymizeOutput;
    }

    public boolean isLowerCaseOutput() {
        return lowerCaseOutput;
    }
}
