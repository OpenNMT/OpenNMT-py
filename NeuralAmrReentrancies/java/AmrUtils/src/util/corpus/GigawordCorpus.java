package util.corpus;

import fig.basic.Fmt;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IntSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import util.utils.Database;
import util.utils.HistMap;
import util.utils.Settings;
import util.corpus.wrappers.Dictionaries;
import util.corpus.wrappers.GigawordWrapper;
import util.corpus.wrappers.NerSentence;

/**
 *
 * @author konstas
 */
public class GigawordCorpus extends AbstractCorpus {
    
    final String outputFilename;
    final String inputPath;
    final String annotationType;
    private final Dictionaries dictionaries;    
    final String infix;
    final boolean annotationUseNeClusters, filterVocabulary;    
    private final Map<String, Integer> amrNlVocabulary;
    private final HistMap<String> nlAnonHist, nlHist;
    private final List<Integer> sentSizes;    
    private int steps = 0;    
    
    public GigawordCorpus(Settings settings, Database db, String corpus) {
        super(settings, db, corpus);
        inputPath = settings.getProperty(corpus + ".down.input");
        outputFilename = settings.getProperty(corpus + ".down.output");
        infix = settings.getProperty(corpus + ".down.infix");
        annotationType = settings.getProperty(corpus + ".annotation");
        annotationUseNeClusters = settings.getProperty(corpus + ".annotation.useNeClusters").equals("true");        
        filterVocabulary = settings.getProperty(corpus + ".annotation.filterVocabulary").equals("true");        
        nlAnonHist = new HistMap<>();
        nlHist = new HistMap<>();
        sentSizes = new ArrayList<>();
        dictionaries = new Dictionaries(settings, true, "gigaword", annotationUseNeClusters);
        amrNlVocabulary = dictionaries.getAmrNlVocabulary();
    }
    
    @Override
    public void preprocess() {
//        List<NerSentence> sentences = new ArrayList();
        
        GigawordWrapper annotatorFile = new GigawordWrapper(inputPath, annotationType, dictionaries.getAmrAnonymizationAlignments(), amrNlVocabulary, dictionaries.getNeNlAmrAlignments(), this);
        writeVocabularies();
        if(infix.equals("giga")) {
            printStats(sentSizes, nlHist, nlAnonHist);
        }
//        sentences.addAll(annotatorFile.getAnnotatedSentences());
        
//        try {
//            writeLinearize(sentences, false);
//        } catch (Exception ioe) {
//            ioe.printStackTrace(System.err);
//        }
    }   

    public void writeLinearize(List<NerSentence> sentences, boolean append, boolean force) {
        int step = 1000000;
        if (!append || force || sentences.size() > step) {
            steps += step;
            System.out.println("Processed (after filtering) " + steps + " sentences...");            
            try (PrintWriter nlWriter = new PrintWriter(new FileOutputStream(
                    String.format("%s%s-dfs-linear_targ.txt", outputFilename, infix), append));
//                    PrintWriter anonymizedAlignmentsWriter = new PrintWriter(new FileOutputStream(
//                            String.format("%s%s-anonymized-alignments.txt", outputFilename, infix), append));
//                    PrintWriter nlIdWriter = new PrintWriter(new FileOutputStream(
//                            String.format("%s%s-nl.txt", outputFilename, infix), append));
//                    PrintWriter nlAnonIdWriter = new PrintWriter(new FileOutputStream(
//                            String.format("%s%s-nl-anon.txt", outputFilename, infix), append))) 
                    ){

//        Collections.shuffle(sentences, new Random(1234)); // shuffle instances to avoid overfitting
                sentences.stream().forEach((NerSentence sentence) -> {
                    Arrays.asList(sentence.toStringNlAnonOnly().split(" ")).stream().forEach(nlAnonHist::add);
                    Arrays.asList(sentence.toStringNlOnly().split(" ")).stream().forEach(nlHist::add);
                    nlWriter.println(sentence.toStringNlAnonOnly());
//                    anonymizedAlignmentsWriter.println(sentence.toStringAnonymizationAlignments());
//                    nlIdWriter.println(sentence.getId() + "\t" + sentence.toStringNlOnly());
//                    nlAnonIdWriter.println(sentence.getId() + "\t" + sentence.toStringNlAnonOnly());
                    sentSizes.add(sentence.getTokens().size());

                });

//                anonymizedAlignmentsWriter.close();
//                nlIdWriter.close();
//                nlAnonIdWriter.close();

                if (append) { // if we are incrementally saving sentences in file then empty the list, to avoid filling up nemory
                    sentences.clear();
                }
            } catch (IOException e) {
                e.printStackTrace(System.err);
            }
        }
    }

    private void writeVocabularies() {        
        try (PrintWriter nlAnonVocabWriter = new PrintWriter(new FileOutputStream(
                String.format("%s%s-nl-vocab-anon.txt", outputFilename, infix)))) {
            // write NL vocabulary                       
            nlAnonHist.getEntriesSorted().stream().map(e -> String.format("%s\t%s", e.getFirst(), e.getSecond())).forEach(nlAnonVocabWriter::println);            
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
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

    public Map<String, Integer> getAmrNlVocabulary() {
        return amrNlVocabulary;
    }

    public boolean isAnnotationUseNeClusters() {
        return annotationUseNeClusters;
    }

    public boolean isFilterVocabulary() {
        return filterVocabulary;
    }

    public String getInfix() {
        return infix;
    }

    private void printStats(List<Integer> sentSizes, HistMap<String> gigaHist, HistMap<String> gigaAnonHist) {
        System.out.print("Collected dataset statistics after filtering: ");
        IntSummaryStatistics stats = sentSizes.stream().collect(Collectors.summarizingInt(Integer::intValue));        
        System.out.println(stats);

        System.out.println("---------");
        System.out.println("Original NL vocabulary from AMR size: " + amrNlVocabulary.size());
        System.out.println("Original NL vocabulary from AMR OOV rate (threshold = 1): " + Fmt.D(computeOovRate(amrNlVocabulary, 1)));
        System.out.println("Original NL vocabulary from AMR OOV rate (threshold = 2): " + Fmt.D(computeOovRate(amrNlVocabulary, 2)));
        System.out.println("Original NL vocabulary from AMR OOV rate (threshold = 5): " + Fmt.D(computeOovRate(amrNlVocabulary, 5)));
        System.out.println("Original NL vocabulary of which NE tokens are : " + dictionaries.getNumOfNlNeTokens()[0]);
        HistMap<String> neTokens = dictionaries.getNeTokens();
        System.out.println("Original NL vocabulary of which the NE portion of the vocabulary is : " + Fmt.D((double) neTokens.size() / (double) amrNlVocabulary.size()));
        System.out.println("Original NL vocabulary of which the NE OOV rate (threshold = 1): " + Fmt.D(computeOovRate(neTokens, 1)));
        System.out.println("Original NL vocabulary of which the NE OOV rate (threshold = 2): " + Fmt.D(computeOovRate(neTokens, 2)));
        System.out.println("Original NL vocabulary of which the NE OOV rate (threshold = 5): " + Fmt.D(computeOovRate(neTokens, 5)));
        System.out.println("---------");
        System.out.println("Giga vocabulary size: " + gigaHist.size());
        System.out.println("Giga anonymized vocabulary size: " + gigaAnonHist.size());
        System.out.println("Giga anonymized vocabulary OOV rate (threshold = 1): " + Fmt.D(computeOovRate(gigaAnonHist, 1)));
        System.out.println("Giga anonymized vocabulary OOV rate (threshold = 2): " + Fmt.D(computeOovRate(gigaAnonHist, 2)));
        System.out.println("Giga anonymized vocabulary OOV rate (threshold = 5): " + Fmt.D(computeOovRate(gigaAnonHist, 5)));

        Map<String, Integer> combinedAnonVocabulary = new HashMap<>();
        combinedAnonVocabulary.putAll(amrNlVocabulary);
        gigaAnonHist.getEntriesFreqs().stream().forEach(gigaEntry -> {
            Integer prevValue = combinedAnonVocabulary.get(gigaEntry.getKey());
            combinedAnonVocabulary.put(gigaEntry.getKey(), prevValue + gigaEntry.getValue());
        });
        System.out.println("---------");
        System.out.println("Combined Original AMR NL + Giga vocabulary size: " + combinedAnonVocabulary.size());
        System.out.println("Combined Original AMR NL + Giga vocabulary OOV rate (threshold = 1): " + Fmt.D(computeOovRate(combinedAnonVocabulary, 1)));
        System.out.println("Combined Original AMR NL + Giga OOV rate (threshold = 2): " + Fmt.D(computeOovRate(combinedAnonVocabulary, 2)));
        System.out.println("Combined Original AMR NL + Giga OOV rate (threshold = 5): " + Fmt.D(computeOovRate(combinedAnonVocabulary, 5)));
    }

    private double computeOovRate(Map<String, Integer> vocab, int threshold) {
        int total = vocab.size();
        int count = 0;
        for (Entry<String, Integer> e : vocab.entrySet()) {
            count = count + (e.getValue() <= threshold ? 1 : 0);
        }
        return (double) count / (double) total;
    }

    private double computeOovRate(HistMap<String> vocab, int threshold) {
        int total = vocab.size();
        int count = 0;
        for (Entry<String, Integer> e : vocab.getEntriesFreqs()) {
            count = count + (e.getValue() <= threshold ? 1 : 0);
        }
        return (double) count / (double) total;
    }
}
