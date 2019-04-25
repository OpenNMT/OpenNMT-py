package util.corpus.wrappers;

import edu.berkeley.nlp.util.Lists;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.commons.lang3.text.WordUtils;
import util.utils.NamedEntityRecognizer;
import util.utils.NamedEntityRecognizerInterface;
import util.server.NamedEntityRecognizerClient;
import util.corpus.GigawordCorpus;

/**
 *
 * @author ikonstas
 */
public class GigawordWrapper {

    private final boolean verbose = false;
    private final GigawordCorpus corpus;
    private NamedEntityRecognizerInterface namedEntityRecognizer;
    private final List<NerSentence> annotatedSentences = new ArrayList<>();
    private static int nameCounter = 0, dateCounter = 0, numberCounter = 0, nlIdOffset = 0;
    private final boolean useNeClusters, filterVocabulary, processingGigaCorpus;
    private final Map<String, String> amrNeAnonymization, neNlAmrAlignments;
    private final Set<String> amrNlVocabulary;
    private final static Pattern ORDINAL_PATTERN = Pattern.compile("[st|nd|rd|th]");
    private final static Pattern NUMBER_ORDINAL_PATTERN = Pattern.compile("[0-9]+[st|nd|rd|th]");
    private final static Pattern NUMBER_PATTERN = Pattern.compile("[0-9]+");
    private final static Set<String> NUMBERS = new HashSet<>(Arrays.asList(new String[]{"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}));
    private final static Set<String> DATE_MODIFIERS = new HashSet<>(Arrays.asList(new String[]{"next", "previous", "last"}));
    private final static Set<String> WEEKDAYS = new HashSet<>(Arrays.asList(new String[]{"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun", "mon.", "tue.", "wed.", "thu.", "fri.", "sat.", "sun."}));
    private final static Set<String> MONTHS = new HashSet<>(Arrays.asList(new String[]{"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."}));

    public GigawordWrapper(String path, String annotationType, Map<String, String> amrNes, Map<String, Integer> amrNlVocabulary, 
            Map<String, String> neNlAmrAlignments, GigawordCorpus corpus) {
        this.amrNeAnonymization = amrNes;
        this.neNlAmrAlignments = neNlAmrAlignments;
        this.amrNlVocabulary = new HashSet<>(amrNlVocabulary.keySet());
        this.corpus = corpus;
        this.useNeClusters = corpus.isAnnotationUseNeClusters();
        this.filterVocabulary = corpus.isFilterVocabulary();
        this.processingGigaCorpus = corpus.getInfix().equals("giga"); // some heuristics are only applicable when processing the gigaword corpus
        namedEntityRecognizer = new NamedEntityRecognizer(false);        
        parse(path, annotationType);
    }

    public GigawordWrapper(Map<String, String> amrNes, Map<String, String> neNlAmrAlignments, boolean useNeClusters) {
        this.amrNeAnonymization = amrNes;
        this.neNlAmrAlignments = neNlAmrAlignments;
        this.amrNlVocabulary = null;
        this.corpus = null;
        this.useNeClusters = useNeClusters;
        this.filterVocabulary = false;
        this.processingGigaCorpus = false;
        try {        
            namedEntityRecognizer = new NamedEntityRecognizerClient(4444);
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }
    
    public void terminateNerClient() {
        namedEntityRecognizer.processToString("terminate_client");
    }
    
    private void parse(String path, String annotationType) {
        System.out.println("Processing sentences from " + path);
        try (Stream<String> stream = Files.lines(Paths.get(path))) {

            stream.map(line -> route(line, annotationType))
                    .map(this::normalize)
                    .filter(this::filterRawSentence)
                    .forEach(sentence -> {
                        NerSentence candidate = parseSentence(sentence, annotationType);
                        if(filterSentence(candidate)) {
                            annotatedSentences.add(candidate);
                            corpus.writeLinearize(annotatedSentences, true, false);
                        }
                    });
//            if(!annotatedSentences.isEmpty()) {
//                corpus.writeLinearize(annotatedSentences, true, true);
//            }
        } catch (IOException ioe) {
            ioe.printStackTrace(System.err);
        }
    }

    public NerSentence anonymizeRaw(String raw) {
        return parseSentence(route(raw, "raw"), "raw");        
    }
    
    private String route(String raw, String annotationType) {
        switch(annotationType) {
            case "ner": default: return raw;
            case "raw": return namedEntityRecognizer.processToString(normalizeBeforeNer(raw));
            case "rawWithId":
                String[] idRaw = raw.split("\t");
                return idRaw[0] + "\t" + namedEntityRecognizer.processToString(normalizeBeforeNer(idRaw[1]));
        }
    }
    
    private NerSentence parseSentence(String raw, String annotationType) {
        switch (annotationType) {
            case "ner": case "raw":
                return parseNer(raw, "Example_", true);            
            case "rawWithId": // NE recognize first. Input sentence has the format: unique_id \t raw_text
                String[] idRaw = raw.split("\t");
                return parseNer(idRaw[1], idRaw[0], false);
            default:
                throw new UnsupportedOperationException("Not supported yet");
        }
    }

    private NerSentence parseNer(String raw, String id, boolean generateId) {
        nameCounter = 0;
        dateCounter = 0;
        numberCounter = 0;
        nlIdOffset = 0;
        Map<List<String>, AmrAnonymizationAlignment> neMap = new HashMap<>();
        raw = !processingGigaCorpus ? preprocessDates(raw) : raw;
        String ar[] = raw.split(" ");
        List<String> sentence = new ArrayList<>(ar.length);
        List<String> anonSentence = new ArrayList<>(ar.length);
        for (int pos = 0; pos < ar.length; pos++) {
            int index = ar[pos].lastIndexOf("/");                 
            String word = ar[pos].substring(0, index);
            String ne = ar[pos].substring(index + 1);
            try {
                if (ne.equals("O")) {
//                    // don't trust the NER yet...
//                    List<String> temp = Lists.newList(word);
//                    List<Integer> tempIds = Lists.newList(nlIdOffset + pos);
//                    // check if the word already exists in the NE mapping
//                    String mappedNe = disambiguateMisc(temp);
//                    if(mappedNe != null) {                                                
//                        anonymizeEntity(temp, mappedNe, neMap, tempIds, sentence, anonSentence);                        
//                    } else {
                        sentence.add(word);
                        anonSentence.add(word);
//                    }
                } else {
                    List<String> multiNes = new ArrayList<>();
                    List<Integer> multiNesIds = new ArrayList<>();
                    multiNes.add(word);
                    multiNesIds.add(nlIdOffset + pos);
                    int j = pos + 1;
                    while (j < ar.length) {
                        String[] nextWordNe = ar[j].split("/");
                        if (nextWordNe[1].equals(ne)) {
                            multiNes.add(nextWordNe[0]);
                            multiNesIds.add(nlIdOffset + j++);
                        } else {
                            break;
                        }
                    } // while
                    pos = j - 1;
                    // anonymize newfound named entity and update anonymized sentence
                    anonymizeEntity(multiNes, ne, neMap, multiNesIds, sentence, anonSentence);
                } // else
            } catch (Exception e) {
                e.printStackTrace(System.err);
            }
        } // for
        return new NerSentence(id, generateId, sentence, anonSentence, neMap.values());
    }

    public List<NerSentence> getAnnotatedSentences() {
        return annotatedSentences;
    }

    private boolean filterRawSentence(String raw) {
        // remove lines with ;. Usually they contain enumerations of verbless sentences/facts, numbers, etc.        
        return !processingGigaCorpus || !(raw.contains(";/O") || raw.isEmpty());
    }

    /**
     * 
     * Remove any instance that has only a single word on the nl side.
     * @param sentence
     * @return 
     */
    private boolean filterSentence(NerSentence sentence) {
        return (!processingGigaCorpus || sentence.getTokens().size() > 1) && 
                (!filterVocabulary || amrNlVocabulary.containsAll(sentence.getTokens()));
    }
    
    private String normalizeBeforeNer(String raw) {
        return raw.replaceAll("@:@", ":").replaceAll("@-@", "-");
    }
            
    private String normalize(String raw) {
        // check first whether line contains an id tab-delimited from actual content
        String[] idRaw = raw.split("\t");
        boolean hasId = idRaw.length == 2;
        if(hasId) {
            raw = idRaw[1];
        } 
        String out = Arrays.stream(raw.split(" ")).filter(token -> {
//            int index = token.lastIndexOf("/");
            int index = token.indexOf("/");
            if (index <= 0) {
                return false;
            }
            String word = token.substring(0, index);
            // remove invalid word tokens
            return !(word.equals("#") || word.equals("^") || word.equals("\\"));
        }).map((String token) -> {
            int index = token.lastIndexOf("/");
            String word = token.substring(0, index);
            String ne = token.substring(index + 1);
            if (isAllUpperCase(word)) {
                word = WordUtils.capitalizeFully(word);
            }
            // replace words with sth else
            switch (word) {
                case "-LRB-":
                case "-LCB-":
                    word = "(";
                    break;
                case "-RRB-":
                case "-RCB-":
                    word = ")";
                    break;
                case "``":
                case "'": 
                case "''":
                    word = "\"";
                    break;                                    
                case "_":
                    word = "--";
                    break;
            }
            return word + "/" + ne;
        }).collect(Collectors.joining(" "));
        return hasId ? (idRaw[0] + "\t" + out) : out;
    }

    public static boolean isAllUpperCase(String str) {
        if (str == null || str.isEmpty()) {
            return false;
        }
        int sz = str.length();
        for (int i = 0; i < sz; i++) {
            if (!Character.isUpperCase(str.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    private void anonymizeEntity(List<String> multiNes, String namedEntity, Map<List<String>, AmrAnonymizationAlignment> identifiedNeMap, List<Integer> sentenceIds, List<String> sentence, List<String> anonSentence) {
        int anonSentencePos = anonSentence.size();
        AmrAnonymizationAlignment alignment = identifiedNeMap.get(multiNes);
        if (alignment != null) {
            sentence.addAll(multiNes);
            anonSentence.add(alignment.getAnonymizedToken());
            return;
        }
        String anonToken = "unk_0";
        switch (namedEntity) {
            case "PERSON":
                anonToken = getNeAnnotation(multiNes, "person_name_");
                break;
            case "ORGANIZATION":
                anonToken = getNeAnnotation(multiNes, "organization_name_");
                break;
            case "LOCATION":
                anonToken = getNeAnnotation(multiNes, "location_name_");
                break;
            case "NUMBER":
                String numCand = multiNes.get(0);
                anonToken = NUMBER_PATTERN.matcher(numCand).find() || NUMBERS.contains(numCand)? ("num_" + numberCounter++) : null;
                break;
            case "MISC":
                anonToken = disambiguateMisc(multiNes);
                break;
            case "TIME": // not handled at the moment
            case "SET":
            case "DURATION":
            case "__MISSING_NER_ANNOTATION__":
                anonToken = null;
                break;
            case "ORDINAL":
                anonToken = processOrdinal(multiNes, sentenceIds, identifiedNeMap, sentence, anonSentence, 0);
                break;
            case "PERCENT":
                anonToken = processPercent(multiNes, sentenceIds, identifiedNeMap, sentence, anonSentence);
                break;
            case "MONEY":
                anonToken = processMoney(multiNes, sentenceIds, identifiedNeMap, sentence, anonSentence);
                break;
            case "DATE":
                anonToken = processDate(multiNes, sentenceIds, identifiedNeMap, sentence, anonSentence);
                break;
            default:
                System.out.println("unexpected NE: " + namedEntity + " NL: " + String.join(" ", multiNes));
                break;
        }
        // add single NE alignment (majority case)
        if (anonToken != null) {
            sentence.addAll(multiNes);            
            alignment = new AmrAnonymizationAlignment(anonToken, nlToAmr(multiNes), multiNes, sentenceIds, anonSentencePos);
            identifiedNeMap.put(multiNes, alignment);
            anonSentence.add(alignment.getAnonymizedToken());
        // ordinal, percent, money, date have been dealt with in the corresponding methods
        } else if (!(namedEntity.equals("ORDINAL") || namedEntity.equals("PERCENT") || namedEntity.equals("MONEY") || namedEntity.equals("DATE"))) { 
            sentence.addAll(multiNes);
            anonSentence.addAll(multiNes);
        }
    }

    private String disambiguateMisc(List<String> multiNes) {
        String ne = amrNeAnonymization.get(String.join(" ", multiNes).toLowerCase());
        if (ne == null) {
            return null;
        }
        return ne + "_name_" + nameCounter++;
    }

    private String preprocessDates(String raw) {
        String[] ar = raw.split(" ");
        List<String> out = new ArrayList<>();
        // super hack: in cases the sentence is, YYYY - MM - DD, just delete the dashes; makes the processing much easier later on
        if (ar.length == 5 && ar[0].endsWith("DATE") && ar[1].equals("-/O") && ar[2].endsWith("DATE") && ar[3].equals("-/O") && ar[4].endsWith("DATE")) {
            return String.format("%s %s %s", ar[0], ar[2], ar[4]);
        }
        for(String tok : ar) {
            int index = tok.lastIndexOf("/");                 
            String word = tok.substring(0, index);
            String ne = tok.substring(index + 1);
            String format = "";
            String day = "", month = "", year = "";
            if(word.matches("[0-9]{6}")) { // YYMMDD
                format = "f1";
                String firstDigit = word.substring(0, 1);
                year = (firstDigit.equals("0") ? "20" : "19") + word.substring(0, 2);
                month = word.substring(2, 4);
                day = word.substring(5, 6);
            }
            else if(word.matches("[0-9]{8}")) { // YYYYMMDD
                format = "f2";
                year = word.substring(0, 4);
                month = word.substring(4, 6);
                day = word.substring(6, 8);
            }
            else if(word.matches("[0-9]{4}-[0-9]{2}-[0-9]{2}")) { // YYYY-MM-DD
                format = "f3";
                year = word.substring(0, 4);
                month = word.substring(5, 7);
                day = word.substring(8, 10);
            }
            if(!format.equals("")) {
                out.add(String.format("%s/DATE", day));
                out.add(String.format("%s/DATE", month));
                out.add(String.format("%s/DATE", year));
            } else {
                out.add(tok);
            }
        }        
        return String.join(" ", out);
    }
    
    private String getNeAnnotation(List<String> multiNes, String defaultNe) {
        if (!useNeClusters) {            
            String ne = disambiguateMisc(multiNes);
            if (ne != null) {
                return ne;
            }
        }
        return defaultNe + nameCounter++;
    }
    
    private String processOrdinal(List<String> multiNes, List<Integer> sentenceIds, Map<List<String>, AmrAnonymizationAlignment> neMap, List<String> sentence, List<String> anonSentence, int posInMultiNe) {
        if (multiNes.size() == 1) {// most common case, such as '3rd' or '21st' or second
            String word = multiNes.get(0);
            String anonToken = "ordinal-entity_num_" + numberCounter++;
            if (NUMBER_ORDINAL_PATTERN.matcher(word).find()) {
                Matcher m = ORDINAL_PATTERN.matcher(word);
                if (m.find()) {
                    String number = word.substring(0, m.start());
                    String suffix = word.substring(m.start());
                    // split number from suffix into two words, and offset accordingly all upcoming sentence ids                    
                    nlIdOffset++;
                    // every time we split a word, increment by one the rest of sentence NL ids
                    for (int i = posInMultiNe + 1; i < sentenceIds.size(); i++) {
                        sentenceIds.set(i, sentenceIds.get(i) + 1);
                    }
                    AmrAnonymizationAlignment alignment = new AmrAnonymizationAlignment(anonToken, nlToAmr(Lists.newList(number)), Lists.newList(number),
                            Lists.newList(sentenceIds.get(posInMultiNe)), anonSentence.size());
                    neMap.put(Lists.newList(number), alignment);
                    sentence.add(number);
                    sentence.add(suffix);
                    anonSentence.add(anonToken);
                    anonSentence.add(suffix);
                }
            } else { // e.g., first, second: don't do anything, as there is no simple consistent way to deal with them per AMR annotation.
                if (!(multiNes.get(0).equals("and") || multiNes.get(0).equals("to"))) {
                    AmrAnonymizationAlignment alignment = new AmrAnonymizationAlignment(anonToken, nlToAmr(multiNes), multiNes, sentenceIds, anonSentence.size());
                    neMap.put(multiNes, alignment);
                } else {
                    anonToken = word;
                }
                anonSentence.add(anonToken);
                sentence.add(word);
            }
        } else {
            for (int i = 0; i < multiNes.size(); i++) {
                processOrdinal(Lists.newList(multiNes.get(i)), sentenceIds, neMap, sentence, anonSentence, i);
            }
        }
        return null;
    }

    private String processPercent(List<String> multiNes, List<Integer> sentenceIds, Map<List<String>, AmrAnonymizationAlignment> neMap, List<String> sentence, List<String> anonSentence) {
        boolean processedSuccesfully = false;
        for (int i = 0; i < multiNes.size(); i++) {
            String word = multiNes.get(i);
            if (word.matches(".*\\d+.*")) {
                String anonToken = "percentage-entity_num_" + numberCounter++;
                processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                processedSuccesfully = true;
            } else {
                anonSentence.add(word);
                sentence.add(word);
            }
        }
        if (!processedSuccesfully && verbose) {
            System.out.println("unexpected NE: PERCENT NL: " + String.join(" ", multiNes));
        }
        return null;
    }

    private String processMoney(List<String> multiNes, List<Integer> sentenceIds, Map<List<String>, AmrAnonymizationAlignment> neMap, List<String> sentence, List<String> anonSentence) {
        boolean processedSuccesfully = false;
        for (int i = 0; i < multiNes.size(); i++) {
            String word = multiNes.get(i);
            if (word.matches(".*\\d+.*") || NUMBERS.contains(word.toLowerCase())) {
                String anonToken = "monetary-quantity_num_" + numberCounter++;
                processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                processedSuccesfully = true;
            } else {
                anonSentence.add(word);
                sentence.add(word);
            }
        }
        if (!processedSuccesfully && verbose) {
            System.out.println("unexpected NE: MONEY NL: " + String.join(" ", multiNes));
        }
        return null;
    }

    private String processDate(List<String> multiNes, List<Integer> sentenceIds, Map<List<String>, AmrAnonymizationAlignment> neMap, List<String> sentence, List<String> anonSentence) {
        boolean foundMonth = false, foundYear = false, foundDate = false, foundCentury = false, foundModifier = false;
        for (int i = 0; i < multiNes.size(); i++) {
            String word = multiNes.get(i);
            if (word.matches(".*\\d+.*")) { // process numbers
                if (word.length() == 4) { // most common year format                  
                    if (NUMBER_ORDINAL_PATTERN.matcher(word).find()) { // careful with 4-word words, it could be a case of a century ordial: e.g, 17th century
                        String anonToken = "ordinal-entity_num_" + numberCounter++;
                        processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                        foundCentury = true;
                    } else if (word.matches("\\d{4}")) { // only numbers here: not handling things like 9-15, etc.
                        int number = Integer.valueOf(word);                    
                        if (number > 999 && number < 10000) {
                            String anonToken = "year_date-entity_" + +dateCounter;
                            processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                            foundYear = true;
                        }                        
                    }
                } else if (word.length() == 5 && word.endsWith("s")) {// most common format for decades, e.g., 1960s
                    String anonToken = "year_date-entity_" + +dateCounter;
                    processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                    foundYear = true;
                } else if (word.matches("\\d+") && (word.length() == 2 || word.length() == 1)) {
                    try {
                    int number = Integer.valueOf(word);
                    if (foundMonth && number > 0 && number <= 31) { // most common format is: November 20, 2003, so month has already been found.
                        String anonToken = "day_date-entity_" + +dateCounter;
                        processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                        foundDate = true;
                    } else if (number > 0 && number <= 12) { // very bad heuristic: if month hasn't been discovered yet then it should be a numerical representation of a month
                        String anonToken = "month_date-entity_" + +dateCounter;
                        processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                        foundMonth = true;
                    } else if (number > 1 && number <= 31) { // very bad heuristic (Part II): otherwise it is a numerical day date
                        String anonToken = "day_date-entity_" + +dateCounter;
                        processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                        foundDate = true;
                    }
                    }catch(Exception e) {
                        System.out.println("here");
                    }
                }
            } else if (WEEKDAYS.contains(word.toLowerCase())) {
                String anonToken = "weekday_date-entity_" + dateCounter;
                processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                foundDate = true;
            } else if (MONTHS.contains(word.toLowerCase())) {
                String anonToken = "month_name_date-entity_" + dateCounter;
                processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                foundMonth = true;
            } else if(DATE_MODIFIERS.contains(word.toLowerCase())) {
                String anonToken = "mod_date-entity";
                processAlignment(anonToken, sentenceIds.get(i), word, neMap, sentence, anonSentence);
                foundModifier = true;
            } else {
                anonSentence.add(word);
                sentence.add(word);
            }
        }
        if (foundYear || foundMonth || foundDate) {
            dateCounter++;
        }
        boolean processedSuccesfully = foundYear || foundMonth || foundDate || foundCentury || foundModifier;
        if (!processedSuccesfully && verbose) {
            System.out.println("unexpected NE: DATE NL: " + String.join(" ", multiNes));
        }
        return null;
    }

    private void processAlignment(String anonToken, int posInSentence, String partialNamedEntity, Map<List<String>, AmrAnonymizationAlignment> neMap, List<String> sentence, List<String> anonSentence) {
        AmrAnonymizationAlignment alignment = new AmrAnonymizationAlignment(anonToken, nlToAmr(Lists.newList(partialNamedEntity)), Lists.newList(partialNamedEntity), Lists.newList(posInSentence), anonSentence.size());
        neMap.put(Lists.newList(partialNamedEntity), alignment);
        anonSentence.add(anonToken);
        sentence.add(partialNamedEntity);
    }

    private String nlToAmr(List<String> multiNes) {
        String key = String.join(" ", multiNes);
        return neNlAmrAlignments.getOrDefault(key.toLowerCase(), key);
    }
}
