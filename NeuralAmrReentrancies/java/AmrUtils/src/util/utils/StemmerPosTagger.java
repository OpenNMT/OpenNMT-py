package util.utils;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import fig.basic.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import util.Stemmer;

/**
 *
 * @author ikonstas
 */
public class StemmerPosTagger {

    private final StanfordCoreNLP pipeline;
    private static final Pattern ANONYMIZED_ENTITY_MATCHER = Pattern.compile(".+_[0-9]+$");
    private final Map<String, HistMap<String>> stemPosToStrings;    
    
    public StemmerPosTagger() {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        props.setProperty("tokenize.language", "Whitespace");
        props.setProperty("pos.model", "lib/models/pos-tagger/english-left3words-distsim.tagger");
        pipeline = new StanfordCoreNLP(props);
        
        stemPosToStrings = new HashMap<>();
    }

    public Pair<List<String>, List<String>> process(String text, int length, boolean lemmaOnly) {
        Annotation document = new Annotation(text);
        pipeline.annotate(document);

        Collection<CoreLabel> tokens = new ArrayList<>();
        document.get(SentencesAnnotation.class).stream().forEach((sent) -> {
            sent.get(TokensAnnotation.class).stream().forEach((token) -> {
                tokens.add(token);
            });
        });        
        assert tokens.size() == length : "length mismatch. Found " + tokens.size() + ", was given " + length + " " + text;
        List<String> stems = new ArrayList<>(length);
        List<String> posTags = new ArrayList<>(length);

        tokens.stream().forEach(token -> {
            String word = token.get(LemmaAnnotation.class).toLowerCase();
            String stem = lemmaOnly ? word : Stemmer.stem(word);
            stems.add(stem);
            String posTag;
            if(stem.equals("@-@")) {
                posTag = ":";
            }
            else if (stem.startsWith("num_")) {
                posTag = "NUM";
            } else if (stem.contains("date-entity")) {
                posTag = "DATE";
            } else if (isAnonymizedEntity(stem)) {
                posTag = "NAME";
            } else {
                posTag = token.get(PartOfSpeechAnnotation.class);
            }
            posTags.add(posTag);
            String key = stem + "^" + posTag;
            HistMap<String> hist = stemPosToStrings.getOrDefault(key, new HistMap<>());
            hist.add(token.get(CoreAnnotations.TextAnnotation.class).toLowerCase());
            stemPosToStrings.put(key, hist);
        });
//        System.out.println(stems);
//        System.out.println(posTags);
        return new Pair(stems, posTags);
    }

    public Pair<String, String> processToString(String text, int length) {
        Pair<List<String>, List<String>> stemsPosTags = process(text, length, false);
        return new Pair<>(stemsPosTags.getFirst().stream().collect(Collectors.joining(" ")),
                stemsPosTags.getSecond().stream().collect(Collectors.joining(" ")));
    }

    public static boolean isAnonymizedEntity(String word) {
        return ANONYMIZED_ENTITY_MATCHER.matcher(word).matches();
    }

    public Map<String, HistMap<String>> getStemPosToStrings() {
        return stemPosToStrings;
    }

    public String stemPosToStringsToString() {
        return stemPosToStringsToString(stemPosToStrings);
    }

    public static String stemPosToStringsToString(Map<String, HistMap<String>> map) {
        StringBuilder str = new StringBuilder();
        map.forEach((k, v) -> str
                .append(k).append("\t")
                .append(v.getEntriesSorted().stream()
                        .map(pair -> pair.getFirst() + " : " + pair.getSecond())
                        .collect(Collectors.joining(", ")))
                .append("\n"));
        return str.toString();
    }

    public static void main(String[] args) {
//        String text = "This is person_name_0 's num_0 bicycle , bought on year_date-entity_0 @-@ month_date-entity_0 @-@ day_date-entity_0 .";
        String text = "char·i·ty    @/@ ˈtʃærɪti Show Spelled [ char-i-tee ] Show IPA . noun , plural -@ ties .";
        StemmerPosTagger spt = new StemmerPosTagger();
        Pair<List<String>, List<String>> stemsPosTags = spt.process(text, 17, false);
        System.out.println("lemmata = " + stemsPosTags.getFirst());
        System.out.println("POS tags = " + stemsPosTags.getSecond());
        Pair<String, String> stemsPosTagsToString = spt.processToString(text, 17);
        System.out.println("lemmataToString = " + stemsPosTagsToString.getFirst());
        System.out.println("POS tags toString = " + stemsPosTagsToString.getSecond());
        System.out.println("\nThings will appear twice as we processed the same sentence twice above\n");
        System.out.println(spt.stemPosToStringsToString());

    }
}
