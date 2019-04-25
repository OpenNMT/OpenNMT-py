package util.utils;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import fig.basic.Pair;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.function.BiFunction;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 *
 * @author ikonstas
 */
public class NamedEntityRecognizer implements NamedEntityRecognizerInterface {

    private final StanfordCoreNLP pipeline;
    private static final Pattern ANONYMIZED_ENTITY_MATCHER = Pattern.compile(".+_[0-9]+$");
    private final boolean tokenize;
    
    public NamedEntityRecognizer(boolean tokenize) {
        this.tokenize = tokenize;
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner");
        props.setProperty("tokenize.language", "Whitespace");
        props.setProperty("ner.model", "lib/models/ner/english.all.3class.distsim.crf.ser.gz");
        props.setProperty("pos.model", "lib/models/pos-tagger/english-left3words-distsim.tagger");
        pipeline = new StanfordCoreNLP(props);
        if(tokenize) {            
        }
    }

    private String tokenize(String text) {
        PTBTokenizer<CoreLabel> tokenizer = new PTBTokenizer<>(new StringReader(text), new CoreLabelTokenFactory(), "");
        return tokenizer.tokenize().stream().map(CoreLabel::toString).collect(Collectors.joining(" "));        
    }
    
    private Pair<List<String>, List<String>> process(String text, int length) {
        if(tokenize) {
            text = tokenize(text);
            length = text.split(" ").length;
        }
        Annotation document = new Annotation(text);
        pipeline.annotate(document);

        Collection<CoreLabel> tokens = new ArrayList<>();
        document.get(SentencesAnnotation.class).stream().forEach((sent) -> {
            sent.get(TokensAnnotation.class).stream().forEach((token) -> {
                tokens.add(token);
            });
        });                
        assert tokens.size() == length : "length mismatch. Found " + tokens.size() + ", was given " + length + " " + text;
        List<String> words = new ArrayList<>(length);
        List<String> namedEntities = new ArrayList<>(length);

        tokens.stream().forEach(token -> {
            String word = token.get(CoreAnnotations.TextAnnotation.class);
            String namedEntity = token.get(NamedEntityTagAnnotation.class);
            words.add(word);                        
            namedEntities.add(namedEntity);            
        });
//        System.out.println(words);
//        System.out.println(namedEntities);
        return new Pair(words, namedEntities);
    }

    @Override
    public String processToString(String text, int length) {
        Pair<List<String>, List<String>> tokensNers = process(text, length);
        return zip(tokensNers.getFirst().stream(),
                tokensNers.getSecond().stream(), (a, b) -> a + "/" + b).collect(Collectors.joining(" "));
    }

    @Override
    public String processToString(String text) {
        return processToString(text, text.split(" ").length);
    }
    
    public static boolean isAnonymizedEntity(String word) {
        return ANONYMIZED_ENTITY_MATCHER.matcher(word).matches();
    }


    public static void main(String[] args) {
//        String text = "Chinese officials stated that the launch of the nigcomsat- 1 aboard a Long March 3 B rocket on May 14 , 2007 represented a commercial challenge .";
        String text = "Township Vibes wonders who would want to become the president of Zimbabwe : “Today the Zimbabwean dollar is trading at $200 million to £1.";
        int length = text.split(" ").length;
        NamedEntityRecognizer spt = new NamedEntityRecognizer(true);
        Pair<List<String>, List<String>> tokensNers = spt.process(text, length);
        System.out.println("tokens = " + tokensNers.getFirst());
        System.out.println("ners = " + tokensNers.getSecond());
        String tokensNersToString = spt.processToString(text, length);
        System.out.println("NER'd sentence = " + tokensNersToString);                

    }
    
    private static <A, B, C> Stream<C> zip(Stream<? extends A> a, Stream<? extends B> b,
			BiFunction<? super A, ? super B, ? extends C> zipper) {
		final Iterator<? extends A> iteratorA = a.iterator();
		final Iterator<? extends B> iteratorB = b.iterator();
		final Iterable<C> iterable = () -> new Iterator<C>() {
			@Override
			public boolean hasNext() {
				return iteratorA.hasNext() && iteratorB.hasNext();
			}

			@Override
			public C next() {
				return zipper.apply(iteratorA.next(), iteratorB.next());
			}
		};
		return StreamSupport.stream(iterable.spliterator(), a.isParallel() || b.isParallel());
	}
}
