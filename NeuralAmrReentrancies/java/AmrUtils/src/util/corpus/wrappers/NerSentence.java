package util.corpus.wrappers;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 *
 * @author ikonstas
 */


public class NerSentence extends AmrSentence {

    private final Collection<AmrAnonymizationAlignment> anonymizationAlignments;
    private final List<String> tokens;
    
    public NerSentence(String id, boolean generateId, List<String> sentence, List<String> anonymizedSentence, Collection<AmrAnonymizationAlignment> anonymizationAlignments) {
        super(id, String.join(" ", sentence));
        this.id = generateId ? (id + incrId) : id;
        this.anonymizedSentence = String.join(" ", anonymizedSentence);
        this.anonymizationAlignments = anonymizationAlignments;
        this.tokens = anonymizedSentence.stream().map(tok -> tok.toLowerCase()).collect(Collectors.toList());
    }
           
    public String toStringAnonymizationAlignments() {
        StringBuilder str = new StringBuilder();
        anonymizationAlignments.stream().forEach(an -> str.append(an).append(" # "));
        return String.format("%s\t%s", getId(), anonymizationAlignments.isEmpty() ? "" : str.substring(0, str.length() - 3).trim());
    }

    public Collection<AmrAnonymizationAlignment> getAnonymizationAlignments() {
        return anonymizationAlignments;
    }

    public List<String> getTokens() {
        return tokens;
    }
    
    public String toStringNlOnly() {
        return sentence.toLowerCase();
    }

    public String toStringNlAnonOnly() {
        return anonymizedSentence.toLowerCase();
    }
    
    @Override
    public String toString() {
        return sentence;
    }
  
}
