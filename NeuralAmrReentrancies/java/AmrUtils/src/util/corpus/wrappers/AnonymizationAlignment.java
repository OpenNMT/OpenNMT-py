package util.corpus.wrappers;

import java.util.List;

/**
 *
 * @author ikonstas
 */
public interface AnonymizationAlignment {

    public String getAnonymizedToken();

    public List<String> getCanonicalizedInputTokens();

    public List<String> getNlTokens();

    public List<Integer> getNlIds();

    public int getAnonymizedNlId();

    public boolean containsNode(int nodeWordId);
    
    public boolean isEmpty();

    public String nlToString();
}
