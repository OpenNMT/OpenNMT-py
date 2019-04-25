package util.corpus.wrappers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 *
 * @author ikonstas
 */
public class AmrAnonymizationAlignment implements AnonymizationAlignment{

    String anonymizedToken, amrToken;
    List<String> nlTokens;
    List<Integer> nlIds;
    int anonymizedNlId;

    public AmrAnonymizationAlignment() {
    }
    
    /**
     * Constructor from raw input. Example input is:
     * university_name_1|||name_University_of_Vermont|||University of
     * Vermont|||47,48,49|||45
     *
     * @param raw
     */
    public AmrAnonymizationAlignment(String raw) {
        String[] tokens = raw.split("\\|\\|\\|");
        assert tokens.length == 5 : "Wrong anonymization alignment format: should contain exactly 5 tokens.";
        this.anonymizedToken = tokens[0];
        this.amrToken = tokens[1];
        this.nlTokens = Arrays.asList(tokens[2].split(" "));
        this.nlIds = readIntegerList(tokens[3]);
        this.anonymizedNlId = Integer.valueOf(tokens[4]);
    }

    public AmrAnonymizationAlignment(String anonymizedToken, String amrTokens) {
        this(anonymizedToken, amrTokens, new ArrayList<>(), new ArrayList<>(), -1);
    }

    public AmrAnonymizationAlignment(String anonymizedToken, String amrTokens, List<String> nlTokens, List<Integer> nlIds, int anonymizedNlId) {
        this.anonymizedToken = anonymizedToken;
        this.amrToken = amrTokens.replaceAll("\"", "");
        this.nlTokens = nlTokens;
        this.nlIds = nlIds;
        this.anonymizedNlId = anonymizedNlId;
    }

    @Override
    public boolean isEmpty() {
        return anonymizedNlId == -1;
    }
    @Override
    public String getAnonymizedToken() {
        return anonymizedToken;
    }

    public String getRawAmrToken() {
        return amrToken;
    }

    /**
     * 
     * Get canonicalized (i.e., non-inflected nominal form) corresponding to an input token.
     * Examples: name_Luke becomes, Luke, name_United_Kingdom becomes United Kingdom.
     * @return 
     */
    @Override
    public List<String> getCanonicalizedInputTokens() {
        return Arrays.asList(amrToken.substring(amrToken.indexOf("_") + 1).split("_"));
    }

    @Override
    public List<String> getNlTokens() {
        return nlTokens;
    }

    @Override
    public List<Integer> getNlIds() {
        return nlIds;
    }

    @Override
    public int getAnonymizedNlId() {
        return anonymizedNlId;
    }

    /**
     * 
     * Returns true if the given node (predicate, or argument from an EasySRL propositions) OR part of it
     * corresponds to the anonymized token of this instance.
     * @param nodeWordId
     * @return 
     */
    @Override
    public boolean containsNode(int nodeWordId) {
        return nlIds.contains(nodeWordId);
    }
    
    public void adjust(int wordToBeAdjustedIndex, boolean anonymizedId, boolean negative) {
        if (anonymizedId) {
            if (wordToBeAdjustedIndex < anonymizedNlId) {
                if(negative)
                    anonymizedNlId--;
                else
                    anonymizedNlId++;
            }
        } else {
            for (int i = 0; i < nlIds.size(); i++) {
                if (wordToBeAdjustedIndex < nlIds.get(i)) {
                    if(negative)
                        nlIds.set(i, nlIds.get(i) - 1);
                    else
                        nlIds.set(i, nlIds.get(i) + 1);
                }
            }
        }
    }

    public int relativePosOfHyphenInNl(int index) {
        if (nlIds.isEmpty() || nlIds.size() == 1) {
            return -1;
        }
        if (index > nlIds.get(0) && index < nlIds.get(nlIds.size() - 1)) {
            for (int i = 1; i < nlIds.size(); i++) {
                if (index < nlIds.get(i)) {
                    return i - 1;
                }
            }
        }
        return -1;
    }

    @Deprecated
    protected String tokensToString(List<String> tokens) {
        if (tokens.size() > 0) {
            StringBuilder str = new StringBuilder();
            tokens.stream().forEach((token) -> {
                str.append(token).append(" ");
            });
            return str.toString().trim();
        }
        return " ";
    }

    @Override
    public String nlToString() {
        return String.join(" ", nlTokens);
    }
    
    protected String idsToString() {
        if (nlIds.size() > 0) {
            StringBuilder str = new StringBuilder();
            nlIds.stream().forEach((id) -> {
                str.append(id).append(",");
            });
            return str.deleteCharAt(str.length() - 1).toString();
        }
        return " ";
    }

    private List<Integer> readIntegerList(String raw) {
        List<Integer> list = new ArrayList<>();
        if (!raw.equals(" ")) {
            for (String token : raw.split(",")) {
                list.add(Integer.valueOf(token));
            }
        }
        return list;
    }

    public String amrNlToString() {
        return amrToken + "|||" + nlTokens.get(0);
    }
    
    public boolean equalsAmr(AmrAnonymizationAlignment alignIn) {
        return alignIn.amrToken.equals(this.amrToken);
    }  
    
    @Override
    public boolean equals(Object obj) {
        assert obj instanceof AmrAnonymizationAlignment;
        AmrAnonymizationAlignment o = (AmrAnonymizationAlignment) obj;
        return anonymizedToken.equals(o.anonymizedToken) && amrToken.equals(o.amrToken)
                && nlIds.equals(o.nlIds) && anonymizedNlId == o.anonymizedNlId;
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 89 * hash + Objects.hashCode(this.anonymizedToken);
        hash = 89 * hash + Objects.hashCode(this.amrToken);
        hash = 89 * hash + Objects.hashCode(this.nlIds);
        hash = 89 * hash + this.anonymizedNlId;
        return hash;
    }

    @Override
    public String toString() {
        return String.format("%s|||%s|||%s|||%s|||%s", anonymizedToken, amrToken, tokensToString(nlTokens), idsToString(), anonymizedNlId);
    }

}
