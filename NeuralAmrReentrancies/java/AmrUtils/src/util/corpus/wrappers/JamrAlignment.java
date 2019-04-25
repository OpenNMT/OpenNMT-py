package util.corpus.wrappers;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author ikonstas
 */
public class JamrAlignment extends AmrAnonymizationAlignment {

    private final String nodeId;

    public JamrAlignment(AmrAnonymizationAlignment alignIn) {
        super();
        nodeId = null;
        this.amrToken = alignIn.amrToken;
        this.anonymizedNlId = alignIn.anonymizedNlId;
        this.anonymizedToken = alignIn.anonymizedToken;
        this.nlIds = alignIn.nlIds;
        this.nlTokens = alignIn.nlTokens;        
    }
    
    public JamrAlignment(List<String> sentence, String rawStr) {
        super();
        anonymizedToken = "-";
        String[] toks = rawStr.split("\t");
        this.nlTokens = new ArrayList<>();
        this.nodeId = toks[1];
        this.amrToken = stripSense(toks[2]);
        this.nlIds = new ArrayList<>();
        if (toks.length == 3) {
            nlIds.add(-1);
        } else {
            String[] startEnd = toks[3].split("-");
            int start = Integer.valueOf(startEnd[0]);
            int end = Integer.valueOf(startEnd[1]);
            switch (end - start) {
                case 1:
                    //one-word span
                    nlIds.add(start);
                    assert start >= 0 && start <= sentence.size() : "start=" + start + " sentence.size()=" + sentence.size() + " sentence=" + sentence;
                    nlTokens.add(sentence.get(start));
                    break;
                case 0:
                    //one-word span
                    start = end -1;
                    nlIds.add(start);
                    assert start >= 0 && start <= sentence.size() : "start=" + start + " sentence.size()=" + sentence.size() + " sentence=" + sentence;
                    nlTokens.add(sentence.get(start));
                    break;
                default:
//                    assert start >= 0 && end < sentence.size() : "start=" + start + " end=" + end + " sentence=" + sentence;
                    if (start >= 0 && end < sentence.size()) {
                        for (int i = start; i < end; i++) {
                            nlIds.add(i);
                            nlTokens.add(sentence.get(i));
                        }
                    } else {
                        nlIds.add(-1);
                    }
                    break;
            }
        }
        anonymizedToken = nodeId;
        try {
            anonymizedNlId = nlIds.get(0);
        } catch (Exception e) {
            System.out.println("Error reading JAMR alignment");
        }
    }

    public String getNodeId() {
        return nodeId;
    }

    @Override
    public boolean isEmpty() {
        return nlIds.isEmpty() || nlIds.get(0) == -1;
    }
    
    private String stripSense(String tok) {
        int id = tok.lastIndexOf("-");
        return id == -1 ? tok : tok.substring(0, id);
    }          
}
