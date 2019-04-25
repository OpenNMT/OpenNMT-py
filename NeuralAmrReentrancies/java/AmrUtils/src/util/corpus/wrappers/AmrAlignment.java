package util.corpus.wrappers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author ikonstas
 */


public class AmrAlignment
{    
    
    public static enum TokenType {ROLE, CONCEPT}
    
    private String token;
    private final TokenType type;
    private final List<Integer> wordIds;
    private String[] words;

    /**
     * 
     * Constructor of alignments between amr tokens and sentence. The annotation
     * on the input token looks like, e.g.: establish-01~e.0 or :time~e.13,16.
     * We parse the part after '~e.' into an array of integer ids that correspond
     * to sentence words
     * @param type the type of the amr token: role or concept
     * @param rawAlignStr the raw string containing the alignment annotation
     */
    public AmrAlignment(TokenType type, String rawAlignStr)
    {
        this.type = type;
        
        int index = rawAlignStr.indexOf("~");        
        if(index >= 0)
        {
            token = rawAlignStr.substring(0, index);
            wordIds = parseAlignments(rawAlignStr);
        }
        else
        {
            token = rawAlignStr;
            wordIds = new ArrayList<>(Arrays.asList(new Integer[]{-1}));
        }
        words = new String[wordIds.size()];
    }

    public AmrAlignment(String token, String word, int wordId, TokenType type) {
        this.token = token;
        this.type = type;
        this.wordIds = Arrays.asList(wordId);
        this.words = new String[] {word};
    }

    
    public final static List<Integer> parseAlignments(String rawStr)
    {
        int index2 = rawStr.lastIndexOf(".");
        String ar[] = rawStr.substring(index2 + 1).split(",");
        List<Integer> ids = new ArrayList(ar.length);
        for(String id : ar)
            ids.add(Integer.valueOf(id));
        return ids;
    }
    
    public List<Integer> getWordIds()
    {
        return wordIds;
    }

    public String[] getWords() 
    {
        return words;
    }
    
    public void addAlignments(AmrAlignment alignment)
    {
        wordIds.addAll(alignment.wordIds);
        token += "_" + alignment.token.replaceAll("\"", "");
    }
    
    public String getToken()
    {
        return token;
    }

    public void setToken(String token) 
    {
        this.token = token;
    }

    public TokenType getType()
    {
        return type;
    }

    public void setWords(String sentence)
    {
        String[] ar = sentence.split(" ");
        for(int i = 0; i < wordIds.size(); i++)
        {
            assert wordIds.get(i) < ar.length;
            words[i] = ar[wordIds.get(i)];
        }
    }

    public void copyFromJamrAlignment(JamrAlignment jamr) 
    {
//        if(token.equals(jamr.amrToken) && !jamr.isEmpty()) // make 100% sure we are referring to the same AMR token
        {
            this.wordIds.clear();
            wordIds.addAll(jamr.nlIds);
            this.words = jamr.nlTokens.toArray(new String[0]);
        }
    }
    
    public boolean isEmpty()
    {
        return wordIds.isEmpty() || (wordIds.size() == 1 && wordIds.get(0) == -1);
    }
    
    @Override
    public String toString()
    {
        StringBuilder str = new StringBuilder(" " + token + " ||| ");
        for(int i = 0; i < wordIds.size(); i++)
            str.append(wordIds.get(i)).append(":").append(words[i]).append(", ");
        return str.toString();
    }
    
    
    
}
