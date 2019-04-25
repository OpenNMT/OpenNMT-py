package util.corpus.wrappers;

/**
 *
 * @author sinantie
 */
public class AmrSentence
{
    protected static int count = 1;
    protected String id, sentence, anonymizedSentence;
    protected final int incrId;
    protected Amr amr;   
    
    public AmrSentence(String id, String sentence)
    {
        this.incrId = count++;
        this.id = id;
        this.sentence = sentence;        
    }

    public void parseAmr(String id, String rawAmr, Dictionaries dictionaries)
    {
        this.amr = new Amr(id, rawAmr, dictionaries);  
        amr.convert();
    }
   
    public String getId()
    {
        return id;
    }

    public String getSentence()
    {
        return sentence.toLowerCase();
    }    

    public int getSentenceSize()
    {
        return sentence.split(" ").length;
    }       
    
    public void setSentence(String sentence)
    {
        this.sentence = sentence;
    }
    
    public String getAnonymizedSentence() 
    {
        return anonymizedSentence;
    }

    public int getAnonymizedSentenceSize()
    {
        return anonymizedSentence.split(" ").length;
    }
    
    public void setAnonymizedSentence(String anonymizedSentence) 
    {
        this.anonymizedSentence = anonymizedSentence;
    }
    
    public Amr getAmr()
    {
        return amr;
    }

    public String rawIdsToString()
    {
        String raw = amr.raw.replaceAll("\\s+", " ").trim(); // get rid of whitespaces
        raw = raw.replaceAll("~e[.][0-9]+,?[0-9]*", ""); // get rid of alignments
        String ar[] = raw.split(" "); // get rid of wiki the lazy way
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < ar.length; i++)
        {
            if (ar[i].equals(":wiki")) {
                i += 1;
            } else {
                str.append(ar[i]).append(" ");
            }
        }
        raw = str.toString().trim();        
        return id + "\t" + raw;
    }
       
}
