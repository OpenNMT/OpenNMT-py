package util.corpus.wrappers;

import javax.swing.tree.DefaultMutableTreeNode;
/**
 *
 * @author bharat
 */
public class AmrNode extends DefaultMutableTreeNode
{
    //Var is variable, wrd is Word, sense is sense id and argLabel is argument label for the current node.
    final String var;
    private final String word;
    private final String sense;
    private String role;
    private AmrAlignment conceptAlignment;
    private AmrAlignment roleAlignment;
    private final boolean cyclicReference;
    private final boolean constant;
   
    
    AmrNode(String variable, String word, String sensestr, String label, boolean cyclicReference, 
            AmrAlignment conceptAlignment, AmrAlignment roleAlignment)
    {
        var = variable;        
        String[] wordSense = stripSense(word);
        if(wordSense.length == 1) {
            this.word = word;
            sense = sensestr;
        } else {
            this.word = wordSense[0];
            this.sense = wordSense[1];
        }       
        int index = label.indexOf(":");
        role = index >= 0 ? label.substring(index + 1) : label;
        this.cyclicReference = cyclicReference;
        this.conceptAlignment = conceptAlignment;
        this.roleAlignment = roleAlignment;
        this.constant = parseConstant(this.word);
    }
    
    public AmrNode(String variable, String word, String sensestr, String label, AmrAlignment conceptAlignment, AmrAlignment roleAlignment)
    {
        this(variable, word, sensestr, label, false, conceptAlignment, roleAlignment);
    }
    
    public AmrNode(String variable, String word, String sensestr, String label)
    {
        this(variable, word, sensestr, label, false, null, null);        
    }

    /**
     * Strips the sense from the input string token if it exists. 
     * The input token format is xx-[0-9]{2}.
     * @param tok
     * @return 
     */
    private String[] stripSense(String tok) {
        int id = tok.lastIndexOf("-");
        if(id != -1) {            
            String candSense = tok.substring(id + 1);
            if(candSense.matches("[0-9]{2}")) {
                return new String[] {tok.substring(0, id), candSense};
            }
        }
        return new String[] {tok};
    }
    
    private boolean parseConstant(String word) {
        return word.startsWith("\"") || word.matches("[0-9]+[,.]?[0-9]*") || word.equals("-");
    }

    public boolean isConstant() {
        return constant;
    }
    
    public String getRole()
    {
        
        return role;
    }

    public void setRole(String role)
    {
        this.role = role;
    }

    public String getVar()
    {
        return var;
    }

    public String getSense()
    {
        return sense;
    }

    public boolean hasSense()
    {
        return !sense.equals("");
    }

    public boolean hasInverseRelation()
    {
        return role != null && role.endsWith("-of");
    }
        
    public String getWord()
    {        
        
        return word;
    }
    
    public AmrAlignment getConceptAlignment()
    {
        return conceptAlignment;
    }

    public AmrAlignment getRoleAlignment()
    {
        return roleAlignment;
    }

    public boolean isCyclicReference()
    {
        return cyclicReference;
    }

    public void addArg(AmrNode arg)
    {
        this.insert(arg, 0);
    }

    //Pretty print which prints the node in the similar format as AMR tree.
    public String pp()
    {
        StringBuilder sb = new StringBuilder();
        if (!role.equals("TOP"))
        {
            sb.append(" :");
            sb.append(role);
        }
        if (!(var.isEmpty() || isConstant()))
        {
            sb.append(" (");
            sb.append(var);
            sb.append(" / ");
        } else
        {
            sb.append(" ");
        }
        sb.append(word);
        if (!sense.isEmpty())
        {
            sb.append("-");
            sb.append(sense);
        }
        for (int i = 0; i < this.getChildCount(); i++)
        {
            AmrNode child = (AmrNode) this.getChildAt(i);
            sb.append(child.pp());
        }
        if (!(var.isEmpty() || isConstant()))
        {
            sb.append(")");
        }
        return sb.toString();
    }

    public String toStringNoChildren()
    {        
        return String.format(":%s ( %s / %s-%s )", role, var, word, sense);
    }
    
    public String toStringWithChildren()
    {        
        StringBuilder str = new StringBuilder(toStringNoChildren());
        str.append(" [ ");
        for (int i = 0; i < this.getChildCount(); i++)
        {
            AmrNode child = (AmrNode) this.getChildAt(i);
            str.append(child.toStringNoChildren()).append(", ");
        }
        str.deleteCharAt(str.length() - 2).append("]");
        return str.toString();
    }
    
    @Override
    public String toString()
    {
        return toStringNoChildren();
//        StringBuilder sb = new StringBuilder();
//        sb.append(" ");
//        sb.append(role);
//        sb.append(" (");
//        sb.append(var);
//        sb.append(" / ");
//        sb.append(word);
//        sb.append("-");
//        sb.append(sense);
//        for (int i = 0; i < this.getChildCount(); i++)
//        {
//            AmrNode child = (AmrNode) this.getChildAt(i);
//            sb.append(child.toString());
//        }
//        sb.append(")");
//        return sb.toString();
    }
    
}
