package util.corpus.wrappers;

import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.TreeMap;

/**
 *
 * @author sinantie, bharat
 */
public class Amr
{
    
    protected final String id;
    protected final String raw;
    protected final Dictionaries dictionaries;
    protected final AmrNode graph;
    private final Map<String, String> varStrMap;
    protected boolean failedToParse = false;
    
    public Amr(String id, String raw, Dictionaries dictionaries)
    {
        this.id = id;
        this.raw = raw;
        this.dictionaries = dictionaries;
        varStrMap = new HashMap<>();
        populateVarMap(raw, varStrMap);
        graph = parse(raw);        
//        convert();
    }
  
    public Amr(String id, AmrNode amr, Dictionaries dictionaries) 
    {
        this.id = id;
        this.raw = null;
        this.graph = amr;
        this.dictionaries = dictionaries;
        varStrMap = new HashMap<>();
    }        
    
    protected void convert() {
        // implemented in sub-class
    }
    
    public boolean isEmpty()
    {
        // implemented in sub-class
        return false;
    }
    
    private void populateVarMap(String rawStr, Map<String, String> varStrMap)
    {
        rawStr = rawStr.replaceAll("\n", " ").replaceAll(" +", " ");
        // Treating opening and closing brackets as separate tokens and adding a dummy ":TOP" argument.
        rawStr = rawStr.replaceAll("\\(", "( ").replaceAll("\\)", " )");
        rawStr = ":TOP "+rawStr;
        String[] parts = rawStr.split(" ");
        for (int i=0; i<parts.length; i++) {
            String cur = parts[i];
            
            //Update the map with new variables
            if(cur.equals("/")){
                varStrMap.put(parts[i-1], stripAlignment(parts[i+1]));
            }
        }
    }
    
    private AmrNode parse(String rawStr) {
        rawStr = rawStr.replaceAll("\n", " ").replaceAll(" +", " ");
        rawStr = treatParsInConstants(rawStr);
        //Variable to word Map eg: m2 -> many        
        Stack<AmrNode> children = new Stack<>();
        Stack<String> cStack = new Stack<>();
        
        // Treating opening and closing brackets as separate tokens and adding a dummy ":TOP" argument.
        rawStr = rawStr.replaceAll("\\(", "( ").replaceAll("\\)", " )").replaceAll(" +", " ");
        rawStr = ":TOP "+rawStr;
        String[] parts = rawStr.split(" ");

        for (int i=0; i<parts.length; i++) {
            String cur = parts[i];
            
            //Update the map with new variables
            if(cur.equals("/")){
                varStrMap.put(parts[i-1], stripAlignment(parts[i+1]));
                cStack.add(cur);
            }
            else if(cur.charAt(0) == ':'){
                AmrAlignment argAlignment = new AmrAlignment(AmrAlignment.TokenType.ROLE, cur);
                /* If argument node doesn't include a word then it can be
                    a) a variable. eg: :ARG0 m2
                    b) string eg: :ARG0 "India"
                    c) constants eg: :quant 2, :polarity -                    
                */
                if(!parts[i+1].equals("(")){  
                    AmrAlignment conceptAlignment = new AmrAlignment(AmrAlignment.TokenType.CONCEPT, parts[i+1]);
                    String var, wrd = conceptAlignment.getToken();
                    // found a cyclic reference - a variable that has been reused - update with the original concept name
                    if( (var = varStrMap.get(wrd)) != null)
                    {
                        wrd = var;
                        children.add(new AmrNode((var != null) ? stripAlignment(parts[i+1]) : "", wrd, "", argAlignment.getToken(), 
                                true, new AmrAlignment(AmrAlignment.TokenType.CONCEPT, parts[i+1]), argAlignment));
                    }
                    else
                    {
                        children.add(new AmrNode((var != null) ? stripAlignment(parts[i+1]) : "", conceptAlignment.getToken(), "", argAlignment.getToken(), 
                                conceptAlignment, argAlignment));
                    }                    
                    i++;
                }
                else{
                    cStack.add(cur);
                }
            }
            else if(cur.charAt(0) == '('){
                cStack.add(cur);
                children.push(new AmrNode("(", "(", "(", "("));
            }
            else if(cur.charAt(0) == ')'){
                try{
                    List<String> argList = new ArrayList();
                    String cPop = cStack.pop();
                    while (cPop.charAt(0) != '(') {
                        argList.add(0, cPop);
                        cPop = cStack.pop();
                    }                
                    AmrAlignment argAlignment = new AmrAlignment(AmrAlignment.TokenType.ROLE, cStack.pop());
                    String arg = argAlignment.getToken();
    //                String arg = cStack.pop();

                    /* If arument node includes a word eg: :ARG0 (m2 / many)
                        a) Get the sense variable.
                        b) Create the node
                        c) Add current node's arguments as its children
                    */
                    if(argList.size() == 3){
                        AmrAlignment conceptAlignment = new AmrAlignment(AmrAlignment.TokenType.CONCEPT, argList.get(2));
                        String sense = "", wrd = conceptAlignment.getToken();
    //                    String sense = "", wrd = argList.get(2);

                        //Get the sense variable.
                        int sind = wrd.lastIndexOf('-');                    
                        if(sind != -1){
                            sense = wrd.substring(sind+1);
                            try{
                                Integer.parseInt(sense);
                                wrd = wrd.substring(0, sind);
                            }
                            catch(NumberFormatException e){
                                sense = "";
                            }
                        }

                        // Create the node
                        try{
                        AmrNode argNode = new AmrNode(stripAlignment(argList.get(0)), wrd, sense, arg, conceptAlignment, argAlignment);

                        // Add current node's arguments as its children
                        AmrNode child = children.pop();
                        while (!child.getWord().equals("(")){                        
                            argNode.addArg(child);
                            child = children.pop();
                        }
                        children.push(argNode);
                        }
                        catch(Exception e)
                        {
                            System.out.println(conceptAlignment + " " + argAlignment);
                        }
                    }
                    else{
                        failedToParse = true;
                        System.err.println("Program should never enter here !\n" + id + "\n" + raw);
                    }
                }catch(EmptyStackException e) {
                    System.err.println("Stack empty while reading raw input (empty or multiple graphs in one example):\n" + id + "\n" + raw);
                    return children.get(0); 
                }
            }
            else{
                cStack.add(cur);
            }
        }
        return children.get(0);        
    }
    
    /**
     * 
     * remove alignment info from concept/role. The format of the string is e.g., establish-01~e.0
     * @param str
     * @return 
     */
    protected String stripAlignment(String str)
    {
        int index = str.indexOf("~");
        if(index >= 0)
        {
            return str.substring(0, index);
        }
        return str;
    }
    
    /**
     * 
     * Replace parentheses in string literals (such as wiki entries), with brackets (to avoid parsing implications)
     * e.g., Rod_Stewart_(singer) becomes Rod_Stewart_[singer]
     * @param rawStr
     * @return 
     */
    private String treatParsInConstants(String rawStr)
    {
        if(rawStr.contains("\""))
        {
            String[] ar = rawStr.split("\"");
            StringBuilder str = new StringBuilder(ar[0]);
            for(int i = 1; i < ar.length - 1; i++)
            {
                str.append("\"").
                    append(ar[i].replaceAll("\\(", "[").replaceAll("\\)", "]")).
                    append("\"").
                    append(ar[++i]);
            }
            return str.toString();
        }
        return rawStr;
    }        
    
    private void appendNewAmrNode(AmrNode root, AmrNode candChild, String relation)
    {
        AmrNode newChild = new AmrNode(candChild.getVar(), candChild.getWord(), candChild.getSense(), relation);
        root.addArg(newChild);
    }
    
    private String canonicaliseRole(String role)
    {
        int index = role.indexOf("-of");
        return index > 0 ? role.substring(0, index) : role;
        
    }
    
    protected boolean isNameConcept(AmrNode node)
    {
        return node.getWord().equals("name") && node.getRole().equals("name");
    }
    
    protected boolean isQuantConceptWithNum(AmrNode node)
    {
        return node.getRole().equals("quant") && isNumber(node.getWord());
    }
    
    protected boolean isValueConceptWithNum(AmrNode node)
    {
        return node.getRole().equals("value") && isNumber(node.getWord());
    }
    
    protected boolean isDateEntityConcept(AmrNode node)
    {
        return node.getWord().equals("date-entity");
    }
    
    protected boolean hasDateEntityRole(AmrNode node)
    {
        return dictionaries.getAmrDateRoles().contains(node.getRole());
    }
    
    protected boolean hasTimeRole(AmrNode node)
    {
        return node.getRole().equals("time");
    }
    
    protected AmrConcept.Type determineConceptType(AmrNode node)
    {
        if(node.hasSense())
            return AmrConcept.Type.OTHER;
        if(dictionaries.getAmrNeConcepts().contains(node.getWord()))
            return AmrConcept.Type.NAME;
        if(dictionaries.getAmrQuantityConcepts().contains(node.getWord()))
            return AmrConcept.Type.QUANTITY;        
        if(dictionaries.getAmrValueConcepts().contains(node.getWord()))
            return AmrConcept.Type.VALUE;
        if(isDateEntityConcept(node))
            return AmrConcept.Type.DATE;
        if(isNumber(node.getWord()))
            return AmrConcept.Type.NUMBER;
        return AmrConcept.Type.OTHER;
    }
    
    protected boolean isNumber(String s)
    {
        return s.matches("-\\p{Digit}+|" + // negative numbers
                             "-?\\p{Digit}+[\\.,]\\p{Digit}+") || // decimals
                             (s.matches("\\p{Digit}+"));
    }
    
    private String aggregateNameConcept(AmrNode node)
    {
        Enumeration<AmrNode> children = node.children();
        Map<Integer, String> opsOrdered = new TreeMap<>();
        while(children.hasMoreElements())
        {
            AmrNode child = children.nextElement();
            String role = child.getRole();
            if(role.startsWith("op"))
            {
                opsOrdered.put(Integer.valueOf(role.substring(2)), child.getWord());
            }
        }
        StringBuilder str = new StringBuilder();
        opsOrdered.values().stream().forEach((value) -> { str.append(value.replaceAll("\"", "")).append(" "); });
        return str.toString().trim();
    }            

    public boolean isFailedToParse() {
        return failedToParse;
    }

    public AmrNode getGraph() {
        return graph;
    }
        
    
    @Override
    public String toString()
    {        
        //return raw.replaceAll("\n", " ").replaceAll(" +", " ");
        return raw.replaceAll("\n", " ").replaceAll(" +", " ").replaceAll("-[0-9]+", "");
    }   
    
    
}
