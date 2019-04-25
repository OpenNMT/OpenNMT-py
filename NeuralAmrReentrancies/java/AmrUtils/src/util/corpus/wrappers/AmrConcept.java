package util.corpus.wrappers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 *
 * @author sinantie
 */
public class AmrConcept extends AmrComponent
{
      
    protected static enum Type {NAME, DATE, TIME, QUANTITY, VALUE, NUMBER, ORDINAL, MONEY, OTHER}
    protected Type type;
    private String sense;
    private String var;
    private boolean cyclicReference;
    private AmrConcept parent;
    private List<AmrProposition> children;   
            
    public AmrConcept(String name, String sense, String var, boolean cyclicReference, Type type, AmrAlignment alignment, String nodeId)
    {
        super(name, alignment);
        this.sense = sense;
        this.var = var;
        this.type = type;
        this.cyclicReference = cyclicReference;
        this.nodeId = nodeId;
        this.children = new ArrayList<>();
    }

    public AmrConcept(String var, String name, Type type, AmrAlignment alignment, String nodeId)
    {
        this(name, null, var, false, type, alignment, nodeId);        
    }
    
    public AmrConcept(String name, Type type, AmrAlignment alignment, String nodeId)
    {
        this(null, name, type, alignment, nodeId);
    }       
    
    public AmrConcept(String name, Type type, AmrAlignment alignment)
    {
        this(name, type, alignment, null);        
    }
                
    public AmrConcept(String name, String sense)
    {
        this(name, sense, null, false, null, null, null);        
    }

    public AmrConcept(String name)
    {
        this(name, null);        
    }
    
    public Type getType()
    {
        return type;
    }

    public void setType(Type type) 
    {
        this.type = type;
    }
        
    public String getSense()
    {
        return sense;
    }

    public String getVar() 
    {
        return var;
    }    
    
    public boolean hasSense()
    {
        return sense != null && !sense.equals("");
    }

    public boolean isCyclicReference()
    {
        return cyclicReference;
    }

    public void addChild(AmrProposition prop) 
    {
        children.add(prop);
    }

    public List<AmrProposition> getChildren() {
        return children;
    }

    public boolean containsChildWithRoleName(String name) {
        return children.stream().anyMatch(p -> p.roleNameEquals(name));        
    }
    
    public void removeChildrenWithRoleName(String name) {
        Iterator<AmrProposition> it = children.iterator();
        while(it.hasNext()) {
            if(it.next().roleNameEquals(":wiki")) {
                it.remove();
            }
        }        
    }
    
    public void setParent(AmrConcept parent) 
    {
        this.parent = parent;
    }

    public AmrConcept getParent() 
    {
        return parent;
    }
    
    @Override
    public String toString()
    {
        return (hasSense() ? name + "-" + sense : name);// + (alignment != null ? alignment : "");
    }
    
    public static AmrConcept newEmptyConcept()
    {
        return new AmrConcept("--");
    }
    
    public static StringBuilder print(AmrConcept node, String role, StringBuilder str, boolean outputBrackets, boolean reshuffleChildren, 
            boolean markLeaves, boolean outputSense, boolean concatBracketsWithRoles) {
        if(!node.getName().equals("")) {
            // don't output brackets for nodes with one or less children
            str.append(outputBrackets && node.getChildren().size() > 0 ? (concatBracketsWithRoles ? (role + "( ") : " ( ") : " ").append(outputSense ? node : node.getName());
            str.append(outputBrackets && node.getChildren().isEmpty() && markLeaves ? " *" : "");
        } 
        List<AmrProposition> childrenSet = node.getChildren();
        if(reshuffleChildren) 
            Collections.sort(childrenSet, (AmrProposition p1, AmrProposition p2) -> p1.getRole().getOrderId() < p1.getRole().getOrderId() ? 1 : -1);
        for(AmrProposition child : childrenSet) {
            String childRole = child.getRole().toString();
            if(!childRole.equals(":unk")) {                
                str.append(" ").append(!concatBracketsWithRoles ? childRole : (child.getArgument().getChildren().isEmpty() ? childRole : ""));                
            }
            print(child.getArgument(), childRole, str, outputBrackets, reshuffleChildren, markLeaves, outputSense, concatBracketsWithRoles);
        }
        return str.append(outputBrackets && node.getChildren().size() > 0 ? (concatBracketsWithRoles ? (" )" + role) : " ) ") : "");
    }
}
