package util.corpus.wrappers;

import java.util.Objects;

/**
 *
 * @author ikonstas
 */
public class AmrProposition implements Comparable<AmrProposition>
{

    private AmrConcept predicate, argument;
    private AmrRole role;

    public AmrProposition() 
    {        
    }

    
    public AmrProposition(AmrConcept predicate, AmrRole role) 
    {
        this.predicate = predicate;        
        this.role = role;
    }
    
    public AmrProposition(AmrConcept predicate, AmrConcept argument) 
    {
        this.predicate = predicate;        
        this.argument = argument;
    }
    
    public AmrProposition(AmrConcept predicate, AmrConcept argument, AmrRole role) 
    {
        this.predicate = predicate;
        this.argument = argument;
        this.role = role;
    }

    /**
     * 
     * 
     * Comma-delimited input: predicate,argument,role
     * @param raw 
     */
    public AmrProposition(String raw) {
       String ar[] = raw.split(",");
       if(ar.length == 3) {
           predicate = new AmrConcept(ar[0]);
           argument = new AmrConcept(ar[1]);
           role = new AmrRole(ar[2]);
       } else {
           predicate = new AmrConcept("N/A");
           argument = new AmrConcept("N/A");
           role = new AmrRole(":unk");
       }       
    }
    
    public void setPredicate(AmrConcept predicate) 
    {
        this.predicate = predicate;
    }

    public AmrConcept getPredicate() 
    {
        return predicate;
    }

    public void setArgument(AmrConcept argument) 
    {
        this.argument = argument;
    }

    public AmrConcept getArgument() 
    {
        return argument;
    }

    public void setRole(AmrRole role) 
    {
        this.role = role;
    }

    public AmrRole getRole() 
    {
        return role;
    }
    
    public boolean roleNameEquals(String name) {
        return role.getName().equals(name);
    }
    
    public boolean isEmpty()
    {
        return predicate == null && argument == null && role == null;
    }
    
    @Override
    public boolean equals(Object obj) 
    {
        assert obj instanceof AmrProposition;
        AmrProposition o = (AmrProposition)obj;
        return predicate.equals(o.predicate) && argument.equals(o.argument) && 
                role.equals(o.role);        
    }

    @Override
    public int hashCode() 
    {
        int hash = 7;
        hash = 17 * hash + Objects.hashCode(this.predicate);
        hash = 17 * hash + Objects.hashCode(this.argument);
        hash = 17 * hash + Objects.hashCode(this.role);
        return hash;
    }
    
    @Override
    public String toString() 
    {
        return String.format("%s,%s,%s", predicate.name, argument.name, role == null ? ":unk" : role.name);
    }

    @Override
    public int compareTo(AmrProposition o) {
        return 1;//this.role.compareTo(o.role);
    }
}
