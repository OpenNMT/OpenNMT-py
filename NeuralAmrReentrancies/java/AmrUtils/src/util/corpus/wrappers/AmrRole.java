package util.corpus.wrappers;

/**
 *
 * @author sinantie
 */
public class AmrRole extends AmrComponent implements Comparable<AmrRole>
{
    
    private int counts;
    private int orderId; // unique id for a particular role; useful for ordering children of concepts in a specific order
    
    public AmrRole(String name)
    {
        super(name);
        this.counts = Integer.MAX_VALUE;
    }

    public AmrRole(String name, AmrAlignment alignment)
    {
        super(name, alignment);
        this.counts = Integer.MAX_VALUE;
    }

        
    public AmrRole(String name, int counts)
    {
        super(null);
        int index = name.indexOf(":");
        this.name = index >= 0 ? name.substring(name.indexOf(":") + 1) : name;        
        this.counts = counts;
    }

    @Override
    public String getNodeId() 
    {
        return "";
    }     

    public int getOrderId() 
    {
        return orderId;
    }

    public void setOrderId(int orderId) 
    {
        this.orderId = orderId;
    }

    public int getCounts()
    {
        return counts;
    }

    public void addCounts(int count)
    {
        counts += count;
    }
    
    @Override
    public String toString()
    {
        return name;// + (alignment != null ? alignment : "");
    }

    @Override
    public int compareTo(AmrRole o)
    {
        return name.compareTo(o.name);
    }

    @Override
    public boolean equals(Object obj) {
        assert obj instanceof AmrRole;
        return this.name.equals(((AmrRole)obj).name);
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 17 * hash + this.name.hashCode();
        return hash;
    }
    
    
    
}
