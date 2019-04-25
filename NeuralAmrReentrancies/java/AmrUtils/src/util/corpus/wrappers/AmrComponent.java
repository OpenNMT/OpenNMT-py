package util.corpus.wrappers;

/**
 *
 * @author ikonstas
 */


public abstract class AmrComponent
{
        
    protected String name;
    protected String nodeId;
    protected AmrAlignment alignment;    
    
    public AmrComponent(String name, AmrAlignment alignment)
    {
        this.name = name;
        this.alignment = alignment;
    }

    public AmrComponent(String name)
    {
        this(name, null);
    }

    public AmrAlignment getAlignment()
    {
        return alignment;
    }
        
    
    public String getName()
    {
        return name;
    }

    public void setName(String name)
    {
        this.name = name;
    }

    public String getNodeId() 
    {
        return nodeId;
    }    
}
