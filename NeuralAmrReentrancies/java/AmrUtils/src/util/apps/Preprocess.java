package util.apps;

/**
 *
 * @author Ioannis Konstas
 */
public class Preprocess extends Application
{
    public Preprocess()
    {
        super();
    }

    @Override
    public void execute()
    {        
        corpus.preprocess();
    }

    public static void main(String[] args)
    {
        Preprocess pc = new Preprocess();
        pc.execute();
    }
}
