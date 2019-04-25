package util.apps;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import util.utils.Database;
import util.utils.Settings;
import util.utils.Utils;
import util.corpus.Corpus;

/**
 *
 * @author Ioannis Konstas
 */
public abstract class Application
{
    protected Database db;
    protected Settings settings;
    protected String corpusString;
    protected Corpus corpus;

    public Application()
    {
        settings = new Settings();
        db = new Database(settings);
        corpusString = settings.getProperty("corpus");
        init();
    }

    private void init()
    {
        try
        {
            Constructor cons = Class.forName("util.corpus." +
                               Utils.toCamelCasing(corpusString) + "Corpus").
                               getConstructor(Settings.class, Database.class,
                                              String.class);            
            corpus = (Corpus) cons.newInstance(settings, db, corpusString);            

        }
        catch (ClassNotFoundException | NoSuchMethodException | SecurityException | 
                InstantiationException | IllegalAccessException | 
                IllegalArgumentException | InvocationTargetException ex)
        {            
            System.err.println(ex.getCause());
            ex.printStackTrace(System.err);
        }
    }

    public abstract void execute();
}
