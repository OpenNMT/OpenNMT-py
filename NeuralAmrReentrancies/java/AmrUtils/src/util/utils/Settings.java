package util.utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 *
 * @author Yannis Konstas
 */
public class Settings 
{
    Properties properties;
//    private static final Logger logger = Logger.getLogger(Settings.class);
    /**
     * Reads the properties file which contains various settings for the 
     * connection with last.fm and the database
     */
    public Settings()
    {
        properties = new Properties();
        try 
        {
            properties.load(new FileInputStream("settings.properties"));   
//            logger.info("Opened properties file");
        } 
        catch (IOException e) 
        {
            System.err.println("Unable to read properties file");
            System.exit(1);
        }
    }
    
    /**
     * Returns the value of a certain property 
     * @param key the property's key 
     * @return the value of the requested key
     */
    public String getProperty(String key)
    {
        String s = properties.getProperty(key);
        return s == null ? null : s.trim();
    }

    /**
     * Returns the value of a certain corpus' specific property
     * @param corpus
     * @param key the property's key
     * @return the value of the requested key
     */
    public String getProperty(String corpus, String key)
    {
        return properties.getProperty(corpus + "." + key).trim();
    }
}
