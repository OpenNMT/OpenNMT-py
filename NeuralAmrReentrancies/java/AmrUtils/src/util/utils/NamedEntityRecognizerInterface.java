package util.utils;

/**
 *
 * @author ikonstas
 */


public interface NamedEntityRecognizerInterface {
    
    public String processToString(String text, int length);
    
    public String processToString(String text);
}
