package util.utils;

/**
 *
 * @author Yannis Konstas
 */
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;


/**
 * A simple wrapper for the usual SQL and update queries for the chosen database
 * as directed in the settings.properties file
 * @author sinantie
 */
public class Database 
{
        
    private Connection conn;
    private final Settings settings;
    /**
     * Creates a connection with the database
     * @param settings
     */
    public Database(Settings settings)
    {
        this.settings = settings;
        conn = getConnection();
    }
    
    /**
     * Ceates a proper connection with the existing dbms, depending on the 
     * values contained in the settings
     * @return a connection to the database
     */
    private Connection getConnection()
    {
        String corpus = settings.getProperty("corpus");
        String dbms = settings.getProperty("dbms");    
        if(dbms.equals("none"))
            return null;
        String driver = settings.getProperty(dbms + ".driver");
        try
        {
            Class.forName(driver);
        }
        catch (ClassNotFoundException e) 
        {
            System.err.println("Unable to find and load driver");
            System.exit(1);
        }
        
//        String host = settings.getProperty(dbms + ".host");
        String database = settings.getProperty(corpus + "." + dbms + ".database");
//        String user = settings.getProperty(dbms + ".user");
//        String password = settings.getProperty(dbms + ".password");
//        String conString = "jdbc:" + dbms + "://" + host + "/" +
//                           database + "?user=" + user + "&password=" + password;
//        if(database.equals("none"))
//            return null;
        
        String conString = settings.getProperty(String.format("%s.%s.url", corpus, dbms));

        try 
        {
//            String propertiesFile = settings.getProperty(corpus + "." + dbms + ".properties");
//            if(propertiesFile != null)
//            {
//                Properties dbProperties = new Properties();
//                dbProperties.load(new FileReader(propertiesFile));
//                conn = DriverManager.getConnection(conString);
//            }
//            else
                conn = DriverManager.getConnection(conString);

            System.err.println("Established connection with " + database);
        }
        catch(SQLException e) 
        {
            displaySQLErrors(e, "open connection");
        }
//        catch(IOException e)
//        {
//            logger.info("Database Properties file not found!");
//        }
        return conn;
    }
	
    /**
     * Closes the connection with the database
     */
    public void closeConnection()
    {
        try
        {
            conn.close();
        }
        catch (SQLException e)
        {
            displaySQLErrors(e, "close connection");
        }
    }

    /**
     * Displays errors return by the DBMS in a 'nice' format
     * @param e the instance of <code>SQLException</code> class
     * @param sql the string containing the query in SQL 
     */
    public void displaySQLErrors(SQLException e, String sql)	
    {
        String out = sql +
                    "\nSQLException: " + e.getMessage() +
                    "\nSQLState: " + e.getSQLState() +
                    "\nVendorError: " + e.getErrorCode();
        System.err.println(out);
    }

    /**
     * Executes a query to the database
     * @param sql the string containing the query in SQL
     * @return a <code>ResultSet</code> containing the results of the query
     */
    public ResultSet executeQuery(String sql)
    {
        ResultSet rs = null;
        try 
        {			
            // Create a statement object in order to represent 
            // a sql statement in java
            Statement stmt = conn.createStatement();
            // Stores the data in a Resultset
            rs = stmt.executeQuery(sql);
        }
        catch (SQLException e) 
        {
            displaySQLErrors(e, sql);
        }
        return rs;
    }

    /**
     * Executes an update query to the database
     * @param sql the string containig the update query
     * @return the number of rows affected in the database
     */
    public int executeUpdate(String sql)
    {
        int rowcount = 0;
        try 
        {			
            // Create a statement object in order to represent 
            // a sql statement in java
            Statement stmt = conn.createStatement();
            // Stores the data in a Resultset
            rowcount = stmt.executeUpdate(sql);
        }
        catch (SQLException e) 
        {
            displaySQLErrors(e, sql);
            rowcount = -1;
        }
        return rowcount;
    }
}