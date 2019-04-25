package util.server;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import util.utils.NamedEntityRecognizerInterface;

/**
 *
 * @author ikonstas
 */
public class NamedEntityRecognizerClient implements NamedEntityRecognizerInterface {

    private Socket socket;
    private BufferedReader in;
    private PrintWriter out;
    private final String host = "localhost";    
    
    public NamedEntityRecognizerClient(int port) throws IOException {                
        socket = null;
        out = null;
        in = null;
        
        try {
            socket = new Socket(host, port);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        } catch (UnknownHostException e) {
            System.err.println("Don't know about host: " + host);
            System.exit(1);
        } catch (IOException e) {
            System.err.println("Couldn't get I/O for the connection to: " + host);
            System.exit(1);
        }

    }
    
    @Override
    public String processToString(String text) {
        
        try {              
            out.write(text + "\n");
            out.flush();
            return in.readLine();
        } catch (IOException ex) {
            System.err.println("Couldn't write/read from socket.");
            ex.printStackTrace(System.err);
        }
        return null;
    }
    
    @Override
    public String processToString(String text, int length) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public void destroy() {
        try {            
            out.close();
            in.close();
            socket.close();
        } catch (IOException ex) {
            System.err.println("Error closing socket");
        }
    }
    
    public static void main(String[] args) {
        try {
            NamedEntityRecognizerClient nerc = new NamedEntityRecognizerClient(4444);
            try(BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
                String line;
                System.out.println("Enter string to NER [q to exit]");
                while(!(line = reader.readLine()).equals("q")) {
                    System.out.println(nerc.processToString(line));
                }
                nerc.processToString("terminate_client");
            } catch (IOException ioe) {
                ioe.printStackTrace(System.err);
            }
            nerc.destroy();
        } catch(IOException ioe) {
            ioe.printStackTrace(System.err);
        }        
    }
    
}
