package util.server;

import java.io.IOException;
import java.net.ServerSocket;
import util.utils.NamedEntityRecognizer;

/**
 *
 * @author sinantie
 */
public class NerServer {

    protected int port;
    protected NamedEntityRecognizer ner;
    protected boolean verbose;

    public NerServer(int port, boolean verbose) {
        this.port = port;
        this.verbose = verbose;
        this.ner = new NamedEntityRecognizer(true);
    }

    public void execute() {

        ServerSocket serverSocket = null;
        boolean listening = true;
        try {            
            serverSocket = new ServerSocket(port);
            System.out.println("NER Server listening on port " + port);
        } catch (IOException e) {
            error("Could not listen on port: " + port);            
        }
        try {
            while (listening) {
                Object res = new NerThread(ner, serverSocket.accept(), verbose).call();
                if (res instanceof String && res.equals("terminate_server")) {
                    listening = false;
                }
            }
        } catch (IOException ioe) {
            message("Could not establish connection!");
        }
        try {
            serverSocket.close();
        } catch (IOException ioe) {
            error("Error closing socket");
        }        

    }

    public static void main(String[] args) throws IOException {
        int port = args.length > 0 && args[0] != null ? Integer.valueOf(args[0]) : 4444;
        boolean verbose = args.length > 1 && args[1] != null ? Boolean.valueOf(args[1]) : true;
        NerServer server = new NerServer(port, verbose);
        server.execute();
    }

    public static void error(String msg) {
        System.err.println(msg);
        System.exit(-1);
    }

    public static void message(String msg) {
        System.out.println(msg);
    }
}
