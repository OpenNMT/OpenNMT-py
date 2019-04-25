package util.server;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.concurrent.Callable;
import util.utils.NamedEntityRecognizer;

/**
 *
 * @author ikonstas
 */
public class NerThread implements Callable {

    private Socket socket = null;
    private final NamedEntityRecognizer ner;
    private final String client;
    private final boolean verbose;

    public NerThread(NamedEntityRecognizer ner, Socket socket, boolean verbose) {
        this.ner = ner;
        this.verbose = verbose;
        this.socket = socket;
        this.client = socket.getRemoteSocketAddress().toString();
        if (verbose) {
            NerServer.message("Established connection with client " + client);
        }
    }

    @Override
    public Object call() {
        String input = null;
        try {
            try (PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
                    BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
                if (verbose) {
                    NerServer.message("Looks like the socket is" + (socket.isClosed() ? " not" : " ") + "open...");
                }
                while (true) {
                    if (verbose) {
                        NerServer.message("Waiting for some input from client...");
                    }

                    input = in.readLine();                    
                    String result = ner.processToString(input) + "\n";
                    if (verbose) {
                        NerServer.message("Just predicted: " + result + " . Writing output to socket...");
                    }
                    out.write(result);
                    out.flush();
                    if (input == null || input.equals("terminate_server") || input.equals("terminate_client")) {
                        break;
                    }
                } // while
            } catch (Exception e) {
                NerServer.error(e.getMessage());
            }

        } finally {
            try {
                socket.close();
                if (verbose) {
                    NerServer.message("Closed connection with client " + client);
                }
            } catch (IOException ex) {
                NerServer.error(ex.getMessage());
            }
        }
        return input != null && input.equals("terminate_server") ? "terminate_server" : null;
    }
}
