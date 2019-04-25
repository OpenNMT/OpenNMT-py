package util.apps;

import java.io.IOException;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import util.server.NamedEntityRecognizerClient;
import util.server.NerServer;

/**
 *
 * @author ikonstas
 */
public class AmrUtilsTest {

    public static class RunServer extends Thread {

        @Override
        public void run() {
            try {
                NerServer.main(new String[]{});
            } catch (IOException e) {
                e.printStackTrace(System.err);
            }
        }
    }

    public AmrUtilsTest() {
    }

    @BeforeClass
    public static void setUpClass() {
        try {
            // start NER server
            RunServer nerServer = new RunServer();
            nerServer.start();
            // wait for server to initialize
            Thread.sleep(4000l);
        } catch (InterruptedException ex) {
            ex.printStackTrace(System.err);
        }
    }

    @AfterClass
    public static void tearDownClass() {
        try {
            // terminate NER server
            NamedEntityRecognizerClient client = new NamedEntityRecognizerClient(4444);
            client.processToString("terminate_server");
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
        }
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {

    }

    /**
     * Test of main method, of class AmrUtils.
     */
//    @Test
    public void testAnonymizeFull() {
        System.out.println("testAnonymizeFull");
        String input = "(h / hold-04 :ARG0 (p2 / person :ARG0-of "
                + "(h2 / have-org-role-91 :ARG1 (c2 / country :name (n3 / name :op1 \"United\" :op2 \"States\")) "
                + ":ARG2 (o / official)))  :ARG1 (m / meet-03 :ARG0 (p / person  "
                + ":ARG1-of (e / expert-01) :ARG2-of (g / group-01))) "
                + ":time (d2 / date-entity :year 2002 :month 1) :location (c / city  :name (n / name :op1 \"New\" :op2 \"York\")))";
        String[] args = {"anonymizeAmrFull", "false", input};
        AmrUtils.main(args);
    }

//    @Test
    public void testAnonymizeStripped() {
        System.out.println("testAnonymizeStripped");
        String input = "help :arg0 (person :name \"Mr. T\") :arg1 ( save :arg0 world ) :time ( date-entity :year 2016 :month 3 :day 4)";
        String[] args = {"anonymizeAmrStripped", "false", input};
        AmrUtils.main(args);
    }

//    @Test
    public void testNerAnonymize() {
        System.out.println("testNerAnonymize");
        String sent = "'Unprecedented' violence sweeps Pakistan in first quarter: CRSS";
        String[] args = {"anonymizeText", "false", sent};
        AmrUtils.main(args);
    }
   
//    @Test
    public void testDeAnonymize() {
        System.out.println("testDeAnonymize");
        String sent = "person_name_0 helped save num_0 cats in day_date-entity_0 month_name_date-entity_0 year_date-entity_0 .";
        String alignments = "person_name_0|||name_John_Pappas\tnum_0|||3\tyear_date-entity_0|||2016\tmonth_date-entity_0|||3\tday_date-entity_0|||4";
        String[] args = {"deAnonymizeText", "false", sent + "#" + alignments};
        AmrUtils.main(args);
    }

//    @Test
    public void testAmrDeAnonymize() {
        System.out.println("testAmrDeAnonymize");
        String amr = "and :op1 ( be-located-at-91 :arg1 country_name_0 :arg2 ( group :mod country_name_1 ) ) :op2 ( play-08 :arg0 country :arg1 ( and :op1 country_name_1 :op2 country_name_2 :op3 country_name_3 ) :time       ( date-entity month_date-entity_0 day_date-entity_0 ) :mod respective )";
        String alignments = "country_name_0|||Tajikistan\tcountry_name_2|||Latvia\tcountry_name_3|||Belarus \tday_date-entity_0|||18\tday_date-entity_0|||19\tcountry_name_1|||Estonia\tmonth_name_date-entity_0|||1\tnum_0|||21";
        String[] args = {"deAnonymizeAmr", "false", amr + "#" + alignments};
        AmrUtils.main(args);
    }
    
//    @Test
    public void testAnonymizeFullFileAmr() {
        System.out.println("testAnonymizeFullFileAmr");
        String input = "resources/sample-data/sample-amr.txt";
        String[] args = {"anonymizeAmrFull", "true", input};
        AmrUtils.main(args);
    }
    
//    @Test
    public void testAnonymizeFileText() {
        System.out.println("testAnonymizeFileText");
        String input = "resources/sample-data/sample-nl.txt";
//        String input = "/Users/ikonstas/Desktop/bug.txt";

        String[] args = {"anonymizeText", "true", input};
        AmrUtils.main(args);
    }

    @Test
    public void testDeAnonymizeAmrFile() {
        System.out.println("testDeAnonymizeAmrFile");
//        String input = "resources/sample-data/sample-amr.txt";
        String input = "/Users/ikonstas/Desktop/cmu-summer-school/ori.txt";

        String[] args = {"deAnonymizeAmr", "true", input};
        AmrUtils.main(args);
    }

//    @Test
    public void testDeAnonymizeFullTextFile() {
        System.out.println("testDeAnonymizeFullTextFile");
        String input = "resources/sample-data/sample-nl.txt";
        String[] args = {"deAnonymizeText", "true", input};
        AmrUtils.main(args);
    }

}
