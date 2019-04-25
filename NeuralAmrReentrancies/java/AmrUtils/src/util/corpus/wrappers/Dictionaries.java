package util.corpus.wrappers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import util.utils.HistMap;
import util.utils.Settings;

/**
 *
 * @author ikonstas
 */


public class Dictionaries {
    
    private Settings settings;
    private final String corpus;
    private final String exportType;
        
    // exportType = linearize (amr)
    private Set<String> linearizedFilter = null;
    private Set<String> amrNeConcepts = null, amrQuantityConcepts = null, 
            amrValueConcepts = null, amrDateRoles = null, mrsFeatures = null;
    private Map<String, String> amrNeClusters = null;
    private Map<AmrRole, Integer> amrRoles;
    
    // exportType = linearize (giga, anonymize)
    private Map<String, String> amrAnonymizationAlignments = null, neNlAmrAlignments = null;
    private Map<String, Integer> amrNlVocabulary = null;
    private int[] numOfNlNeTokens = null;
    private HistMap<String> neTokens = null; 
    
    public Dictionaries(Settings settings, boolean verbose, String corpus, boolean annotationUseNeClusters) {
        this.settings = settings;
        this.corpus = corpus;
        exportType = settings.getProperty(corpus + ".export.type");
        switch (corpus) {
            case "amr": 
                switch(exportType)
                {                    
                    case "linearize" : preLoadLinearizeDictionaries(verbose); break;
                }   break;
            case "mrs": preLoadLinearizeMrsDictionaries(verbose); break;
            case "gigaword":
                numOfNlNeTokens = new int[2]; neTokens = new HistMap<>();
                preLoadLinearizeDictionariesGiga(verbose, annotationUseNeClusters);
                break;
            default:
                preLoadLinearizeDictionariesAnonymize(verbose, annotationUseNeClusters);
                break;
        }
    }
    
    private void preLoadLinearizeDictionaries(boolean verbose)
    {
        // read in Named-Entity Concepts
        amrNeConcepts = readHashSet(settings.getProperty(corpus + ".concepts.ne"));
        if(verbose)
            System.out.println("Read " + amrNeConcepts.size() + " NE concepts from propreties.");
        // read in Named-Entity Clusters        
        amrNeClusters = readNeClusters(settings.getProperty(corpus + ".concepts.neClusters"));
        
        // read in Quantity Concepts
        amrQuantityConcepts = readHashSet(settings.getProperty(corpus + ".concepts.quantity"));
        if(verbose)
            System.out.println("Read " + amrQuantityConcepts.size() + " Quantity concepts from propreties.");
        // read in Value Concepts
        amrValueConcepts = readHashSet(settings.getProperty(corpus + ".concepts.value"));
        if(verbose)
            System.out.println("Read " + amrValueConcepts.size() + " Value concepts from propreties.");
        // read in Date Roles
        amrDateRoles = readHashSet(settings.getProperty(corpus + ".roles.date"));
        if(verbose)
            System.out.println("Read " + amrDateRoles.size() + " Date roles from propreties.");
        // read in filtered roles/concepts for linearized output
        linearizedFilter = readHashSet(settings.getProperty(corpus + ".linearize.filter"));
        if(verbose)
            System.out.println("Read " + linearizedFilter.size() + " concepts/roles to filter from propreties.");
        linearizedFilter.addAll(amrNeConcepts);
        linearizedFilter.addAll(amrQuantityConcepts);
        linearizedFilter.addAll(amrValueConcepts);
        
        amrRoles = new HashMap<>();
        Random rnd = new Random(1234);
        try {
            String path = settings.getProperty(corpus + ".roles.path");
            if(!(path == null || path.equals(""))) {
                boolean reentrancesRoles = settings.getProperty(corpus + ".down.reentrancesRoles").equals("true");
                    List<String> roles = reentrancesRoles 
                            ? Files.lines(Paths.get(path)).map(line -> line.split("\t")[0].toLowerCase()).collect(Collectors.toList())
                            : Files.lines(Paths.get(path)).filter(line -> !line.split("\t")[0].toLowerCase().contains("-r"))
                                .map(line -> line.split("\t")[0].toLowerCase())
                                .collect(Collectors.toList()); 
                    roles.add("<unk>");
                    List<Integer> ids = new ArrayList<>(roles.size());
                    for(int i = 0; i < roles.size(); i++) ids.add(i);
                    Collections.shuffle(ids, rnd);
                    for(int i = 0; i< roles.size(); i++) 
                        amrRoles.put(new AmrRole(roles.get(i)), ids.get(i));
                }
            } catch(IOException ioe) {
                System.err.println("Couldn't read roles file");
            }            
    }
    
    private void preLoadLinearizeMrsDictionaries(boolean verbose)
    {
        preLoadLinearizeDictionaries(verbose);
        mrsFeatures = readHashSet(settings.getProperty("mrs.roles.features"));
        if(verbose)
            System.out.println("Read " + mrsFeatures.size() + " MRS features from propreties.");
    }        
    
    private void preLoadLinearizeDictionariesGiga(boolean verbose, boolean annotationUseNeClusters) {
        // read in Named-Entity Concepts
        amrNeConcepts = readHashSet(settings.getProperty("amr.concepts.ne"));
        if(verbose)
            System.out.println("Read " + amrNeConcepts.size() + " NE concepts from propreties.");
        // read in Named-Entity Clusters
        amrNeClusters = readNeClusters(settings.getProperty("amr.concepts.neClusters"));
        // read in Anonymization Alignments from AMR corpus        
        amrAnonymizationAlignments = readNeAmrAlignments(settings.getProperty(corpus + ".down.amrAnonymization"), annotationUseNeClusters);
        neNlAmrAlignments = readNeNlAmrAlignments(settings.getProperty(corpus + ".down.amrAnonymization"));
        amrNlVocabulary = readHistMap(settings.getProperty(corpus + ".down.amrVocabulary"));
    }
    
    private void preLoadLinearizeDictionariesAnonymize(boolean verbose, boolean annotationUseNeClusters) {
        // read in Named-Entity Concepts
        amrNeConcepts = readHashSet(settings.getProperty("amr.concepts.ne"));
        if(verbose)
            System.out.println("Read " + amrNeConcepts.size() + " NE concepts from propreties.");
        // read in Named-Entity Clusters
        amrNeClusters = readNeClusters(settings.getProperty("amr.concepts.neClusters"));
        // read in Anonymization Alignments from AMR corpus        
        amrAnonymizationAlignments = readNeAmrAlignments(settings.getProperty("gigaword.down.amrAnonymization"), annotationUseNeClusters);
        neNlAmrAlignments = readNeNlAmrAlignments(settings.getProperty("gigaword.down.amrAnonymization"));
    }
    
     private Map<String, Integer> readHistMap(String path) {
        Map<String, Integer> out = new HashMap<>();
        try (Stream<String> stream = Files.lines(Paths.get(path))) {
            stream.forEach(line -> {
                String[] wordFreq = line.split("\t");
                out.put(wordFreq[0], Integer.valueOf(wordFreq[1]));
            });
        } catch (IOException ioe) {
            ioe.printStackTrace(System.err);
        }
        return out;
    }
     
    private Set<String> readHashSet(String str) {
        return str == null ? new HashSet<>() : new HashSet<>(Arrays.asList(str.split(",")));
    }
    
    private Map<String, String> readNeClusters(String str) 
    {
        Map<String, String> out = new HashMap();
        if (str == null) {
            return out;
        }
        Arrays.asList(str.split("#")).stream()
                .forEach(cluster -> {
                    String[] nameToks = cluster.split(":");
                    Arrays.asList(nameToks[1].split(",")).stream()
                        .forEach(ne -> out.put(ne, nameToks[0]));
                });
        return out;
    }
       
    private Map<String, String> readNeAmrAlignments(String path, boolean annotationUseNeClusters) {
        Map<String, String> out = new HashMap<>();
        
        try (Stream<String> stream = Files.lines(Paths.get(path))) {
            stream.forEach(example -> {
                String[] nameAlignments = example.split("\t");
                if (nameAlignments.length > 1) { // there is at least one alignment                    
                    Arrays.asList(nameAlignments[1].split(" # ")).stream()
                            .forEach((String alignment) -> {
                                AnonymizationAlignment a = new AmrAnonymizationAlignment(alignment);
                                if (!a.isEmpty() && a.getAnonymizedToken().contains("name")) {
                                    String anonymizedEntity = a.getAnonymizedToken().substring(0, a.getAnonymizedToken().indexOf("_"));
                                    String ne = annotationUseNeClusters ? amrNeClusters.get(anonymizedEntity) : anonymizedEntity;
                                    // attempt to store NL / NE-cluster-label pairs
                                    String nl = a.nlToString().toLowerCase();
                                    if (!nl.isEmpty()) {
                                        out.put(nl, ne);
                                        if(numOfNlNeTokens != null && neTokens != null) {
                                            numOfNlNeTokens[0] += a.getNlTokens().size();
                                            a.getNlTokens().stream().forEach(neTokens::add);
                                        }
                                    }
                                    String amrToken = ((AmrAnonymizationAlignment) a).getRawAmrToken().toLowerCase();
                                    out.put(amrToken.substring(amrToken.indexOf("_") + 1, amrToken.length()).replaceAll("_", " "), ne);
                                }
                            });
                }
            });
        } catch (IOException ioe) {
            ioe.printStackTrace(System.err);
        }        
        return out;
    }
    
    /**
     * 
     * Read NL to AMR alignments for NEs only (e.g., Chinese aligns to China)
     * @param path
     * @return 
     */
    private Map<String, String> readNeNlAmrAlignments(String path) {
        Map<String, String> out = new HashMap<>();
        try (Stream<String> stream = Files.lines(Paths.get(path))) {
            stream.forEach(example -> {
                String[] nameAlignments = example.split("\t");
                if (nameAlignments.length > 1) { // there is at least one alignment                    
                    Arrays.asList(nameAlignments[1].split(" # ")).stream()
                            .forEach((String alignment) -> {
                                AnonymizationAlignment a = new AmrAnonymizationAlignment(alignment);
                                if (!a.isEmpty()) {                                                                        
                                    // attempt to store NL / Canonicalized AMR tokens
                                    String nl = a.nlToString().toLowerCase();
                                    if (!nl.isEmpty()) {
                                        String amrToken = ((AmrAnonymizationAlignment) a).getRawAmrToken();
                                        out.put(nl, amrToken.substring(amrToken.indexOf("_") + 1, amrToken.length()).replaceAll("_", " "));
                                    }                                    
                                }
                            });
                }
            });
        } catch (IOException ioe) {
            ioe.printStackTrace(System.err);
        }
        return out;
    }
        

    public Set<String> getAmrDateRoles() {
        return amrDateRoles;
    }
    
    public Map<String, String> getAmrNeClusters() {
        return amrNeClusters;
    }

    public Set<String> getAmrNeConcepts() {
        return amrNeConcepts;
    }

    public Set<String> getAmrQuantityConcepts() {
        return amrQuantityConcepts;
    }

    public Set<String> getAmrValueConcepts() {
        return amrValueConcepts;
    }
    
    public Set<String> getLinearizedFilter() {
        return linearizedFilter;
    }
    
    public Map<AmrRole, Integer> getAmrRoles() {
        return amrRoles;
    }
        
    public Integer getAmrRoleOrderId(AmrRole role) 
    {
        return amrRoles.getOrDefault(role, 0);
    }

    public Map<String, String> getAmrAnonymizationAlignments() {
        return amrAnonymizationAlignments;
    }

    public Map<String, String> getNeNlAmrAlignments() {
        return neNlAmrAlignments;
    }

    public Map<String, Integer> getAmrNlVocabulary() {
        return amrNlVocabulary;
    }

    public int[] getNumOfNlNeTokens() {
        return numOfNlNeTokens;
    }

    public HistMap<String> getNeTokens() {
        return neTokens;
    }

    public Set<String> getMrsFeatures() {
        return mrsFeatures;
    }
        
}
