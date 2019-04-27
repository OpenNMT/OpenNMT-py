package util.corpus.wrappers;

import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.commons.collections4.map.MultiValueMap;

/**
 *
 * @author sinantie
 */
public class AmrLinearize extends Amr
{
           
    protected  List<AmrComponent> linearizedGraph;
    private List<AmrProposition> propositions;
    private final Set<String> filteredSet;
    protected MultiValueMap<String, AmrConcept> varToConceptObj;
    protected final String linearizeType;
    private final boolean ALLOW_WIKI = false;
    protected final boolean ALLOW_NEs = false, ALLOW_NUMS = false;
    
    public AmrLinearize(String id, String raw, Dictionaries dictionaries, String linearizeType)
    {
        super(id, raw, dictionaries);       
        this.linearizeType = linearizeType;
        filteredSet = dictionaries == null ? Collections.EMPTY_SET : dictionaries.getLinearizedFilter();        
    }
    
    public AmrLinearize(String id, AmrNode amr, Dictionaries dictionaries, String linearizeType)
    {
        super(id, amr, dictionaries);
        this.linearizeType = linearizeType;
        filteredSet = dictionaries == null ? Collections.EMPTY_SET : dictionaries.getLinearizedFilter();        
    }
          
    @Override
    protected void convert()
    {
        linearizedGraph = new LinkedList<>();
        propositions = new LinkedList<>();
        varToConceptObj = new MultiValueMap<>();        
        switch (linearizeType)
        {
            case "dfs":
                dfs(graph, null, "0");                
                //rename(graph);
                break;
            case "majority":
                throw new UnsupportedOperationException("Not supported yet.");
            case "classifier":
                throw new UnsupportedOperationException("Not supported yet.");
            default:
                throw new UnsupportedOperationException("Not supported yet.");
        }        
    }

/*    private void rename(AmrNode root)
    {
        ((AmrConcept)root).setName(root.getVar() + ' ' + '/' + ' ' + ((AmrConcept)root).getName());
        if(root.isLeaf())
            return;
        Enumeration<AmrNode> children = root.children();
        while(children.hasMoreElements())
        {
            AmrNode child = children.nextElement();
            rename(child);
        }
    }*/

    private void dfs(AmrNode root, AmrProposition proposition, String nodeId)
    {        
        // add concept to the traversal path
        AmrConcept.Type conceptType = determineConceptType(root);
        AmrComponent concept = new AmrConcept(root.getWord() ,//+ (root.isCyclicReference() ? "-c" : ""), //exclude cyclic references
                root.getSense(), root.getVar(), root.isCyclicReference(), conceptType, root.getConceptAlignment(), nodeId);        
        linearizedGraph.add(concept);
        varToConceptObj.put(root.getVar(), concept);
        if(proposition != null) // add argument to the proposition headed by its parent from the previous recursion
        {
            proposition.setArgument((AmrConcept) concept);
            ((AmrConcept)concept).setParent(proposition.getPredicate());
        }
        if(root.isLeaf())                     
            return;
        Enumeration<AmrNode> children = root.children();
        int childId = 0;
        while(children.hasMoreElements())
        {
            AmrNode child = children.nextElement();
            // tackle Named Entities
            if(!ALLOW_NEs && conceptType == AmrConcept.Type.NAME && isNameConcept(child))
            {
                // aggregate op* name alignments
                Enumeration<AmrNode> grandChildren = child.children();
                while(grandChildren.hasMoreElements())
                {
                    AmrNode grandChild = grandChildren.nextElement();
                    child.getConceptAlignment().addAlignments(grandChild.getConceptAlignment());
                }
                AmrConcept auxConcept = new AmrConcept(root.getVar(), dictionaries.getAmrNeClusters().getOrDefault(concept.getName(), concept.getName()) + "_name", AmrConcept.Type.NAME, child.getConceptAlignment(), String.format("%s.%s", nodeId, childId++));
                updateGraph((AmrConcept)concept, auxConcept);
            }
            else if(!ALLOW_NUMS && conceptType == AmrConcept.Type.QUANTITY && isQuantConceptWithNum(child))
            {
                AmrConcept auxConcept = new AmrConcept(root.getVar(), concept.getName() + "_num", AmrConcept.Type.QUANTITY, child.getConceptAlignment(), String.format("%s.%s", nodeId, childId++));
                updateGraph((AmrConcept)concept, auxConcept);
            }
            else if(!ALLOW_NUMS && conceptType == AmrConcept.Type.VALUE && isValueConceptWithNum(child))
            {
                AmrConcept auxConcept = new AmrConcept(root.getVar(), concept.getName() + "_num", AmrConcept.Type.VALUE, child.getConceptAlignment(), String.format("%s.%s", nodeId, childId++));
                updateGraph((AmrConcept)concept, auxConcept);
            }
            else if(conceptType == AmrConcept.Type.DATE && hasDateEntityRole(child))
            {
//                AmrConcept auxConcept = new AmrConcept(child.getWord() + "_" + child.getRole(), AmrConcept.Type.DATE, child.getConceptAlignment());
                AmrConcept auxConcept = new AmrConcept(root.getVar(), child.getRole() + "_date-entity" , AmrConcept.Type.DATE, child.getConceptAlignment(), String.format("%s.%s", nodeId, childId++));
                updateGraph((AmrConcept)concept, auxConcept);
            }
//            else if(conceptType == AmrConcept.Type.DATE && hasTimeRole(child))
//            {                
//                AmrConcept auxConcept = new AmrConcept(root.getVar(), child.getRole() + "_entity" , AmrConcept.Type.TIME, child.getConceptAlignment(), String.format("%s.%s", nodeId, childId++));
//                updateGraph((AmrConcept)concept, auxConcept);
//            }            
            else
            {
                // append role names in the traversal path
                AmrRole role = new AmrRole(":" + child.getRole(), child.getRoleAlignment());                
                role.setOrderId(dictionaries.getAmrRoleOrderId(role));
                linearizedGraph.add(role);
                AmrProposition newProposition = new AmrProposition((AmrConcept) concept, role);
                propositions.add(newProposition);
                ((AmrConcept)concept).addChild(newProposition);
                dfs(child, newProposition, String.format("%s.%s", nodeId, childId++)); // recurse for the rest of the graph
            }            
        } // for all children
    }  
    
    private void updateGraph(AmrConcept predicate, AmrConcept argument) 
    {
        linearizedGraph.add(argument);
        AmrProposition newProposition = new AmrProposition((AmrConcept)predicate, (AmrConcept)argument, new AmrRole(":unk"));
        predicate.addChild(newProposition);
        argument.setParent(predicate);
        propositions.add(newProposition);
    }
    
    protected List<AmrComponent> filterGraph()
    {
        boolean removedFilteredToken = false;        
        ListIterator<AmrComponent> it = linearizedGraph.listIterator();                
        while(it.hasNext())
        {            
            AmrComponent next = it.next();
            if(next.getName().contains("\""))
            {
                next.setName(next.getName().replaceAll("\"", ""));
            }
            if(removedFilteredToken && it.hasNext() && next instanceof AmrRole && !next.getName().equals(":polarity"))
            {
                it.remove();
                if(!ALLOW_WIKI && next.getName().equals(":wiki"))
                {
                    it.next();
                    it.remove();
                }
                removedFilteredToken = false;
            }
            else if(filteredSet.contains(next.getName()))
            {                
                if(next instanceof AmrConcept && ((AmrConcept)next).hasSense())
                {
                }
                else
                {
                    it.remove();
                    removedFilteredToken = true;
                }                
            }   
//            else if(!ALLOW_WIKI && next.getName().equals(":wiki"))
//            {
//                it.remove();
//                it.next();
//                it.remove();
//            }
            // no need to keep two tokens for :polarity - (there is never a positive polarity)
            else if(next.getName().equals(":polarity"))
            {
                next.setName(":polarity-neg");
                it.next();
                it.remove();
            }
            // replace number literal with <num> placeholder
            else if(next instanceof AmrConcept && ((AmrConcept)next).getType() == AmrConcept.Type.NUMBER &&
                    ((AmrConcept)next).getType() != AmrConcept.Type.DATE)
            {
//                next.setName("<num>");
            }            
            else if(it.hasNext() && next instanceof AmrConcept && ((AmrConcept)next).isCyclicReference()) // remove cyclic references
            {                
                it.remove();
                it.previous();
                it.remove();
            }
        }
        return linearizedGraph;
    }
    
    /**
     * 
     * Normalizes graph in terms of removing quotes, redundant NE-related tokens, :wiki edges (if existing)
     * and replaces anonymized named entities / dates / numbers with a minimal realized token from AMR if <code>deAnonymize</code> is set to true.
     * @param deAnonymize
     * @param includeReentrances
     * @param reentrancesRoles
     * @return 
     */
    protected List<AmrComponent> simplifyGraph(boolean deAnonymize, boolean includeReentrances, boolean reentrancesRoles) {        
        ListIterator<AmrComponent> it = linearizedGraph.listIterator();                
        while(it.hasNext()) {            
            AmrComponent next = it.next();
            if(next.getName().contains("\"")) {
                next.setName(next.getName().replaceAll("\"", ""));
            }
            if(next instanceof AmrConcept) {
                AmrConcept concept = (AmrConcept)next;
                switch(concept.type) {
                    case NAME:
                        if(concept.getName().contains("_name_")) {
                            if(deAnonymize) {
                                String[] toks = concept.getAlignment().getToken().split("_");
                                concept.setName(Arrays.stream(toks, 1, toks.length).collect(Collectors.joining(" ")));
                            }
                            if(it.hasPrevious()) {
                                it.previous();
                                AmrComponent prev = it.previous();
                                // we remove only the token that redundantly refers to the actual (anonymized) NE (prev); 
                                // therefore alignment ids between NL and AMR should remain intact.
                                if(filteredSet.contains(prev.getName())) {
                                    if(prev instanceof AmrConcept && ((AmrConcept)prev).hasSense()) {
                                        it.next(); // forward again as we didn't remove anything
                                        if(it.hasNext()) {it.next();}
                                    } else {
                                        // special treatment for children propositions: 
                                        // for every child of prev (which is going to be removed), connect to prev's father. 
                                        AmrConcept parent = ((AmrConcept)prev).getParent();
                                        ((AmrConcept) next).setParent(parent); // NE concept (next) gets assigned prev's parent
                                        // for every child of the parent that is connected to prev, reconnect it to the NE concept (next)
                                        if(parent != null) {
                                            parent.getChildren().stream().forEach( (AmrProposition p) -> {
                                                if(p.getArgument().equals(prev)) {
                                                    p.setArgument((AmrConcept)next);
                                                } 
                                            });
                                        }
                                        // cycle through prev's children and make sure there is no argument left that shouldn't be deleted. 
                                        // If there is, then re-attach it to the NE concept (careful not to re-attach the NE concept itself)
                                        ((AmrConcept)prev).getChildren().stream().forEach( (AmrProposition p) -> {
                                            String role = p.getRole().getName();
                                            if(!(role.equals(":unk") || role.equals(":wiki"))) {// the NE concept itself or :wiki --we don't won't these either
                                                concept.addChild(p);
                                            }
                                        });
                                        
                                        it.remove();   
                                        it.next(); // forward again to go past the NE                                        
                                        
                                    }                                    
                                } else {
                                    it.next(); // forward again as we didn't remove anything
                                    if(it.hasNext()) {it.next();}
                                }
                            }
                        } break;
                    case DATE:
                        if(concept.getName().contains("_date-entity_") && deAnonymize) {
                            concept.setName(concept.getAlignment().getToken());
                        } break;
//                    case TIME:
//                        if(concept.getName().contains("time_entity_")) {
//                            if(deAnonymize)
//                                concept.setName(concept.getAlignment().getToken());
//                            if(it.hasPrevious()) {
//                                it.previous();
//                                AmrComponent prev = it.previous();
//                                // we remove only token that redundantly refers to the actual (anonymized) NE; 
//                                // therefore alignment ids between NL and AMR should remain intact.
//                                if(prev.getName().equals("date-entity")) {
//                                    if(prev instanceof AmrConcept && ((AmrConcept)prev).hasSense()) {
//                                        it.next(); // forward again as we didn't remove anything
//                                        if(it.hasNext()) {it.next();}
//                                    } else {
//                                        it.remove();   
//                                        it.next(); // forward again to go past the NE
//                                    }                                    
//                                } else {
//                                    it.next(); // forward again as we didn't remove anything
//                                    if(it.hasNext()) {it.next();}
//                                }
//                            }                            
//                        } break;
                    case QUANTITY: case VALUE: case NUMBER:
                        if(concept.getName().contains("num_") && deAnonymize) {
                            concept.setName(concept.getAlignment().getToken());
                        } break;
                }
                if (concept.isCyclicReference()) { // remove cyclic reference but keep the role before it
                    AmrConcept parent = ((AmrConcept)concept).getParent();                    
                    // for every child of the parent that is connected to the cyclic reference, 
                    // change the name to ""
                    if(parent != null) {
                        parent.getChildren().stream().forEach( (AmrProposition p) -> {
                            if(p.getArgument().equals(concept)) {
                                if(!includeReentrances) {
                                    p.getArgument().setName("");
                                } else if(reentrancesRoles) {
                                    p.getRole().setName(p.getRole().getName() + "-r");
                                }
                            } 
                        });
                    }
                    if(!includeReentrances) {
                        it.remove();
                    }
                } // cyclicReference
                else if(!ALLOW_WIKI && concept.containsChildWithRoleName(":wiki")) { // check for children propositions with :wiki
                    concept.removeChildrenWithRoleName(":wiki");
                }
//            } else if(it.hasNext() && next.getName().equals(":polarity")) { // no need to keep two tokens for :polarity - (there is never a positive polarity)
//                next.setName(":polarity-neg");
//                it.next();
//                it.remove();
            } else if(!ALLOW_WIKI && next.getName().equals(":wiki")) { // remove the whole branch of the graph
//                    it.next();
                    it.remove();
                    if(it.hasNext()) {
                        it.next();
                        it.remove();
                    }
            }
        }
        return linearizedGraph;
    }

    protected List<AmrProposition> filterPropositions()
    {
        ListIterator<AmrProposition> it = propositions.listIterator();
        if(propositions.size() > 1)
        {
            swapAnonymizedEntityAtFirstPositionHeuristic(propositions);            
            while(it.hasNext())
            {
                AmrProposition next = it.next();
                if(next.getRole() == null && next.getPredicate().getType() != AmrConcept.Type.DATE)
                {
                    // asign the anonymized concept as an argument to the grand parent proposition and
                    // delete the intermediate propositions (usually AMR intermediate ontology triples).
                    AmrConcept anonymizedArgumentConcept = next.getArgument();                
                    AmrProposition grandParent = visitBackwards(it, next, next.getPredicate().getVar());                 
                    grandParent.setArgument(anonymizedArgumentConcept); 
                    // this is the parent NE concept with the correct variable information, used in re-entrancies. Use it to
                    // correctly carry forward anonymization in the rest re-entrance predicate/arguments.
                    String predicateConceptVar = next.getPredicate().getVar(); 
                    varToConceptObj.getCollection(predicateConceptVar).stream().forEach(reEntrance -> {
                        if(reEntrance != grandParent.getPredicate())
                            reEntrance.setName(anonymizedArgumentConcept.getName());
                    });
                }            
            }
        }
        it = propositions.listIterator(); // second round
        while(it.hasNext())
        {
            AmrProposition next = it.next();
            // heuristic: triples like (or, person_name_2, :op2) (person, have-org-role,  :ARG1-of)
            // should become (or, person_name_2, :op2) (person_name_2, have-org-role,  :ARG1-of)
            if(it.previousIndex() > 0)
            {
                AmrProposition prev = propositions.get(it.previousIndex() - 1);
                if(next.getPredicate().getName().equals("person") 
                        && prev.getArgument().getName().startsWith("person_name"))                        
                {
                    next.setPredicate(prev.getArgument());
                }
            }
            // heuristic: triples like (p/person, person_name_1,:domain) probably reference another
            // parent triple like (p/person, concept,:role): we remove the first triple and replace any instance
            // of the variable p with person_name_1.
            if(next.getPredicate().getName().equals("person") 
                    && next.getArgument().getName().contains("_name") 
                    && (next.getRole() != null && next.getRole().getName().equals(":domain")))
            {
                String predicateConceptVar = next.getPredicate().getVar(); 
                varToConceptObj.getCollection(predicateConceptVar).stream().forEach(reEntrance -> {                    
                        reEntrance.setName(next.getArgument().getName());
                });
                it.remove();
            }
        }
        return propositions;
    }
    
    private AmrProposition visitBackwards(ListIterator<AmrProposition> it, AmrProposition current, String childVar)
    {
        // reached back to the grandparent node that is not headed by a filtered (most likely NE) concept
        // OR reached the beginning of the graph.
        if(!filteredSet.contains(current.getPredicate().getName()) 
                || !childVar.equals(current.getPredicate().getVar()) || it.previousIndex() < 0)// || current.getPredicate().getName().equals("<num>"))
        
        
        {            
            return current;
        }
        it.remove();
        return visitBackwards(it, it.previous(), current.getPredicate().getVar());
    }
    
    /**
     * 
     * This method looks for an anonymized entity at the begin of the propositions' list, and pushes
     * it further down, i.e., 'after' its siblings headed by the same concept.
     * Example: (monetary-quantity, monetary-quantity_num_0, :unk) (monetary-quantity, balance, :domain), should be
     * swapped.
     * @param propositions 
     */
    private void swapAnonymizedEntityAtFirstPositionHeuristic(List<AmrProposition> propositions) 
    {
        AmrProposition first = propositions.get(0);
        if(first.getRole() == null && first.getPredicate().getType() != AmrConcept.Type.DATE) 
        {
            String parentVar = first.getPredicate().getVar();
            int i = 1;            
            String nextParentVar = propositions.get(i).getPredicate().getVar();
            while(parentVar.equals(nextParentVar))
            {                
                Collections.swap(propositions, i - 1, i);
                i++;
                nextParentVar = i < propositions.size() ? propositions.get(i).getPredicate().getVar() : "";
            }
        }
    }
    
    public List<AmrComponent> getLinearizedGraph()
    {
        return linearizedGraph;
    }
    
    public String propositionsToString()
    {
        StringBuilder str = new StringBuilder();
        propositions.stream().forEach((prop) -> {
            str.append(prop).append(" ");
        });
        return str.toString().trim();
    }
    
    public boolean isEmpty()
    {
        return linearizedGraph.isEmpty();
    }
    
    public int size()
    {
        return linearizedGraph.size();
    }
    
    public int propositionsSize()
    {
        return propositions.size();
    }
    
    @Override
    public String toString()
    {        
        StringBuilder str = new StringBuilder();
        AmrConcept root = (AmrConcept) linearizedGraph.get(0);
        String out = AmrConcept.print(root, "", str, false, false, false, false, false, true).toString().toLowerCase().trim();
        return out;
//        return linearizedGraph.stream().map(comp -> lowerCaseOutput ? comp.getName().toLowerCase() : comp.getName())
//                .collect(Collectors.joining(" "));
//                str.append(comp.getName()).append("^").append(comp.getNodeId()).append(" ");
    }           
    
    public String toStringBrackets(boolean reshuffleChildren, boolean markLeaves, boolean outputSense, boolean concatBracketsWithRoles)
    {        
        StringBuilder str = new StringBuilder();
        AmrConcept root = (AmrConcept) linearizedGraph.get(0);
        String out = AmrConcept.print(root, "", str, true, reshuffleChildren, markLeaves, outputSense, concatBracketsWithRoles, true).toString().toLowerCase().trim();
        System.out.println(out);
        // remove first and last parentheses...kind of redundant. Sorry, LISP fans.
        return (out.split(" ").length > 1 && out.length() > 2 ? out.substring(2, out.length() - 2) : out).trim(); 
    }           
}
