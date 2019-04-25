package util.corpus.wrappers;

import fig.basic.Indexer;
import fig.basic.Pair;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import util.utils.HistMap;
import util.utils.StemmerPosTagger;

/**
 *
 * @author sinantie
 */
public class AmrLinearizedSentence extends AmrSentence
{    
    private final String SOS = "<s>", EOS = "</s>", UNK = "<unk>", NUM = "<num>", DATE = "<date>";
    protected final String linearizeType;
    protected Set<AmrAnonymizationAlignment> anonymizationAlignments;
    private Set<AmrAnonymizationAlignment> contentAlignments;
    private final Map<String, JamrAlignment> jamrAlignments;    
    private final String FORMAT = "[0-9]{2,4}[-]?[0-9]{2}[-]?[0-9]{2}";
    
    public AmrLinearizedSentence(String id, String sentence, Map<String, JamrAlignment> jamrAlignments, String linearizeType)
    {
        super(id, sentence);
        this.linearizeType = linearizeType;
        this.jamrAlignments = jamrAlignments;
    }
    
    public AmrLinearizedSentence(String id, String sentence, AmrNode root, Dictionaries dictionaries, Map<String, JamrAlignment> jamrAlignments, String linearizeType)
    {
        super(id, sentence);
        this.linearizeType = linearizeType;
        this.jamrAlignments = jamrAlignments;
        amr = new AmrLinearize(id, root, dictionaries, linearizeType);
        anonymizationAlignments = new LinkedHashSet<>();
        contentAlignments = new LinkedHashSet<>();
        amr.convert();
    }

    @Override
    public void parseAmr(String id, String rawAmr, Dictionaries dictionaries)
    {
        amr = new AmrLinearize(id, rawAmr, dictionaries, linearizeType);
        anonymizationAlignments = new LinkedHashSet<>();
        contentAlignments = new LinkedHashSet<>();
//        if(!amr.failedToParse) {
            amr.convert();
//        }
    }
       
     
    /**
     * 
     * Apply the following alignment-based heuristics:
     * - Replace NEs, Numbers, Dates in the sentence with placeholder tokens
     * - Extract words that align with each amr token and match them with an id from word2vec if it exists
     * pretrained word embeddings for the amr tokens.
     * @param amrNlAlignmentsMap an empty map to fill in with unique content-only alignments between AMR and NL words coming either from unsupervised or JAMR alignments.
     * @param deAnonymizeGraph
     * @param includeReentrances
     * @param reentrancesRoles
     */
    public void applyAlignmentHeuristics(Map<String, HistMap<String>> amrNlAlignmentsMap, boolean deAnonymizeGraph, boolean includeReentrances, boolean reentrancesRoles)
    {
        try{
        int nameCounter = 0, quantityCounter = 0, numberCounter = 0, dateCounter = 0, dayCounter = 0, monthCounter = 0, yearCounter = 0, timeCounter = 0;
        int dateOffset = 0, offset = 0;
        Map<String, Pair<String, String>> varToPlaceHolder = new HashMap<>();
        String[] anonymizedSent = sentence.split(" ");
        String[] sent = sentence.split(" ");
        Iterator<AmrComponent> it = ((AmrLinearize)amr).getLinearizedGraph().listIterator();
        while(it.hasNext())
        {            
            AmrComponent token = it.next();
            if(token instanceof AmrConcept)
            {
                AmrConcept concept = (AmrConcept)token;
                offset = anonymizedSent.length - sent.length;
                switch(concept.type)
                {
                    case NAME: 
                        if(concept.getName().endsWith("_name")) 
                        {
                            String placeholder = concept.getName() + "_" + nameCounter;
                            anonymizeSentenceTokens(anonymizedSent, concept, placeholder, true, offset);
                            varToPlaceHolder.put(concept.getVar(), new Pair(placeholder, concept.getAlignment().getToken()));
                            concept.setName(concept.getName() + "_" + nameCounter++);
                        } 
                        else if(concept.isCyclicReference() && varToPlaceHolder.containsKey(concept.getVar()))
                        {
                            Pair<String, String> placeholderToken = varToPlaceHolder.get(concept.getVar());
                            concept.getAlignment().setToken(placeholderToken.getSecond());
                            anonymizeSentenceTokens(anonymizedSent, concept, placeholderToken.getFirst(), true, offset);
                        }
                        // don't forget to create content alignments for concepts that are generally tagged as redundant ontology-based NEs, 
                        //such  as city, country etc., but are used this time as normal nouns in the NL.
                        else 
                        {
                            createContentAlignment(anonymizedSent, concept, amrNlAlignmentsMap);
                        }
                    break;
                    case QUANTITY: case VALUE: 
                        if(concept.getName().endsWith("_num")) 
                        {
                            anonymizeSentenceTokens(anonymizedSent, concept, concept.getName() + "_" + quantityCounter, true, offset); 
                            concept.setName(concept.getName() + "_" + quantityCounter++);
                        }
                    break;                    
//                    case NUMBER: anonymizeSentenceTokens(anonymizedSent, concept.getAlignment(), NUM, true); break;
                    case NUMBER: anonymizeSentenceTokens(anonymizedSent, concept, "num_" + numberCounter, true, offset); 
                    concept.setName("num_" + numberCounter++);
                    break;
                    case DATE: 
//                        if(!concept.getName().equals("date-entity")) 
                        if(concept.getName().endsWith("_date-entity")) 
                        {// do all the fancy alignment business, but don't replace the NL side with the anonymization token
//                            anonymizeSentenceTokens(anonymizedSent, concept.getAlignment(), concept.getName(), false);
                         // do all the fancy alignment business, AND replace the NL side with the anonymization token
                            String[] anonymizedSentIn;
                            switch(concept.getName().substring(0, concept.getName().indexOf("_"))) {
                                case "day" : anonymizedSentIn = anonymizeDateSentenceTokens(sent, anonymizedSent, concept, dayCounter, dateOffset, "day", offset);                                             
                                             concept.setName(concept.getName() + "_" + dayCounter++); break;
                                case "month" : anonymizedSentIn = anonymizeDateSentenceTokens(sent, anonymizedSent, concept, monthCounter, dateOffset, "month", offset);
                                             concept.setName(concept.getName() + "_" + monthCounter++); break;
                                case "year" : anonymizedSentIn = anonymizeDateSentenceTokens(sent, anonymizedSent, concept, yearCounter, dateOffset, "year", offset);
                                             concept.setName(concept.getName() + "_" + yearCounter++); break;
                                default : anonymizedSentIn = anonymizeDateSentenceTokens(sent, anonymizedSent, concept, dateCounter, dateOffset, "", offset);
                                             concept.setName(concept.getName() + "_" + dateCounter++);
                            }                            
                            dateOffset = dateOffset + (anonymizedSentIn[0].equals("n") ? 0 : 1);
                            anonymizedSent = Arrays.copyOfRange(anonymizedSentIn, 1, anonymizedSentIn.length);
                        } 
                    break;
//                    case TIME: 
//                        if(concept.getName().endsWith("time_entity")) 
//                        {
//                            anonymizeSentenceTokens(anonymizedSent, concept, concept.getName() + "_" + timeCounter, true, offset);
//                            concept.setName(concept.getName() + "_" + timeCounter++);
//                        }
//                        break;
                    case OTHER: default: createContentAlignment(anonymizedSent, concept, amrNlAlignmentsMap); break;
                } // switch                                
            } // if
        } // while
//        setSentence(repack(sent, sentenceVocabulary, useVocabulary));        
        
        setAnonymizedSentence(cleanupAnonymizedSentence(rejoinHyphenatedWords(repack(anonymizedSent, null, false), true)));
        setSentence(rejoinHyphenatedWords(sentence, false));
        
//        ((AmrLinearize)amr).filterGraph(); // remove non-frequent amr tokens completely from the graph
        ((AmrLinearize)amr).simplifyGraph(deAnonymizeGraph, includeReentrances, reentrancesRoles); 
//        ((AmrLinearize)amr).filterPropositions(); // remove non-frequent amr tokens completely from the graph
//        System.out.println(amr);
        }catch(Exception e)
        {
            System.err.println(amr.id);
            e.printStackTrace(System.err);
        }

        Iterator<AmrComponent> it = ((AmrLinearize)amr).getLinearizedGraph().listIterator();
        /*AmrComponent root = it;
        if(root instanceof AmrConcept)
        {
            AmrConcept root_concept = (AmrConcept)root;
            if (!root_concept.getVar().isEmpty())
                root_concept.setName(root_concept.getVar() + ' ' + '/' + ' ' + root_concept.getName());
        }*/
        /*List<AmrComponent> lin_graph = ((AmrLinearize)amr).getLinearizedGraph();
        System.out.println(lin_graph);
        for (AmrComponent component: lin_graph) {
            if(component instanceof AmrConcept){
                AmrConcept concept = (AmrConcept)component;
                if (!concept.getVar().isEmpty())
                    concept.setName(concept.getVar() + ' ' + '/' + ' ' + concept.getName());
            }
        }*/
        it.next();
        while(it.hasNext())
        {
            AmrComponent token = it.next();
            if(token instanceof AmrConcept)
            {
                AmrConcept concept = (AmrConcept)token;
                if (!concept.getVar().isEmpty())
                    concept.setName(concept.getVar() + ' ' + '/' + ' ' + concept.getName());        
            }
        }
    }
    
    private void anonymizeSentenceTokens(String[] sentence, AmrConcept concept, String placeholder, boolean replaceInNl, int offset)
    {        
        List<Integer> nlIds = new ArrayList<>();
        List<String> nlTokens = new ArrayList<>();        
        AmrAlignment alignment = concept.getAlignment();        
        // jamr doesn't handle rentrances; in fact it skips their ids entirely and assigns them to the next token in the graph! OMG
        if(alignment.isEmpty() && !concept.isCyclicReference() && jamrAlignments.containsKey(concept.nodeId))
        {
            JamrAlignment jamr = jamrAlignments.get(concept.nodeId);
            concept.getAlignment().copyFromJamrAlignment(jamr);
        }
        for(int wordId : alignment.getWordIds())
        {
            if(wordId >= 0 && wordId < sentence.length)
            {
                // in case we have introduced a new word (such as a canonicalized date entry; 
                // eg NL 08071998 introduces to new canonicalized date entries
                int offsetWordId = wordId + offset; 
                nlTokens.add(sentence[wordId]);
                if(replaceInNl) 
                    sentence[offsetWordId] = placeholder;
                nlIds.add(wordId);                
            }
        }
        if(!nlIds.isEmpty())
            anonymizationAlignments.add(new AmrAnonymizationAlignment(placeholder, alignment.getToken(), nlTokens, nlIds, nlIds.get(0) + offset));
        else
            anonymizationAlignments.add(new AmrAnonymizationAlignment(placeholder, alignment.getToken()));
    }
    
    private String[] anonymizeDateSentenceTokens(String[] sentence, String[] anonSentenceIn, AmrConcept concept, int counter, int dateOffset, String dateType, int offset)
    {        
        List<String> anonSentence = new LinkedList<>(Arrays.asList(anonSentenceIn));
        AmrAlignment alignment = concept.getAlignment();
        boolean replacedWithJamr = false;
        if(concept.getAlignment().isEmpty() && jamrAlignments.get(concept.nodeId) != null)
        {
            JamrAlignment jamr = jamrAlignments.get(concept.nodeId);
            int jamrIdToUse = -1;
            switch (jamr.nlIds.size()) {
            // we assume that the NL is in a canonical form of DD Month YYYY
                case 3:
                    switch(dateType) {
                        case "year": jamrIdToUse = 2; break;
                        case "month": jamrIdToUse = 1; break;
                        case "day": jamrIdToUse = 0; break;
                    }   break;
            // we assume that the NL is in a canonical form of Month YYYY
                case 2:
                    switch(dateType) {
                        case "year": jamrIdToUse = 1; break;
                        case "month": jamrIdToUse = 0; break;
                    }   break;
                default:
                    concept.getAlignment().copyFromJamrAlignment(jamr);
                    break;
            }
            if(jamrIdToUse > -1)
            {
                alignment.getWordIds().set(0, jamr.nlIds.get(jamrIdToUse)); 
                alignment.getWords()[0] = jamr.nlTokens.get(jamrIdToUse);
            }
            replacedWithJamr = true;
        }
        String placeholder = concept.getName() + "_" + counter;
        List<Integer> nlIds = new ArrayList<>();
        List<String> nlTokens = new ArrayList<>();
        int anonymizedId = -1;
        boolean modifiedSentenceLength = false;
        for(int wordId : alignment.getWordIds())
        {
            if(wordId >= 0 && wordId < sentence.length)
            {
                // in case we have PREVIOUSLY introduced a new word (such as a canonicalized date entry; 
                // eg NL 08071998 introduces to new canonicalized date entries
                int offsetWordId = wordId + dateOffset; 
                String nlWord = sentence[wordId];
                if(dateType.equals("month") && nlWord.matches("\\p{Alpha}+"))
                    placeholder = "month_name_date-entity_" + counter;
                String format = "";
                if(nlWord.matches("[0-9]{6}"))
                    format = "f1";
                else if(nlWord.matches("[0-9]{8}"))
                    format = "f2";
                else if(nlWord.matches("[0-9]{4}-[0-9]{2}-[0-9]{2}"))
                    format = "f3";
                if(!format.equals(""))
                {
                    placeholder = dateType + "_" + format + "_date-entity_" + counter;
                    anonymizedId = offsetWordId;//offsetWordId + dateOffset;    
                    if(dateOffset > 0) {
                        adjustAlignmentIndices(anonymizedId, true, false);
                    }
                    if(anonymizedId < anonSentence.size() && anonSentence.get(anonymizedId).matches(FORMAT)) // remove formatted date from anonSentence (the first time)
                        anonSentence.remove(anonymizedId);
                    anonSentence.add(anonymizedId, placeholder);
                    modifiedSentenceLength = true;
                }
                else
                {
                    anonymizedId = wordId + offset;//nlIds.isEmpty() ? -1 : nlIds.get(0);
                    anonSentence.set(anonymizedId, placeholder);
                    
                }
                nlTokens.add(sentence[wordId]);
                nlIds.add(wordId);                                    
            }
        }        
        if(replacedWithJamr)
            anonymizedId = alignment.getWordIds().get(0);        
        if(!nlIds.isEmpty())
            anonymizationAlignments.add(new AmrAnonymizationAlignment(concept.getName() + "_" + counter, alignment.getToken(), nlTokens, nlIds, anonymizedId));
        else
            anonymizationAlignments.add(new AmrAnonymizationAlignment(placeholder, alignment.getToken()));                
        anonSentence.add(0, modifiedSentenceLength ? "y" : "n");        
        return anonSentence.toArray(new String[0]);
    }
    
    private void createContentAlignment(String[] sentence, AmrConcept concept, Map<String, HistMap<String>> amrNlAlignmentsMap)//, Map<String, Integer> word2vecInvMap)
    {
        AmrAlignment alignment = concept.getAlignment();
        List<Integer> nlIds = new ArrayList<>();        
        List<String> nlTokens = new ArrayList<>();
        boolean createdContentAlignment = false;
        if(alignment != null && !alignment.isEmpty())
        {
            alignment.getWordIds()
                    .stream().filter(wordId -> (wordId >= 0 && wordId < sentence.length))
                    .map(wordId -> { nlTokens.add(sentence[wordId]); return wordId;})
                    .forEach(wordId -> nlIds.add(wordId));
            if(!nlIds.isEmpty())
            {
//                String compoundNlWord = nlTokens.stream().collect(Collectors.joining("_")).toLowerCase();
//                // nl word2vec policy: try querying first concatenated nl tokens, e.g., wheel_parts; if that doesn't work default to -1
//                int nlWord2vecId = word2vecInvMap.getOrDefault(compoundNlWord, -1);
//                // amr word2vec policy: try querying first the compound amr token if it exists after replacing dashes with underscores, 
//                // e.g., catch_up; if that fails output -1.
//                String compoundAmrToken = concept.getName().replaceAll("-", "_");
//                int amrWord2vecId = word2vecInvMap.getOrDefault(compoundAmrToken, -1);
//                contentAlignments.add(new AmrWord2VecAlignment("-", concept.getName(), nlTokens, nlIds, nlIds.get(0), amrWord2vecId, nlWord2vecId));
                AmrAnonymizationAlignment contentAlignment = new AmrAnonymizationAlignment(concept.getNodeId(), concept.getName(), nlTokens, nlIds, nlIds.get(0));
                contentAlignments.add(contentAlignment);
                HistMap<String> hist = amrNlAlignmentsMap.getOrDefault(contentAlignment.amrToken, new HistMap<>());
//                hist.add(contentAlignment.getNlTokens().get(0).toLowerCase());
                hist.add(contentAlignment.getNlTokens().stream().collect(Collectors.joining(" ")).toLowerCase());
                amrNlAlignmentsMap.put(contentAlignment.amrToken, hist);
                createdContentAlignment = true;
            }
        }
        if(!createdContentAlignment && !concept.isCyclicReference() && jamrAlignments.containsKey(concept.nodeId)) {
            JamrAlignment jamrAlignment = jamrAlignments.get(concept.nodeId);
            if(!jamrAlignment.isEmpty()) {
                HistMap<String> hist = amrNlAlignmentsMap.getOrDefault(jamrAlignment.amrToken, new HistMap<>());
//                hist.add(jamrAlignment.getNlTokens().get(0).toLowerCase());
                hist.add(jamrAlignment.getNlTokens().stream().collect(Collectors.joining(" ")).toLowerCase());
                amrNlAlignmentsMap.put(jamrAlignment.amrToken, hist);
            }
        }        
    }
    
    /**
     * 
     * Take an array of tokens constituting a sentence and append them to a string.
     * Make sure we remove duplicates: this might have happened during the anonymization
     * process, e.g., 'in United States territory' becomes 'in country_name country_name territory'.
     * We also make sure we don't include OOV words by checking if they exist in the given vocabulary 
     * (usually for validation set).
     * @param toks
     * @param vocabulary 
     * @return 
     */
    private String repack(String[] toks, Indexer<String> vocabulary, boolean processOovWords)
    {
        StringBuilder str = new StringBuilder();
        String prevTok = "";
        int totalWords = 0;
        for(int i = 0; i < toks.length; i++)
        {
            String tok = toks[i];
            if(tok.equals("@-@") && prevTok.contains("_") && !prevTok.contains("date-entity")) // remove hyphen only between NEs and not between dates (e.g., intervals)
            {
                adjustAlignmentIndices(totalWords, true, true);                
            }
//            else if(tok.equals("@:@") && prevTok.contains("time_entity")) // remove @:@ between time
//            {
//                adjustAlignmentIndices(totalWords, true, true);                
//            }
            else if(!tok.equals(prevTok))
            {
                str.append(!processOovWords || !isUnkWord(vocabulary, tok) ? tok : UNK).append(" ");
                totalWords++;
                prevTok = tok;
            }            
            else // we omit a token in the anonymizedSentence so we have to adjust alignment indices in the AnonymizationAlignments list
            {
                adjustAlignmentIndices(totalWords, true, true);
                prevTok = tok;
            }
//            prevTok = tok;
        }
        return str.toString().trim();
    }
    
    private String rejoinHyphenatedWords(String sent, boolean anonymizedSentence)
    {              
        String[] toks = sent.split(" ");        
        StringBuilder str = new StringBuilder(toks[0]);
        String prevTok = toks[0];
        int totalWords = 1;
        for(int i = 1; i < toks.length; i++)
        {
            String tok = toks[i];
            if(tok.equals("@-@") && !(prevTok.contains("_date-entity") || prevTok.matches(FORMAT) || prevTok.matches("[0-9]+")))
            {
                str.append("-");
                if(!anonymizedSentence)
                    addHyphenToNlAlignments(totalWords, "-");
                adjustAlignmentIndices(totalWords, anonymizedSentence, true);
            }            
            else if (prevTok.equals("@-@") && !(tok.contains("_date-entity") || tok.matches(FORMAT) || tok.matches("[0-9]+"))) 
            {
                str.append(tok);
                adjustAlignmentIndices(totalWords, anonymizedSentence, true);
            }
//            else if(tok.equals("@:@") && prevTok.matches("[0-9]+"))
//            {
//                str.append(":");
//                if(!anonymizedSentence)
//                    addHyphenToNlAlignments(totalWords, ":");
//                adjustAlignmentIndices(totalWords, anonymizedSentence, true);
//            }
//            else if (prevTok.equals("@:@") && tok.matches("[0-9]+"))
//            {
//                str.append(tok);
//                adjustAlignmentIndices(totalWords, anonymizedSentence, true);
//            }
            else 
            {
                str.append(" ").append(tok);
                totalWords++;
            }
            prevTok = tok;
        }        
        return str.toString();
    }
    
    private void addHyphenToNlAlignments(int index, String symbol)
    {
        addHyphenToNlAlignments(index, anonymizationAlignments, symbol);
        addHyphenToNlAlignments(index, contentAlignments, symbol);
        addHyphenToNlAlignments(index, jamrAlignments.values(), symbol);
    }
    
    private void addHyphenToNlAlignments(int index, Collection<? extends AmrAnonymizationAlignment> alignments, String symbol)
    {
        for(AmrAnonymizationAlignment al : alignments)
        {            
            int pos = al.relativePosOfHyphenInNl(index);
            if(pos != -1)
            {
                String oldTok = al.nlTokens.get(pos);
                al.nlTokens.set(pos, oldTok + symbol + al.nlTokens.remove(pos + 1));
                al.nlIds.remove(pos + 1);
            }
        }
    }
    
    private void adjustAlignmentIndices(int wordToBeAdjustedIndex, boolean anonymizedSentence, boolean negative)
    {
        adjustAlignmentIndices(wordToBeAdjustedIndex, anonymizationAlignments, anonymizedSentence, negative);
        adjustAlignmentIndices(wordToBeAdjustedIndex, contentAlignments, anonymizedSentence, negative);
        adjustAlignmentIndices(wordToBeAdjustedIndex, jamrAlignments.values(), anonymizedSentence, negative);
    }
    
    private void adjustAlignmentIndices(int wordToBeAdjustedIndex, Collection<? extends AmrAnonymizationAlignment> alignments, boolean anonymizedSentence, boolean negative)
    {        
        alignments.stream().forEach(al -> al.adjust(wordToBeAdjustedIndex, anonymizedSentence, negative));        
    }
    
    private String cleanupAnonymizedSentence(String sentenceIn) 
    {
        StringBuilder str = new StringBuilder();
        for(String tok : sentenceIn.split(" ")) 
        {
            if(tok.length() > 1 && tok.matches("[0-9]+[-.,]?[0-9]+"))
            {
                str.append("num_unk ");
            } 
            else if(tok.length() > 1 && tok.startsWith("£"))
            {
                str.append("currency_unk ");
            }
            else
            {
                str.append(tok).append(" ");
            }
        }
        return str.toString().trim();
    }
    
    private boolean isUnkWord(Indexer<String> vocabulary, String word)
    {
        // check if vocabulary is empty. In this case the vocabulary is probably not populated yet
        // so we cannot say if the word is unknown; the default is false (i.e., not <unk>)
        if(vocabulary.isEmpty() || (vocabulary.size() == 3 && 
                vocabulary.contains(SOS) && vocabulary.contains(EOS) && vocabulary.contains(UNK)))
            return false;
        return !vocabulary.contains(word.toLowerCase());
    }
    
    public void updateVocabularies(Indexer<String> amrVocabulary, HistMap<String> amrHist, 
            Indexer<String> nlVocabulary, HistMap<String> nlHist)
    {
        for(AmrComponent amrToken : ((AmrLinearize)amr).getLinearizedGraph())
        {
//            amrVocabulary.add(amrToken.toString());
//            amrHist.add(amrToken.toString());
            amrVocabulary.add(amrToken.getName()); // exclude senses
            amrHist.add(amrToken.getName());
        }
        for(String word : getAnonymizedSentence().split(" "))
        {
            nlVocabulary.add(word.toLowerCase());
            nlHist.add(word.toLowerCase());
        }
    }
    
    /**
     * 
     * lower capitalize words and add <s> </s> symbols at the beginning and end
     * of the sentence.
     * @return 
    */
//    @Override
//    public String getSentence()
//    {
////        return String.format("%s %s %s", SOS, sentence.toLowerCase(), EOS);
//        return sentence.toLowerCase();
//    }
    
    @Override
    public String getAnonymizedSentence()
    {
//        return String.format("%s %s %s", SOS, sentence.toLowerCase(), EOS);
        return anonymizedSentence.toLowerCase();
    }

     
    public String getSentenceNormalisedIndices(Indexer<String> vocabulary, boolean augmentLexicon)
    {
        StringBuilder str = new StringBuilder();
        str.append(vocabulary.getIndex(SOS));
        for(String word : getAnonymizedSentence().split(" "))
        {
            if(augmentLexicon)
                str.append(" ").append(vocabulary.getIndex(word.toLowerCase()));
            else
            {
                int index = vocabulary.indexOf(word.toLowerCase());
                str.append(" ").append(index != -1 ? index : vocabulary.getIndex(UNK));
            }
        }
        str.append(" " ).append(vocabulary.getIndex(EOS));
        return str.toString();
    }
    
    public String getAmrIndices(Indexer<String> vocabulary, boolean augmentLexicon)
    {
        StringBuilder str = new StringBuilder();
        str.append(vocabulary.getIndex(SOS));
        for(String token : amr.toString().split(" "))
        {
            if(augmentLexicon)
                str.append(" ").append(vocabulary.getIndex(token));
            else
            {
                int index = vocabulary.indexOf(token);
                str.append(" ").append(index != -1 ? index : vocabulary.getIndex(UNK));
            }
        }
        str.append(" " ).append(vocabulary.getIndex(EOS));
        return str.toString().trim();
    }
    
    @Override
    public String getId()
    {
        return id; // incrId
    }

    public Collection<JamrAlignment> getJamrAlignments() {
        return jamrAlignments.values();
    }

    public Collection<AmrAnonymizationAlignment> getContentAlignments() {
        return contentAlignments;
    }

    public Set<AmrAnonymizationAlignment> getAnonymizationAlignments() {
        return anonymizationAlignments;
    }
    
    /**
     * 
     * Output propositions-anonymized nl pairs ..
     * The format is:
     * dummyId\t id\t propositions-string \t   anonymized nl-string   \t weight (not used)
     * @return 
     */    
    public String toStringPropositionsAnonymizedNl()
    {
//        return String.format("0\t%s\t%s\t%s\t0", getId(), ((AmrLinearize)amr).propositionsToString(), getAnonymizedSentence());
        return String.format("%s\t%s\t%s", getId(), ((AmrLinearize)amr).propositionsToString(), getAnonymizedSentence());
    }
    
    /**
     * 
     * Output anonymization aligned pairs
     * The format is:
     * id\t anonymized_pair_1 # anonymized_pair_2...
     * 
     * @param lowerCase
     * @return 
     */
    public String toStringAnonymizationAlignments(boolean lowerCase)
    {
        StringBuilder str = new StringBuilder();
        anonymizationAlignments.stream().forEach(an -> str.append(an).append(" # "));       
        return String.format("%s\t%s", getId(), anonymizationAlignments.isEmpty() ? "" : 
                (lowerCase ? str.substring(0, str.length() - 3) : str.substring(0, str.length() - 3)).trim());        
    }
    
    public String toStringContentAlignments()
    {
        StringBuilder str = new StringBuilder();
        contentAlignments.stream().forEach(an -> str.append(an).append(" # "));
        return String.format("%s\t%s", getId(), contentAlignments.isEmpty() ? "" : str.substring(0, str.length() - 3).trim());
    }
    
    public String toStringJamrAlignments()
    {
        StringBuilder str = new StringBuilder();
        jamrAlignments.values().stream().forEach(an -> str.append(an).append(" # "));
        return String.format("%s\t%s", getId(), jamrAlignments.isEmpty() ? "" : str.substring(0, str.length() - 3).trim());
    }
    
    public String toStringAmrNlStemsPosTags(StemmerPosTagger processor) {
        Pair<String, String> stemsPosTags = processor.processToString(getAnonymizedSentence(), getAnonymizedSentenceSize());
        return String.format("%s\t%s\t%s\t%s\t%s", getId(), amr.toString(), getAnonymizedSentence(), stemsPosTags.getFirst(), stemsPosTags.getSecond());
    }
    
    /**
     * 
     * Output a sequence which is as long as the output anonymized NL sentence, and has in each position the corresponding position of 
     * the aligned AMR token.
     * @param mergedAlignments
     * @return 
     */
    public String toStringAmrNlSeqAlignments(Collection<AmrAnonymizationAlignment> mergedAlignments) {               
        Integer[] sequence = new Integer[anonymizedSentence.split(" ").length];
        Arrays.fill(sequence, -1);        
        Map<String, Integer> nodeIdToAmrPosMap = new HashMap<>();
        mergedAlignments.stream().forEach((AmrAnonymizationAlignment a) -> {
                if(!a.isEmpty() && a.anonymizedNlId < sequence.length)
                    nodeIdToAmrPosMap.put(a.anonymizedToken, a.anonymizedNlId);
        });
        List<AmrComponent> graph = ((AmrLinearize)amr).getLinearizedGraph();
        for(int i = 0; i < graph.size(); i++) {
            AmrComponent node = graph.get(i);
            String nodeId = node.getNodeId();                         
            // either use the unique node id as key (e.g., 0.2.0.1.0), or the token name in case of anonymized nodes            
            int nlId = nodeIdToAmrPosMap.getOrDefault(nodeId.isEmpty() ? node.getName() : nodeId, -1);
            if(node.getName().contains("_name_")) // prefer AmrAnonymizationAlignments to JamrAlignments for NEs
            {
                nlId = nodeIdToAmrPosMap.getOrDefault(node.getName(), nodeIdToAmrPosMap.getOrDefault(nodeId, -1));                
            }
            if(nlId != -1)
                sequence[nlId] = i;
        }
        return id + "\t" + Arrays.asList(sequence).stream().map(String::valueOf).collect(Collectors.joining(" "));        
    }
    
    public String toStringNlOnly()
    {
        return String.format("%s\t%s", getId(), sentence);
    }
    
    public String toStringIdToNumIdOnly()
    {
        return String.format("%s\t%s", id, incrId);
    }
    
    /**
     * 
     * Output amr-nl pairs in flattened format following a linearization strategy.
     * The format is:
     * dummyId\t id\t amr-string \t   anonymized nl-string   \t weight (not used)
     * @return 
     */
    @Override
    public String toString()
    {
        return String.format("%s\t%s\t%s", getId(), amr.toString(), getSentence());
    }
    
    public String toStringBrackets(boolean reshuffleChildren, boolean markLeaves, boolean outputSense, boolean concatBracketsWithRoles)
    {
        return String.format("%s\t%s\t%s", getId(), ((AmrLinearize)amr).toStringBrackets(reshuffleChildren, markLeaves, outputSense, concatBracketsWithRoles), getSentence());
    }
    
    public String toStringAnonymized(boolean outputBrackets, boolean reshuffleChildren, boolean markLeaves, boolean outputSense, boolean concatBracketsWithRoles)
    {
        return String.format("%s\t%s\t%s", getId(), outputBrackets ? ((AmrLinearize)amr).toStringBrackets(reshuffleChildren, markLeaves, outputSense, concatBracketsWithRoles) : amr.toString(), getAnonymizedSentence());        
    }
    
    /**
     * 
     * Output amr-nl pairs in flattened format following a linearization strategy.
     * Words are replaced with their respective unique ids
     * The format is:
     * dummyId\t id\t amr-string \t   nl-string   \t weight (not used)
     * @param amrVocabulary the indexer for amr tokens
     * @param nlVocabulary the indexer for words
     * @param augmentLexicon extend lexicon with new entries (in case of training) or
     * use &lt; UNK &gt; for OOV words
     * @return 
     */
    public String toStringLinearizeIndices(Indexer<String> amrVocabulary, Indexer<String> nlVocabulary, boolean augmentLexicon)
    {
        return String.format("0\t%s\t%s\t%s\t0", getId(), getAmrIndices(amrVocabulary, augmentLexicon), 
        getSentenceNormalisedIndices(nlVocabulary, augmentLexicon));
    }                     
}
