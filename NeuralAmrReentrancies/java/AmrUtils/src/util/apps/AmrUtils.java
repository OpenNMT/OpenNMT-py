package util.apps;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.EmptyStackException;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import util.utils.HistMap;
import util.utils.Settings;
import util.corpus.wrappers.AmrAlignment;
import util.corpus.wrappers.AmrAnonymizationAlignment;
import util.corpus.wrappers.AmrLinearize;
import util.corpus.wrappers.AmrLinearizedSentence;
import util.corpus.wrappers.AmrNode;
import util.corpus.wrappers.AnonymizationAlignment;
import util.corpus.wrappers.Dictionaries;
import util.corpus.wrappers.GigawordWrapper;
import util.corpus.wrappers.NerSentence;

/**
 *
 * @author ikonstas
 */
public class AmrUtils {

    private enum Action {
        ANONYMIZE_AMR, DEANONYMIZE_AMR, ANONYMIZE_TEXT, DEANONYMIZE_TEXT
    };
    private final Settings settings;
    private Dictionaries dictionaries;
    private HistMap<Character> vars;
    private final String[] MONTH_NAMES = {"January", "Feburary", "March",
        "April", "May", "June", "July", "August", "September", "October",
        "November", "December"};
    private String errorMessage = "Failed to parse.";

    public AmrUtils() {
        this.settings = new Settings();
        vars = new HistMap<>();
    }

    private void initDictionaries(Action action) {
        switch (action) {
            case ANONYMIZE_AMR:
                dictionaries = new Dictionaries(settings, false, "amr", true);
                break;
            case DEANONYMIZE_AMR:
                dictionaries = new Dictionaries(settings, false, "amr", false);
                break;
            case ANONYMIZE_TEXT:
                dictionaries = new Dictionaries(settings, false, "anonymize", false);
                break;
        }
    }

    private String anonymizeAmr(String input, boolean strippedAmr, boolean isInputFile) {
        if (isInputFile) {
            anonymizeFile(input, false, strippedAmr, null);
        } else {
            return anonymizeSingleAmr(input, strippedAmr);
        }
        return "";
    }

    private String deAnonymizeAmr(String input, boolean isInputFile) {
        if (isInputFile) {
            deAnonymizeFile(input, false);
        } else {
            return expandStrippedToFull(deAnonymizeSingle(input, false), true);
        }
        return "";
    }

    private String anonymizeText(String input, boolean isInputFile) {
        GigawordWrapper anonymizer = new GigawordWrapper(dictionaries.getAmrAnonymizationAlignments(), dictionaries.getNeNlAmrAlignments(), false);
        String out = "";
        if (isInputFile) {
            anonymizeFile(input, true, false, anonymizer);
        } else {
            out = anonymizeSingleText(input, anonymizer);
        }
        anonymizer.terminateNerClient();
        return out;
    }

    private String deAnonymizeText(String input, boolean isInputFile) {
        if (isInputFile) {
            deAnonymizeFile(input, true);
        } else {
            return deAnonymizeSingle(input, true);
        }
        return "";
    }

    private void anonymizeFile(String path, boolean isText, boolean strippedAmr, GigawordWrapper anonymizer) {
        try (PrintWriter anonymizedWriter = new PrintWriter(new FileOutputStream(path + ".anonymized"));
                PrintWriter alignmentsWriter = new PrintWriter(new FileOutputStream(path + ".alignments"));) {
            Files.lines(Paths.get(path)).forEach(line -> {
                String[] anonAligns = (isText ? anonymizeSingleText(line, anonymizer) : anonymizeSingleAmr(line, strippedAmr)).split("#");
                anonymizedWriter.println(anonAligns[0]);
                alignmentsWriter.println(anonAligns.length == 1 ? "" : anonAligns[1]);
            });
        } catch (IOException ex) {

        }
    }

    private String anonymizeSingleAmr(String input, boolean strippedAmr) {
        String id = "";
        AmrLinearizedSentence amrSentence;
        boolean failedToParse;
        if (input.isEmpty()) {
            return " # ";
        }
        if (strippedAmr) {
            input = normalizeAmr(input);
            List<String> tokens = Arrays.asList(input.split(" "));
            tokens = collateNames(tokens);
            if (checkBrackets(tokens) && checkStructure(tokens)) {
                AmrNode root = new AmrNode("", tokens.get(0), "", ":TOP");
                readGraph(tokens, 1, root);
                amrSentence = new AmrLinearizedSentence(id, "", root, dictionaries, Collections.EMPTY_MAP, "dfs");
            } else {
                return "FAILED_TO_PARSE#" + errorMessage;
            }
        } else {
            amrSentence = new AmrLinearizedSentence(id, "", Collections.EMPTY_MAP, "dfs");
            try {
                amrSentence.parseAmr(id, input, dictionaries);
            } catch (EmptyStackException | StringIndexOutOfBoundsException e) {
                return "FAILED_TO_PARSE#" + errorMessage;
            }
        }
        failedToParse = amrSentence.getAmr().isFailedToParse();
        if (!failedToParse) {
            amrSentence.applyAlignmentHeuristics(new HashMap<>(), false, true, false);
            String alignments = alignmentsToString(amrSentence.getAnonymizationAlignments());
            return "rootvar / " +((AmrLinearize) amrSentence.getAmr()).toStringBrackets(false, false, false, false) + "#" + alignments + "#" + toVisJs(((AmrLinearize) amrSentence.getAmr()).getGraph());
        }
        return "FAILED_TO_PARSE#" + errorMessage;
    }

    private String anonymizeSingleText(String input, GigawordWrapper anonymizer) {
        input = normalizePunctuation(input);
        if (input.isEmpty() || input.equals("?")) {
            return " # ";
        }
        NerSentence nerSentence = anonymizer.anonymizeRaw(input);
        normalizeAnonymizedSentence(nerSentence);
        return nerSentence.toStringNlAnonOnly() + "#" + alignmentsToString(nerSentence.getAnonymizationAlignments());
    }

    private void deAnonymizeFile(String path, boolean isText) {
        try (PrintWriter deAnonymizedWriter = new PrintWriter(new FileOutputStream(path + ".pred"))) {
            String anonymizedPath = new File(path + ".pred.anonymized").exists() ? path + ".pred.anonymized" : path + ".anonymized";
            zip(Files.lines(Paths.get(anonymizedPath)),
                    Files.lines(Paths.get(path + ".alignments")), (a, b) -> a + "#" + b).forEach(l -> deAnonymizedWriter.println(isText ? deAnonymizeSingle(l, true) : expandStrippedToFull(deAnonymizeSingle(l, false), false)));
//                                deAnonymizeSingle(l, false)).split("#")[0] ));
        } catch (IOException ex) {

        }
    }

    private String deAnonymizeSingle(String input, boolean isText) {
        String[] inputAligns = input.split("#");
        if(!(inputAligns.length == 0 || inputAligns[0].isEmpty() || inputAligns[0].equals(" "))) {
            if (inputAligns.length > 1) {
                Map<String, AnonymizationAlignment> alignments = getAlignmentsFromString(inputAligns[1]);
                return isText ? deAnonymizeSingleText(inputAligns[0], alignments) : deAnonymizeSingleAmr(inputAligns[0], alignments);
            } else if (isText) {
                return inputAligns[0];
            } else {
                return deAnonymizeSingleAmr(inputAligns[0], Collections.EMPTY_MAP);
            }
//        } else if (isValidAmr(normalizeAmr(inputAligns[0]))) {
//            return normalizeAmr(inputAligns[0]);
        } else {
            return "FAILED_TO_PARSE#" + errorMessage;
        }
    }

    private String deAnonymizeSingleText(String input, Map<String, AnonymizationAlignment> map) {
        List<String> deanonymizedTokens = new ArrayList<>();
        String[] tokens = input.split(" ");
        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];
            boolean isAnonymizedFormattedDate = isAnonymizedFormattedDate(token);
            boolean isAnonymizedMonthDate = isAnonymizedMonthDate(token);
            AnonymizationAlignment alignment = isAnonymizedFormattedDate
                    ? map.get(getAnonymizedDateNormalForm(token)) : isAnonymizedMonthDate
                    ? map.get(getAnonymizedMonthDateNormalForm(token)) : map.get(token);
            if (alignment != null) {
                if (isAnonymizedFormattedDate) {
                    if (i == 0) {
                        deanonymizedTokens.add(alignment.getCanonicalizedInputTokens().get(0));
                    } else if (i >= 1 && !isAnonymizedFormattedDate(tokens[i - 1])) {
                        deanonymizedTokens.add(alignment.getCanonicalizedInputTokens().get(0));
                    }
                } else if (isAnonymizedMonthDate) {
                    deanonymizedTokens.add(deAnonymizeMonthDate(alignment));
                } else {
                    List<String> nlTokens = map.get(token).getCanonicalizedInputTokens();
                    nlTokens.stream().forEach(word -> deanonymizedTokens.add(word));
                }
            } else {
                deanonymizedTokens.add(token);
            }
        }
        return String.join(" ", deanonymizedTokens);
    }

    private String deAnonymizeSingleAmr(String input, Map<String, AnonymizationAlignment> map) {
        input = normalizeAmr(input);
        List<String> tokens = Arrays.asList(input.split(" "));
        if (isValidAmr(input)) {
            return String.join(" ", deAnonymizeAmrTokens(tokens, map));
        } else {
            return "FAILED_TO_PARSE#" + errorMessage;
        }
    }

    private boolean isValidAmr(String input) {
        List<String> tokens = Arrays.asList(input.split(" "));
        return checkBrackets(tokens) && checkStructure(tokens);
    }

    private String expandStrippedToFull(String input, boolean outputVisJs) {
        if (input.isEmpty() || input.equals(" ")) {
            return " # ";
        }
        if (input.startsWith("FAILED_TO_PARSE")) {
            return input;
        }
        vars = new HistMap<>();
        AmrLinearizedSentence amrSentence;
        String id = "";
        List<String> tokens = Arrays.asList(input.split(" "));
        // expand to full AMR
        String rootConcept = tokens.get(0);
        AmrNode root = new AmrNode(getVar(rootConcept), rootConcept, "", ":TOP");
        readGraph(tokens, 1, root);
        amrSentence = new AmrLinearizedSentence(id, "", root, dictionaries, Collections.EMPTY_MAP, "dfs");
        boolean failedToParse = amrSentence.getAmr().isFailedToParse();
        if (!failedToParse) {
            return root.pp().trim() + (outputVisJs ? "#" + toVisJs(((AmrLinearize) amrSentence.getAmr()).getGraph()) : "");
        }
        return "FAILED_TO_PARSE#" + errorMessage;
    }

    private List<String> deAnonymizeAmrTokens(List<String> tokens, Map<String, AnonymizationAlignment> map) {
        List<String> out = new ArrayList<>(tokens.size());
        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i);
            if (token.contains("name_")) {
                out.addAll(processName(token, i == 0 || !tokens.get(i - 1).equals("("), map));
            } else if (token.contains("num_")) {
                if (token.contains("quantity")) {
                    out.add(":quant");
                } else if (token.contains("entity")) {
                    out.add(":value");
                }
//                out.addAll(getAmrValue(token, map));
                out.add(getAmrValue(token, map).get(0).split(" ")[0]);
            } else if (token.contains("date-entity_")) {
                out.add(":" + token.substring(0, token.indexOf("_")));
//                out.addAll(getAmrValue(token, map));
                out.add(getAmrValue(token, map).get(0).split(" ")[0]);
            } else {
                out.add(token);
            }
        }
        return out;
    }

    private List<String> processName(String node, boolean encloseInBrackets, Map<String, AnonymizationAlignment> map) {
        List<String> out = new ArrayList<>();
        String headNode = node.substring(0, node.indexOf("_"));
        if (encloseInBrackets) {
            out.add("(");
        }
        out.add(headNode);
        out.add(":name");
        out.add("(");
        out.add("name");
        String[] value = getAmrValue(node, map).get(0).split(" ");
        for (int i = 1; i <= value.length; i++) {
            out.add(":op" + i);
            out.add("\"" + value[i - 1] + "\"");
        }
        out.add(")");
        if (encloseInBrackets) {
            out.add(")");
        }
        return out;
    }

    private List<String> getAmrValue(String key, Map<String, AnonymizationAlignment> map) {
        AnonymizationAlignment alignment = map.get(key);
        if (alignment != null) {
            return anonymizationAlignmentToString(alignment);
        } else if (key.startsWith("month_date-entity")) { // hack: check whether we have stored the key under month_name
            String num = key.substring(key.lastIndexOf("_") + 1);
            alignment = map.get("month_name_date-entity_" + num);
            if (alignment != null) {
                return anonymizationAlignmentToString(alignment);
            }
        } // try the first closest matching key without the number
        int lastIndex = key.lastIndexOf("_");
        String keyWithoutNum = key.substring(0, lastIndex != -1 ? lastIndex : key.length());
        for (Map.Entry<String, AnonymizationAlignment> entries : map.entrySet()) {
            if (entries.getKey().startsWith(keyWithoutNum)) {
                return anonymizationAlignmentToString(entries.getValue());
            }
        }

        return Arrays.asList("N-A");
    }

    private List<String> anonymizationAlignmentToString(AnonymizationAlignment alignment) {
        List<String> value = new ArrayList<>();
        List<String> nlTokens = alignment.getCanonicalizedInputTokens();
        nlTokens.stream().forEach(word -> value.add(normalizeNode(word)));
        return value;
    }

    private String normalizeNode(String node) {
        return node.replaceAll("/", "-").replaceAll(":", "-").replaceAll("~", "-").replaceAll("\"", "a").replaceAll("\\(", "[");
    }

    private String normalizePunctuation(String input) {
        Charset ascii = Charset.forName("ISO-8859-1");
        input = new String(input.getBytes(ascii), ascii);
        return input.replaceAll("!", ".").replaceAll("#", "-");
    }

    private String normalizeAmr(String input) {
        input = input.replaceAll("\n", " ").replaceAll("\\p{Space}+", " ");
        input = input.replaceAll("\\(", "( ").replaceAll("\\)", " )").replaceAll(" +", " ");
        return input;
    }

    private void normalizeAnonymizedSentence(NerSentence input) {
        input.setAnonymizedSentence(input.getAnonymizedSentence().replaceAll("`", "'").replaceAll("#", "-"));
    }
    
    private Map<String, AnonymizationAlignment> getAlignmentsFromString(String inputAligns) {
        return Stream.of(inputAligns.split("\t")).map(str -> {
            String anonAmr[] = str.split("[|]{3}");
            return new AmrAnonymizationAlignment(anonAmr[0], anonAmr[1]);
        }).collect(Collectors.toMap(AmrAnonymizationAlignment::getAnonymizedToken, Function.identity(), (oldVal, newVal) -> oldVal));
    }

    private String alignmentsToString(Collection<AmrAnonymizationAlignment> alignments) {
        return alignments.stream()
                .map(a -> a.getAnonymizedToken() + "|||" + a.getRawAmrToken())
                .collect(Collectors.joining("\t"));
    }

    private List<String> collateNames(List<String> words) {
        List<String> out = new ArrayList<>(words.size());
        for (int i = 0; i < words.size(); i++) {
            String word = words.get(i);
            out.add(word);
            if (word.equals(":name")) {
                int j;
                List<String> name = new ArrayList<>();
                for (j = i + 1; j < words.size(); j++) {
                    name.add(words.get(j));
                    if (words.get(j).endsWith("\"")) {
                        i = j;
                        out.add("(");
                        out.add("name");
                        out.add(":op1");
                        out.add(String.join("_", name));
                        out.add(")");
                        break;
                    }
                }
            }
        }
        return out;
    }

    private boolean checkBrackets(List<String> words) {
        int openBrackets = 0;
        boolean correct = true;
        for (String word : words) {
            if (isOpenBracket(word)) {
                openBrackets++;
            } else if (isCloseBracket(word)) {
                openBrackets--;
            }
            if (openBrackets < 0) {
                correct = false;
                errorMessage = "More closing than opening brackets.";
                break;
            }
        }
        if (!correct) {
            errorMessage = "Number of opening and closing brackets doesn't match.";
        }
        return correct && openBrackets == 0;
    }

    private boolean checkStructure(List<String> words) {
        boolean correct = true;
        int length = words.size();
        for (int i = 0; i < length; i++) {
            String word = words.get(i);
            if (i < length - 1) {
                String nextWord = words.get(i + 1);
                if (isConcept(word)) {
                    if (isSpecialConcept(word) && isSpecialConcept(nextWord)) {
                        continue;
                    } else if (isConcept(nextWord)) {
                        errorMessage = "Two concepts in a row.";
                        correct = false;
                        break;
                    } else if (isOpenBracket(nextWord)) {
                        errorMessage = "Opening bracket after concept.";
                        correct = false;
                        break;
                    }
                } else if (isRole(word)) {
                    if (i == 0) {
                        errorMessage = "Role at the beginning of the graph.";
                        correct = false;
                        break;
                    } else if (isRole(nextWord)) {
                        errorMessage = "Two roles in a row.";
                        correct = false;
                        break;
                    } else if (isCloseBracket(nextWord)) {
                        errorMessage = "Closing bracket after role.";
                        correct = false;
                        break;
                    }
                } else if (isOpenBracket(word)) {
                    if (!isConcept(nextWord)) {
                        errorMessage = "Not concept after open bracket.";
                        correct = false;
                        break;
                    }
                } else if (isCloseBracket(word)) {
                    if (isConcept(nextWord)) {
                        errorMessage = "Concept after closing bracket (should be closing bracket or role).";
                        correct = false;
                        break;
                    } else if (isOpenBracket(nextWord)) {
                        errorMessage = "Opening bracket after closing bracket (should be a role, or closing bracket instead).";
                        correct = false;
                        break;
                    }
                }
            }
            if (i == length - 1 && isRole(word)) {
                errorMessage = "Role at the end of the graph.";
                correct = false;
                break;
            }
        }
        return correct;
    }

    private int readGraph(List<String> graph, int pos, AmrNode parent) {
        while (pos < graph.size()) {
            String currentToken = graph.get(pos);
            if (isCloseBracket(currentToken)) {
                return pos;
            } else {
                String role = currentToken;
                AmrAlignment roleAlignment = new AmrAlignment(AmrAlignment.TokenType.ROLE, role);
                String nextToken = graph.get(++pos);                
                if (isConcept(nextToken)) {
                    AmrAlignment conceptAlignment = new AmrAlignment(AmrAlignment.TokenType.CONCEPT, nextToken);
                    parent.add(new AmrNode(getVar(nextToken), nextToken, "", role, conceptAlignment, roleAlignment));
                } else if (isOpenBracket(nextToken)) {
                    AmrAlignment conceptAlignment = new AmrAlignment(AmrAlignment.TokenType.CONCEPT, graph.get(pos + 1));
                    AmrNode subTreeParent = new AmrNode(getVar(graph.get(pos + 1)), graph.get(pos + 1), "", role, conceptAlignment, roleAlignment);
                    parent.add(subTreeParent);
                    int nextPos = readGraph(graph, pos + 2, subTreeParent);
                    pos = nextPos;
                } else {
                    errorMessage = "Couldn't convert to full AMR graph.";
                }
            }                            
            pos++;
        }
        return pos;
    }

    private String toVisJs(AmrNode graph) {
        List<String> nodes = new ArrayList<>();
        List<String> edges = new ArrayList<>();
        Map<AmrNode, Integer> ids = new HashMap<>();
        Enumeration<AmrNode> traversal = graph.preorderEnumeration();
        int id = 0;
        while (traversal.hasMoreElements()) {
            AmrNode node = traversal.nextElement();
            int nodeId = ids.getOrDefault(node, ++id);
            if (nodeId == id) {
                ids.put(node, nodeId);
                nodes.add(String.format("{\"id\":%s,\"label\":\"%s\"}", nodeId, node.getWord().replaceAll("\"", "\\\\\"")));
            }
            if (node.getParent() != null) {
                edges.add(String.format("{\"from\":%s,\"to\":%s,\"label\":\"%s\"}", ids.get((AmrNode) node.getParent()), nodeId, node.getRole()));
            }
        }
        return "\"nodes\":[" + String.join(",", nodes) + "],\"edges\":[" + String.join(",", edges) + "]";
    }

    private List<String> getConceptStructure(String node) {
        if (node.equals("-") || node.matches("[0-9]+") || (node.startsWith("\"") && node.endsWith("\""))) {
            return new ArrayList<>(Arrays.asList(new String[]{node}));
        }
        return new ArrayList<>(Arrays.asList(new String[]{getVar(node), "/", node}));
    }

    private String getVar(String token) {
        Character firstChar = token.charAt(0);
        vars.add(firstChar);
        int freq = vars.getFrequency(firstChar);
        return String.valueOf(firstChar) + (freq > 1 ? freq : "");
    }

    private boolean isRole(String word) {
        return word.charAt(0) == ':';
    }

    private boolean isOpenBracket(String word) {
        return word.charAt(0) == '(';
    }

    private boolean isCloseBracket(String word) {
        return word.charAt(0) == ')';
    }

    private boolean isBracket(String word) {
        return isOpenBracket(word) || isCloseBracket(word);
    }

    private boolean isConcept(String word) {
        return !(isRole(word) || isOpenBracket(word) || isCloseBracket(word));
    }

    private boolean isSpecialConcept(String word) {
        return word.contains("date-entity") || word.contains("-quantity") || word.contains("-entity");
    }

    private boolean isAnonymizedFormattedDate(String token) {
        return token.contains("date-entity") && token.contains("_f");
    }

    private boolean isAnonymizedMonthDate(String token) {
        return token.contains("month_name_date-entity");
    }

    private String getAnonymizedDateNormalForm(String token) {
        int index = token.indexOf("_f");
        return token.substring(0, index) + token.substring(index + 3);
    }

    private String getAnonymizedMonthDateNormalForm(String token) {
        int index = token.lastIndexOf("_");
        return "month_date-entity" + token.substring(index);
    }

    private String deAnonymizeMonthDate(AnonymizationAlignment alignment) {
        int month;
        try {
            month = Integer.valueOf(alignment.getCanonicalizedInputTokens().get(0)) - 1;
        } catch (NumberFormatException e) {
            month = 0;
        }
        return MONTH_NAMES[month];
    }

    private static <A, B, C> Stream<C> zip(Stream<? extends A> a, Stream<? extends B> b,
            BiFunction<? super A, ? super B, ? extends C> zipper) {
        final Iterator<? extends A> iteratorA = a.iterator();
        final Iterator<? extends B> iteratorB = b.iterator();
        final Iterable<C> iterable = () -> new Iterator<C>() {
            @Override
            public boolean hasNext() {
                return iteratorA.hasNext() && iteratorB.hasNext();
            }

            @Override
            public C next() {
                return zipper.apply(iteratorA.next(), iteratorB.next());
            }
        };
        return StreamSupport.stream(iterable.spliterator(), a.isParallel() || b.isParallel());
    }

    public static void main(String[] args) {
        String usage = "Usage: java_exec anonymizeAmrStripped|anonymizeAmrFull|deAnonymizeAmr|anonymizeText|deAnonymizeText input_file[true|false] input";
        if (args.length != 3) {
            System.out.println(usage);
            System.exit(1);
        }
        boolean strippedAmr = false;
        Action action = null;
        switch (args[0]) {
            case "anonymizeAmrStripped":
                strippedAmr = true;
                action = Action.ANONYMIZE_AMR;
                break;
            case "anonymizeAmrFull":
                strippedAmr = false;
                action = Action.ANONYMIZE_AMR;
                break;
            case "deAnonymizeText":
                action = Action.DEANONYMIZE_TEXT;
                break;
            case "anonymizeText":
                action = Action.ANONYMIZE_TEXT;
                break;
            case "deAnonymizeAmr":
                action = Action.DEANONYMIZE_AMR;
                break;
            default:
                System.out.println(usage);
                System.exit(1);
        }
        boolean inputIsFile = Boolean.valueOf(args[1]);
        String input = args[2].trim();
        AmrUtils agas = new AmrUtils();
        agas.initDictionaries(action);

        switch (action) {
            case ANONYMIZE_AMR:
                System.out.println(agas.anonymizeAmr(input, strippedAmr, inputIsFile));
                break;
            case DEANONYMIZE_AMR:
                System.out.println(agas.deAnonymizeAmr(input, inputIsFile));
                break;
            case ANONYMIZE_TEXT:
                System.out.println(agas.anonymizeText(input, inputIsFile));
                break;
            case DEANONYMIZE_TEXT:
                System.out.println(agas.deAnonymizeText(input, inputIsFile));
                break;
        }
    }
}
