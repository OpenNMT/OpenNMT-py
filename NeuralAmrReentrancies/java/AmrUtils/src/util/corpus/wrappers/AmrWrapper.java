package util.corpus.wrappers;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 *
 * @author sinantie
 */
public class AmrWrapper {

    protected final List<AmrSentence> amrSentences = new ArrayList();
    private final boolean isLinearize;
    private final Map<String, Map<String, JamrAlignment>> jamrAlignments;
    protected final Dictionaries dictionaries;
    protected final String linearizeType;
    protected final String[] lines;

    public AmrWrapper(String linearizeType, boolean isLinearize, Map<String, Map<String, JamrAlignment>> jamrAlignments,
            Dictionaries dictionaries, String[] lines) {
        this.linearizeType = linearizeType;
        this.isLinearize = isLinearize;
        this.jamrAlignments = jamrAlignments;
        this.dictionaries = dictionaries;
        this.lines = lines;
    }

    public List<AmrSentence> getAmrSentences() {
        return amrSentences;
    }

    public void parse() {
        // first line is a preamble, and second is empty: ignore
        for (int i = 2; i < lines.length; i++) {
            String[] tokens = lines[i].substring(2).split("::");
            if (tokens[1].startsWith("id")) {
                String id = tokens[1].substring(2).trim();
                if (lines[++i].substring(2).startsWith("::snt") || lines[i].substring(2).startsWith("::tok")) {
                    // check if sentence is already tokenized (in case we are reading from pre-aligned amr inputm where sentence
                    // are tokenized by default)
                    String sentence = normalize(lines[i].substring(2).startsWith("::snt")
                            ? tokenize(lines[i].substring(8)) : lines[i].substring(8));
                    i++;
                    if (lines[i].startsWith("# ::zh")) {
                        i++; // skip Chinese sentence if any
                    }//                    if(lines[i].startsWith("# ::alignments"))
//                        i++; // skip alignments if any
                    i++; // skip next line
                    StringBuilder str = new StringBuilder();
                    while (i < lines.length && !lines[i].equals("")) {
                        str.append(lines[i++]).append("\n");
                    }
//                    if(!str.toString().contains("multi-sentence")) // rule (Nathan Schneider): skip multi-sentence AMR graphs entirely!
                    {
                        AmrSentence candSent = isLinearize
                                ? new AmrLinearizedSentence(id, sentence, jamrAlignments.getOrDefault(id, Collections.EMPTY_MAP), linearizeType)
                                : new AmrSentence(id, sentence);
                        candSent.parseAmr(id, str.toString(), dictionaries);
                        if (!candSent.getAmr().isEmpty()) {
                            amrSentences.add(candSent);
                        }
                    }
                } else {
                    System.err.println("Could not find sentence!");
                }
            } else {
                System.err.println("Could not find id!");
            }
        }
    }

    public static String tokenize(String input) {
        StringBuilder str = new StringBuilder();
        PTBTokenizer<CoreLabel> ptbt = new PTBTokenizer(new StringReader(input), new CoreLabelTokenFactory(), "ptb3Escaping=false");
        ptbt.tokenize().stream().forEach(label -> {
            if (!label.toString().equals("\"") && !label.toString().equals("\\")) {
                str.append(label).append(" ");
            }
        });
        return str.toString().trim();
    }

    public static String normalize(String input) {
        StringBuilder str = new StringBuilder();
        String[] toks = Arrays.asList(input.split(" ")).stream().filter(token -> !token.equals("")).collect(Collectors.toList()).toArray(new String[0]);
        int i = 0;
        for (String tok : toks) {
            // defeciencies such as 'blah blah no.' 
            if (i == toks.length - 1 && tok.endsWith(".") && tok.length() > 1) {
                int index = tok.indexOf(".");
                if (index == tok.lastIndexOf(".")) // but not 'U.N.'
                {
                    str.append(tok.substring(0, index)).append(" ").append(tok.substring(index));
                } else {
                    index = tok.lastIndexOf(".."); // rare but possible: ...happened in the U.S..
                    if (index != -1) {
                        str.append(tok.substring(0, index)).append(". .");
                    } else {
                        str.append(tok).append(" ");
                    }
                }
            } else if (tok.matches("[a-zA-z0-9]+[.]{2,}+")) {
                int index = tok.indexOf(".");
                str.append(tok.substring(0, index)).append(" ... ");
            } else if (tok.matches("[.]{2,}+[a-zA-z0-9]+")) {
                int index = tok.indexOf(".");
                str.append("... ").append(tok.substring(0, index));
            } else if (tok.matches("[a-zA-z0-9\"]+[.]{2,}+[a-zA-z0-9)?\"]+")) {
                int index = tok.indexOf(".");
                str.append(tok.substring(0, index)).append(" ... ").append(tok.substring(tok.lastIndexOf(".") + 1));
            } else if (tok.contains("/") && !(tok.equals("@/@")) && !(tok.contains("http") || tok.contains("www") || tok.contains(".com") || tok.contains(".uk")) && tok.length() > 2) // exclude emoticons
            {
                String[] ar = tok.split("/");
                for (int k = 0; k < ar.length - 1; k++) {
                    str.append(ar[k]).append(" @/@ ");
                }
                str.append(ar[ar.length - 1]).append(" ");
            } else {
                switch (tok) {
                    case "'":
                        str.append("\" ");
                        break;
                    case ".'":
                        str.append("\" . ");
                        break;
                    case "'.":
                        str.append("\" . ");
                        break;
                    case "';":
                        str.append("\" ; ");
                        break;
                    case "\".":
                        str.append("\" . ");
                        break;
                    case ".\"":
                        str.append(". \" ");
                        break;
                    case "',":
                        str.append("\" , ");
                        break;
                    case "\",":
                        str.append("\" , ");
                        break;
                    case "\'\'":
                        str.append("\" ");
                        break;
                    case "“":
                        str.append("\" ");
                        break;
                    case "”":
                        str.append("\" ");
                        break;
                    case "``":
                        str.append("\" ");
                        break;
                    case "\"...":
                        str.append("\" ... ");
                        break;
                    case "/":
                        str.append("@/@ ");
                        break;
                    case "\\\\":
                    case "#":
                        break;
                    default:
                        str.append(tok.replaceAll("\\\\", "").replaceAll("#", "").replaceAll("\'\'", "")).append(" ");
                }
            }
            i++;
        }
        return str.toString().trim();
//        return input.replaceAll(" ' ", " \" ").replaceAll("[.]\"", "\" .").replaceAll("\\\\", "").replaceAll("#", "").trim();
    }
}
