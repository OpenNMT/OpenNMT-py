package util.metrics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import util.corpus.wrappers.AnonymizationAlignment;
import static util.metrics.GenerationPerformance.METRICS_FORMATTER;
import util.metrics.GenerationPerformance.Metrics;

/**
 *
 * @author ikonstas
 */
public class Hypothesis {

//    private final Pattern FORMATTED_DATE = Pattern.compile("[0-9]{2,4}[-]?[0-9]{2}[-]?[0-9]{2}");
    private static final Pattern anonymizedEntityMatcher = Pattern.compile(".+_[0-9]+$");
    
    private final List<Token> tokens, deanonymizedTokens;
    // Contains the last word added to the hypothesis after de-anonymization
    // For example if the word was 'person_name_0', the list might be 'Alan Turing'.
    private List<String> lastDeAnonymizedWord;    
    private boolean endingWithEOS, startingWithSOS, lastTokenDeAnonymized;    
    private final Map<Metrics, Double> metrics;

    private Hypothesis() {
        this.tokens = new ArrayList<>();
        this.deanonymizedTokens = new ArrayList<>();
        this.lastDeAnonymizedWord = new ArrayList<>();        
        this.metrics = new HashMap<>();
    }
    

    public Hypothesis(String raw) {
        this();
        tokens.addAll(Arrays.stream(raw.split(" ")).map(token -> new Token(-1, token)).collect(Collectors.toList()));               
    }

    public Hypothesis(String raw, String rawAnonymized) {
        this();
        deanonymizedTokens.addAll(Arrays.stream(rawAnonymized.split(" ")).map(token -> new Token(-1, token)).collect(Collectors.toList()));                
        tokens.addAll(Arrays.stream(rawAnonymized.split(" ")).map(token -> new Token(-1, token)).collect(Collectors.toList()));        
    }
        
    public boolean isStartingWithSOS() {
        return startingWithSOS;
    }

    public boolean isEndingWithEOS() {
        return endingWithEOS;
    }

    public List<Token> getTokens() {
        return tokens;
    }

    public double getMetric(Metrics metric) {
        return metrics.getOrDefault(metric, 0.0);
    }

    public void setMetric(Metrics metric, double value) {
        metrics.put(metric, value);
    }

    public Token getLastToken() {
        return tokens.isEmpty() ? Token.emptyToken() : tokens.get(tokens.size() - 1);
    }

    public String getLastWord() {
        return getLastToken().getWord();
    }

    public List<String> getLastDeAnonymizedWord() {
        return lastDeAnonymizedWord;
    }

    public boolean isLastTokenDeAnonymized() {
        return lastTokenDeAnonymized;
    }

    public Token getFirstToken() {
        return tokens.isEmpty() ? Token.emptyToken() : tokens.get(0);
    }    

    public void deAnonymize(Map<String, AnonymizationAlignment> map) {
        if (map == null || map.isEmpty()) {
            deanonymizedTokens.addAll(tokens);
        } else {
            for (int i = 0; i < tokens.size(); i++) {
                Token token = tokens.get(i);
//            tokens.stream().forEach(token -> {
                boolean isAnonymizedFormattedDate = isAnonymizedFormattedDate(token);
                boolean isAnonymizedMonthDate = isAnonymizedMonthDate(token);
                AnonymizationAlignment alignment = isAnonymizedFormattedDate
                        ? map.get(getAnonymizedDateNormalForm(token)) : isAnonymizedMonthDate
                        ? map.get(getAnonymizedMonthDateNormalForm(token)) : map.get(token.getWord());
                if (alignment != null) {
                    if (isAnonymizedFormattedDate) {
                        if (i == 0) {
                            deanonymizedTokens.add(new Token(alignment.getCanonicalizedInputTokens().get(0)));                            
                        }
                        else if (i >= 1 && !isAnonymizedFormattedDate(tokens.get(i - 1))) {
                            deanonymizedTokens.add(new Token(alignment.getCanonicalizedInputTokens().get(0)));
                        }
                    } else if (isAnonymizedMonthDate) {
                        deAnonymizeMonthDate(alignment);
                    } else {
                        List<String> nlTokens = map.get(token.getWord()).getCanonicalizedInputTokens();
                        nlTokens.stream()
                                .forEach(word -> deanonymizedTokens.add(new Token(word)));
                    }
                } else {
                    deanonymizedTokens.add(new Token(token));
                }
            }
        }
    }

    private boolean deAnonymize(Token token, Map<String, AnonymizationAlignment> anonAlignments) {
        lastDeAnonymizedWord.clear();
        if (anonAlignments == null || anonAlignments.isEmpty()) {
            deanonymizedTokens.add(token);
            lastDeAnonymizedWord.add(token.getWord());
            return true;
        } else {
            AnonymizationAlignment alignment = anonAlignments.get(token.getWord());
            if (alignment != null) {
                anonAlignments.get(token.getWord()).getCanonicalizedInputTokens().stream()
                        .forEach(word -> {
                            deanonymizedTokens.add(new Token(word));
                            lastDeAnonymizedWord.add(word);
                        });
                return true;
            } else {
                boolean isAnonymizedFormattedDate = isAnonymizedFormattedDate(token);
                boolean isAnonymizedMonthDate = isAnonymizedMonthDate(token);
                alignment = isAnonymizedFormattedDate
                        ? anonAlignments.get(getAnonymizedDateNormalForm(token)) : isAnonymizedMonthDate
                        ? anonAlignments.get(getAnonymizedMonthDateNormalForm(token)) : null;
                if (alignment != null) {
                    if (isAnonymizedFormattedDate) {
                        if(length() == 1) {
                            String word = deAnonymizeFormattedDate(token, anonAlignments);                            
                            lastDeAnonymizedWord.add(word);
                            return true;
                        }
                        else if (length() > 1 && !isAnonymizedFormattedDate(tokens.get(length() - 2))) {
                            String word = deAnonymizeFormattedDate(token, anonAlignments);                            
                            lastDeAnonymizedWord.add(word);
                            return true;
                        }
                    } else if (isAnonymizedMonthDate) {
                        String word = deAnonymizeMonthDate(alignment);                        
                        lastDeAnonymizedWord.add(word);
                        return true;
                    } else {
                        lastDeAnonymizedWord.add(tokens.get(length() - 2).getWord());
                        return true;
                    }
                } else {
                    deanonymizedTokens.add(token);
                    lastDeAnonymizedWord.add(token.getWord());   
                    // return false if the token contains an anonymized date but 
                    // there is no alignment (useful to override wordLevelHeuristics)
                    return !(isAnonymizedFormattedDate || isAnonymizedEntity(token.getWord()));                    
                }
            }
        }
        return false;
    }

    public void deAnonymizeFake() {
        deanonymizedTokens.addAll(tokens);
    }

    private String deAnonymizeFormattedDate(Token token, Map<String, AnonymizationAlignment> alignments) {
        String[] info = getFormattedDateInfo(token);
        String srcDateEntityTemplate = "_date-entity_" + info[2];
        String year = getDatePartFromSrc("year" + srcDateEntityTemplate, alignments);
        String month = getDatePartFromSrc("month" + srcDateEntityTemplate, alignments);
        String day = getDatePartFromSrc("day" + srcDateEntityTemplate, alignments);
        String word;
        switch (info[1]) {
            case "1":
                word = String.format("%s%s%s", year.substring(2), month, day); break;
            case "2":
            default:
                word = String.format("%s%s%s", year, month, day); break;
            case "3":
                word = String.format("%s-%s-%s", year, month, day);
        }
        deanonymizedTokens.add(new Token(word));
        return word;
    }

    private String getDatePartFromSrc(String part, Map<String, AnonymizationAlignment> alignments) {
        AnonymizationAlignment al = alignments.get(part);
        String value = al == null ? "00" : al.getCanonicalizedInputTokens().get(0);
        return value.length() == 1 ? "0" + value : value;
    }

    private String deAnonymizeMonthDate(AnonymizationAlignment alignment) {
        String word;
        switch (alignment.getCanonicalizedInputTokens().get(0)) {
            case "1":
            default:
                word = "January";
                break;
            case "2":
                word = "Feburary";
                break;
            case "3":
                word = "March";
                break;
            case "4":
                word = "April";
                break;
            case "5":
                word = "May";
                break;
            case "6":
                word = "June";
                break;
            case "7":
                word = "July";
                break;
            case "8":
                word = "August";
                break;
            case "9":
                word = "September";
                break;
            case "10":
                word = "October";
                break;
            case "11":
                word = "November";
                break;
            case "12":
                word = "December";
                break;
        }
        deanonymizedTokens.add(new Token(word));
        return word;
    }

    private boolean isAnonymizedFormattedDate(Token token) {
        return token.getWord().contains("date-entity") && token.getWord().contains("_f");
    }

    private boolean isAnonymizedMonthDate(Token token) {
        return token.getWord().contains("month_name_date-entity");
    }

    private String getAnonymizedDateNormalForm(Token token) {
        int index = token.getWord().indexOf("_f");
        return token.getWord().substring(0, index) + token.getWord().substring(index + 3);
    }

    private String getAnonymizedMonthDateNormalForm(Token token) {
        int index = token.getWord().lastIndexOf("_");
        return "month_date-entity" + token.getWord().substring(index);
    }

    /**
     *
     * Get info from formatted anonymized entity:
     * day|month|year_date-entity_fX_Y
     *
     * @param token
     * @return
     */
    private String[] getFormattedDateInfo(Token token) {
        int index = token.getWord().indexOf("_");
        String part = token.getWord().substring(0, index);
        index = token.getWord().indexOf("f");
        String format = token.getWord().substring(index + 1, index + 2);
        index = token.getWord().lastIndexOf("_");
        String id = token.getWord().substring(index + 1);
        return new String[]{part, format, id};
    }

    public int length() {
        return tokens.size();
    }

    public int deAnonymizedLength() {
        return deanonymizedTokens.size();
    }

    public String metricsToString() {

        return metrics.isEmpty() ? ""
                : String.format("[BLEU score : %s\t"
                        + "BLEU anonymized score : %s\t"
                        + "METEOR score : %s\t"
                        + "METEOR anonymized score : %s\t"
                        + "pos : %s]",
                        METRICS_FORMATTER.format(metrics.get(Metrics.BLEU)),
                        METRICS_FORMATTER.format(metrics.get(Metrics.BLEU_ANONYMIZED)),
                        METRICS_FORMATTER.format(metrics.get(Metrics.METEOR)),
                        METRICS_FORMATTER.format(metrics.get(Metrics.METEOR_ANONYMIZED)),
                        METRICS_FORMATTER.format(metrics.getOrDefault(Metrics.POS, 0d))
                );

    }

    public String tokensToString(List<Token> tokens) {
        StringBuilder str = new StringBuilder();
        tokens.stream().forEach((token) -> {
            str.append(token).append(" ");
        });
        return str.toString().trim();
    }
    
    public String tokensToStringCapitalizeFirstWord(List<Token> tokens) {
        StringBuilder str = new StringBuilder();
        if(tokens.size() > 0)
            tokens.get(0).capitalize();
        tokens.stream().forEach((token) -> {
            str.append(token).append(" ");
        });
        return str.toString().trim();
    }

    public String deAnonymizedToString() {
        return tokensToString(deanonymizedTokens);
    }
    
    public String deAnonymizedCapitalizeFirstWordToString() {
        return tokensToStringCapitalizeFirstWord(deanonymizedTokens);
    }

    public String anonymizedToString() {
        return tokensToString(tokens);
    }
 
    @Override
    public String toString() {
        return deAnonymizedToString();
    }

    public static Hypothesis newEmptyHypothesis() {
        return new Hypothesis();
    }

    public static boolean isAnonymizedEntity(String word) {
        return anonymizedEntityMatcher.matcher(word).matches();
    }
}
