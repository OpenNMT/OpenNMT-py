package util.metrics;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 *
 * Convenience class that keeps a word and its corresponding index in an external vocabulary.
 * The user has to do the house-keeping for the vocabulary.
 * @author ikonstas
 */
public class Token {

    private final int index;
    private String word;
    private final Set<String> EOS_CHARACTERS = new HashSet<>(Arrays.asList(new String[] {".", "?", "!"}));
    
    public Token(int index, String word) {
        this.index = index;
        this.word = word;
    }

    public Token(String word) {
        this(-1, word);
    }
    
    public Token(Token tokenIn) {
        this(tokenIn.index, tokenIn.word);
    }
    
    public int getIndex() {
        return index;
    }

    public String getWord() {
        return word;
    }

    public boolean isEOSCharacter() {
        return EOS_CHARACTERS.contains(word);
    }
    
    public void capitalize() {
        word = Character.toUpperCase(word.charAt(0)) + word.substring(1);
    }
    
    @Override
    public String toString() {
        return word;
    }

    @Override
    public boolean equals(Object obj) {
        assert obj instanceof Token;
        return ((Token) obj).index == this.index;
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 43 * hash + this.index;
        return hash;
    }

    public static Token emptyToken() {
        return new Token(-1, "");
    }
}
