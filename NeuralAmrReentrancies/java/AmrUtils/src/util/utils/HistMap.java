package util.utils;

import fig.basic.Pair;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

/**
 *
 * @author konstas
 * @param <T>
 */
public class HistMap<T> implements Serializable
{
    private static final long serialVersionUID = -1;
    private final HashMap<T, Counter> map = new HashMap<>();
    private T topFreqElement;
    
    public void add(T word)
    {
        Counter counter = map.get(word);
        if (counter == null) 
        {
            counter = new Counter(word);
            map.put(word, counter);
        }
        counter.incr();        
    }
   
    public Set<Entry<T, Counter>> getEntries()
    {
        return map.entrySet();
    }
    
    public Set<Entry<T, Integer>> getEntriesFreqs()
    {
        Map<T, Integer> m = new HashMap();
        map.entrySet().stream().forEach((e) -> {
            m.put(e.getKey(), e.getValue().value);
        });
        return m.entrySet();
    }
    
    public Set<T> getKeys()
    {
        return map.keySet();
    }
   
    public int getTotalFrequency()
    {
        int total = 0;
        total = getEntriesFreqs().stream().map((e) -> e.getValue()).reduce(total, Integer::sum);
        return total;
    }
    
    public int size()
    {
        return map.size();
    }
    
    /**
     *
     * Return the most frequent key in the Map.
     * Warning: this can be a slow computation. Always pre-cache.
     * @return 
     */
    public T getFirstKey()
    {
        return getKeysSorted().get(0);
    }
    
    public Pair<T, Integer> getFirstEntry()
    {
        return getEntriesSorted().get(0);
    }
    
    public List<T> getKeysSorted()
    {
        List<Counter> list = new ArrayList<>(map.values());
        Collections.sort(list);
        List<T> out = new ArrayList();
        list.stream().forEach((c) -> {
            out.add(c.key);
        });
        return out;
    }
    
    public List<Pair<T,Integer>> getEntriesSorted()
    {
        List<Counter> list = new ArrayList<>(map.values());
        Collections.sort(list);
        List<Pair<T,Integer>> out = new ArrayList();
        list.stream().forEach((c) -> {
            out.add(new Pair(c.key, c.value));
        });
        return out;
    }
    
    public int getFrequency(T key)
    {
        return map.containsKey(key) ? map.get(key).value : - 1;
    }

    public T getTopFreqElement() {
        return topFreqElement;
    }

    public void setTopFreqElement() {
        this.topFreqElement = getFirstKey();
    }
    
    /**
     * Returns frequency map in decreasing order
     * @return 
     */
    @Override
    public String toString()
    {
        StringBuilder str = new StringBuilder();
        List<Counter> list = new ArrayList<>(map.values());
        Collections.sort(list);
        list.stream().forEach((c) -> {
            str.append(c).append("\n");
        });
//        for(Entry<T, Counter> e : map.entrySet())
//        {
//            str.append(String.format("%s : %s\n", e.getKey(), e.getValue()));
//        }
//        str.delete(str.lastIndexOf(","), str.length());
        return str.toString();
    }
    
    public String toStringOneLine()
    {
        List<Counter> list = new ArrayList<>(map.values());
        Collections.sort(list);
        return list.stream().map(i -> i.toString()).collect(Collectors.joining("\t"));
    }
    
    final class Counter implements Comparable, Serializable
    {
        private static final long serialVersionUID = -1;
        private T key;
        private int value;

        public Counter(T key)
        {
            this.key = key;
        }
        
        public int getValue()
        {
            return value;
        }

        public void incr()
        {
            value++;
        }

        @Override
        public String toString()
        {
//            return String.valueOf(value);
            return String.format("%s : %s", key, value);
        }

        @Override
        public int compareTo(Object o)
        {            
            return ((Counter)o).value - value;            
        }
    }         
}
