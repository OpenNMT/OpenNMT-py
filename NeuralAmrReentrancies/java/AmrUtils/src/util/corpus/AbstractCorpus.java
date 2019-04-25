package util.corpus;

import util.utils.Database;
import util.utils.Settings;

/**
 *
 * @author Yannis Konstas
 */
public abstract class AbstractCorpus implements Corpus {

    Settings settings;
    Database db;
    String corpus;

    public AbstractCorpus(Settings settings, Database db, String corpus) {
        this.settings = settings;
        this.db = db;
        this.corpus = corpus;
    }

    public Database getDb() {
        return db;
    }

    public Settings getSettings() {
        return settings;
    }    
}
