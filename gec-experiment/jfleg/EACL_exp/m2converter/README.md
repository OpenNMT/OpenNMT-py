# (unofficial) M2 converter 

The m2 converter takes source and reference sentences and create .m2 annotation. (Error types are not specified.)

(N.B. This is not an official script for generating M2 annotation.)


- prerequisites
  - nltk
  - pattern
  - stanford pos tagger
        
            run sh getpostagger.sh


- run 
        
        (e.g.) python m2converter.py -s  source_txt -r reference_dir > reference.m2


- Questions?
  - E-mail to keisuke@cs.jhu.edu
