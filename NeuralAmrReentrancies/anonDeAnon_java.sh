#!/bin/bash

CLASSPATH=lib/AmrUtils.jar:lib/Helper.jar:lib/commons-collections4-4.0-alpha1.jar:lib/stanford-corenlp-2017-04-14-build.jar:lib/commons-lang3-3.4.jar
#options are: anonymizeAmrStripped (AMR-generation), anonymizeAmrFull (AMR-generation), deAnonymizeAmr (AMR-parsing), anonymizeText (NL-parsing), deAnonymizeText (AMR-generation)
TYPE=$1
INPUT_IS_FILE=$2
INPUT=$3
java -cp ${CLASSPATH} util.apps.AmrUtils ${TYPE} ${INPUT_IS_FILE} "${INPUT}"

