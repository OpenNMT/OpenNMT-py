#!/bin/bash

cd java/AmrUtils
ant clean
ant jar
cp dist/AmrUtils.jar ../../lib/
cd ../..
