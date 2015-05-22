all: DB1 DB2 DB3 DB4 DB5 DB6 DB7 DB8 DB9 DB10

DB1 DB2 DB3 DB4 DB5 DB6 DB7 DB8 DB9 DB10: test test.cpp test.py
	./test $@a $@m
	python test.py -p $@a
	python test.py -p $@m

test: test.cpp
	g++ -std=c++11 -o $@ test.cpp

clean:
	-rm test *.dat

.PHONY: clean all DB1 DB2 DB3 DB4 DB5 DB6 DB7 DB8 DB9 DB10
