main: *.cc *.h
	mpicxx -Wall -O2 *.cc -o main -ferror-limit=5
