CXXFLAGS = -O3 -std=c++11 -fopenmp

all: a.out

a.out: main.o mnist.o rwfile.o
	g++ -fopenmp $^

clean:
	rm -f a.out
	rm -f *.o
	rm -f *.user
	rm -f *.gz
	rm -f t10k-images-idx3-ubyte
	rm -f t10k-labels-idx1-ubyte
	rm -f train-images-idx3-ubyte
	rm -f train-labels-idx1-ubyte