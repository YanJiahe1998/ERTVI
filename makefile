CppCompiler = g++

EigenPath = ./Eigen

Oflag = -O3
Openmp = -fopenmp

AllMod = main.o 



all: $(AllMod)
	$(CppCompiler) $(Oflag) -o  MultiR2.out $(AllMod) -I $(EigenPath) $(Openmp) 
	rm main.o

main.o: main.cpp 
	$(CppCompiler) $(Oflag) -c main.cpp $(Openmp) 