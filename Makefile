STD = -std=c++11

OPT = -O2

FLAGS = -pthread
FLAGS += -fopenmp
# FLAGS += -ffast-math
FLAGS += -I$(HOME)/local/include


default: cuda

main:
	$(CXX) $(STD) $(OPT) $(FLAGS) matrix/nu_critical.cpp -o nu_critical


cuda:
	nvcc $(OPT) matrix/nu_critical.cu -o nu_critical_gpu
