STD = -std=c++11
OPT = -O2
FLAGS = -pthread
FLAGS += -fopenmp
# FLAGS += -ffast-math
FLAGS += -I$(HOME)/local/include

main:
	$(CXX) $(STD) $(OPT) $(FLAGS) matrix/nu_critical.cpp -o nu_critical
