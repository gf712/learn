all:
	gcc -O3 -std=c++11 -I ~/ngraph/build/eigen/src/ext_eigen/ -I ~/ngraph/build/mkldnn/include/ mean.cpp eigensolver.cpp main.cpp -lstdc++ -lm -lngraph -lcpu_backend -o pca
