all: halide/cHalide_wrap.cxx halide/_cHalide.so

halide/_cHalide.so: halide/cHalide_wrap.cxx halide/*.py halide/cHalide.i ../src/*.cpp ../src/runtime/*.cpp ../src/*.h ../include/*.h halide/py_util.cpp halide/py_util.h halide/expand_types.h setup.py
	python setup.py build_ext --inplace

halide/cHalide_wrap.cxx: halide/cHalide.i ../src/*.cpp ../src/runtime/*.cpp ../src/*.h ../include/*.h halide/py_util.cpp halide/py_util.h halide/expand_types.h setup.py
	make -C ..
	swig -c++ -python -w362,325,314,389,381,382,361,401,503,509 -I../include halide/cHalide.i

install: all
	python setup.py install

clean:
	rm -rf build
	rm -f halide/cHalide_wrap.cxx
	rm -f halide/halide.so
	rm -f halide/_cHalide.so
	rm -f halide/cHalide.py halide/*.pyc

run_apps:
	PYTHONPATH=.:$(PYTHONPATH) python apps/local_laplacian.py
	PYTHONPATH=.:$(PYTHONPATH) python apps/interpolate.py
	PYTHONPATH=.:$(PYTHONPATH) python apps/bilateral_grid.py
	PYTHONPATH=.:$(PYTHONPATH) python apps/erode.py
	PYTHONPATH=.:$(PYTHONPATH) python apps/blur.py

run_apps_headless:
	PYTHONPATH=.:$(PYTHONPATH) python apps/local_laplacian.py apps/out_local_laplacian.png
	PYTHONPATH=.:$(PYTHONPATH) python apps/interpolate.py apps/out_interpolate.png
	PYTHONPATH=.:$(PYTHONPATH) python apps/bilateral_grid.py apps/out_bilateral_grid.png
	PYTHONPATH=.:$(PYTHONPATH) python apps/erode.py apps/out_erode.png
	PYTHONPATH=.:$(PYTHONPATH) python apps/blur.py apps/out_blur.png

test:
	PYTHONPATH=.:$(PYTHONPATH) python test_halide.py
