#include <Halide.h>
#include <stdio.h>

using std::vector;

using namespace Halide;

int main(int argc, char **argv) {
    ImageParam input(Float(32), 2);

    Var x, y;

    Func f;
    f(x, y) = input(x, y) * 2;

    f.gpu_tile(x, y, 8, 8);
    f.compile_to_file("opencl_multiqueue", input,
        get_target_from_environment().with_feature(Target::UserContext).with_feature(Target::OpenCL));
    return 0;
}
