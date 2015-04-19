#include "Halide.h"

using namespace Halide;

void Blur() {
    const int nChannels = 4;

    ImageParam input8(UInt(8), 3, "input");
    input8.set_stride(0, nChannels)
        .set_stride(1, Halide::Expr())
        .set_stride(2, 1)
        .set_bounds(2, 0, nChannels);  // expecting chunky image

    Var x, y, c;
    Func input;
    input(x, y, c) = input8(clamp(x, input8.left(), input8.right()),
                            clamp(y, input8.top(), input8.bottom()), c);

    Func blur_x("blur_x");
    blur_x(x, y, c) = cast<uint8_t>(
        (input(x, y, c) + input(x + 1, y, c) + input(x + 2, y, c)) / 3);
    blur_x.output_buffer()
        .set_stride(0, nChannels)
        .set_stride(1, Halide::Expr())
        .set_stride(2, 1)
        .set_bounds(2, 0, nChannels);  // expecting chunky image
    blur_x.bound(c, 0, 4);

    Func result("result");
    result(x, y, c) = cast<uint8_t>(
        (blur_x(x, y, c) + blur_x(x, y + 1, c) + blur_x(x, y + 2, c)) / 3);
    result.output_buffer()
        .set_stride(0, nChannels)
        .set_stride(1, Halide::Expr())
        .set_stride(2, 1)
        .set_bounds(2, 0, nChannels);  // expecting chunky image

    result.bound(c, 0, 4);
    result.rs(x, y, c).vectorize(c);

    std::vector<Argument> args;
    args.push_back(input8);
    result.compile_to_file("rs_halide_generated", args);
}

int main(int argc, char **argv) {
    Blur();

    std::cout << "Done!" << std::endl;
}