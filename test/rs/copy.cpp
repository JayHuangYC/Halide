#include "Halide.h"

using namespace Halide;
using namespace Halide::Internal;

// This visitor takes a string snapshot of all Pipeline IR nodes.
class SnapshotPipeline : public IRMutator {
    void visit(const Pipeline *op) {
        pipeline << op->produce;
    }
  public:
    std::ostringstream pipeline;
};

Image<uint8_t> make_interleaved_image(uint8_t *host, int W, int H) {
    buffer_t buf = {0};
    buf.host = host;
    buf.extent[0] = W;
    buf.stride[0] = 4;
    buf.extent[1] = H;
    buf.stride[1] = buf.stride[0] * buf.extent[0];
    buf.extent[2] = 4;
    buf.stride[2] = 1;
    buf.elem_size = 1;
    return Image<uint8_t>(&buf);
}

Image<uint8_t> make_planar_image(uint8_t *host, int W, int H) {
    buffer_t buf = {0};
    buf.host = host;
    buf.extent[0] = W;
    buf.stride[0] = 1;
    buf.extent[1] = H;
    buf.stride[1] = buf.stride[0] * buf.extent[0];
    buf.extent[2] = 4;
    buf.stride[2] = buf.stride[1] * buf.extent[1];
    buf.elem_size = 1;
    return Image<uint8_t>(&buf);
}

std::string copy_interleaved(bool isVectorized = false) {
    const int nChannels = 4;

    ImageParam input8(UInt(8), 3, "input");
    input8.set_stride(0, nChannels)
        .set_stride(1, Halide::Expr())
        .set_stride(2, 1)
        .set_bounds(2, 0, nChannels);  // expecting interleaved image
    uint8_t in_buf[128*128*4];
    uint8_t out_buf[128*128*4];
    Image<uint8_t> in = make_interleaved_image(in_buf, 128, 128);
    Image<uint8_t> out = make_interleaved_image(out_buf, 128, 128);
    // Image<uint8_t> in = make_planar_image(in_buf, 128, 128);
    // Image<uint8_t> out = make_planar_image(out_buf, 128, 128);
    input8.set(in);

    Var x, y, c;
    Func result("result");
    result(x, y, c) = input8(x, y, c);
    result.output_buffer()
        .set_stride(0, nChannels)
        .set_stride(1, Halide::Expr())
        .set_stride(2, 1)
        .set_bounds(2, 0, nChannels);  // expecting interleaved image

    result.bound(c, 0, 4);
    result.rs(x, y, c);
    if (isVectorized) result.vectorize(c);

    auto p = new SnapshotPipeline();
    result.add_custom_lowering_pass(p);
    result.realize(out);

    return p->pipeline.str();
}

std::string copy_interleaved_vectorized() {
    return copy_interleaved(true);
}

int main(int argc, char **argv) {
    std::string expected_vectorized_ir = 
        R"|(let copy_to_device_result$2 = halide_copy_to_device(result.buffer, halide_rs_device_interface())
assert((copy_to_device_result$2 == 0), copy_to_device_result$2)
let copy_to_device_result = halide_copy_to_device(input.buffer, halide_rs_device_interface())
assert((copy_to_device_result == 0), copy_to_device_result)
parallel<RS> (result.s0.y.__block_id_y, 0, result.extent.1) {
  parallel<RS> (result.s0.x.__block_id_x, 0, result.extent.0) {
    allocate __shared[uint8 * 0]
    parallel<RS> (.__thread_id_x, 0, 1) {
      image_store(x4("result"), x4(result.buffer), x4((result.s0.x.__block_id_x + result.min.0)), x4((result.s0.y.__block_id_y + result.min.1)), ramp(0, 1, 4), image_load(x4("input"), x4(input.buffer), x4(((result.s0.x.__block_id_x + result.min.0) - input.min.0)), x4(input.extent.0), x4(((result.s0.y.__block_id_y + result.min.1) - input.min.1)), x4(input.extent.1), ramp(0, 1, 4), x4(4)))
    }
    free __shared
  }
}
set_dev_dirty(result.buffer, uint8(1))
)|";
    std::string pipeline_ir = copy_interleaved_vectorized();
    if (expected_vectorized_ir != pipeline_ir) {
        std::cout << "FAIL: Expected vectorized output:\n"
                  << expected_vectorized_ir
                  << "Actual output:\n" << pipeline_ir;
        return 1;
    }

    std::string expected_ir =
        R"|(let copy_to_device_result$5 = halide_copy_to_device(result$2.buffer, halide_rs_device_interface())
assert((copy_to_device_result$5 == 0), copy_to_device_result$5)
let copy_to_device_result$4 = halide_copy_to_device(input.buffer, halide_rs_device_interface())
assert((copy_to_device_result$4 == 0), copy_to_device_result$4)
parallel<RS> (result$2.s0.y$2.__block_id_y, 0, result$2.extent.1) {
  parallel<RS> (result$2.s0.x$2.__block_id_x, 0, result$2.extent.0) {
    allocate __shared[uint8 * 0]
    parallel<RS> (.__thread_id_x, 0, 1) {
      for<RS> (result$2.s0.c$2, 0, 4) {
        image_store("result$2", result$2.buffer, (result$2.s0.x$2.__block_id_x + result$2.min.0), (result$2.s0.y$2.__block_id_y + result$2.min.1), result$2.s0.c$2, image_load("input", input.buffer, ((result$2.s0.x$2.__block_id_x + result$2.min.0) - input.min.0), input.extent.0, ((result$2.s0.y$2.__block_id_y + result$2.min.1) - input.min.1), input.extent.1, result$2.s0.c$2, 4))
      }
    }
    free __shared
  }
}
set_dev_dirty(result$2.buffer, uint8(1))
)|";
    pipeline_ir = copy_interleaved();
    if (expected_ir != pipeline_ir) {
        std::cout << "FAIL: Expected output:\n" << expected_ir
                  << "Actual output:\n" << pipeline_ir;
        return 2;
    }

    std::cout << "Done!" << std::endl;
    return 0;
}