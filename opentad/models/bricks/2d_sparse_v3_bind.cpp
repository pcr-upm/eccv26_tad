#include <torch/extension.h>
torch::Tensor launch_fwd(torch::Tensor, torch::Tensor, torch::Tensor, int, int);
torch::Tensor launch_bwd_in(torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);
torch::Tensor launch_bwd_w(torch::Tensor, torch::Tensor, torch::Tensor, int, int);
torch::Tensor implicit_gemm_fwd_v2(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int);
torch::Tensor implicit_gemm_fwd_v2_chunked(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &launch_fwd, "Forward");
    m.def("bwd_in", &launch_bwd_in, "Backward Input");
    m.def("bwd_w", &launch_bwd_w, "Backward Weight");
    m.def("implicit_gemm_fwd_v2", &implicit_gemm_fwd_v2, "Implicit GEMM Forward V2 (Fused Rulebook)");
    m.def("implicit_gemm_fwd_v2_chunked", &implicit_gemm_fwd_v2_chunked, "Implicit GEMM Forward V2 Chunked (Memory Efficient)");
}
