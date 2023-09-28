#include <limits>
#include <utility>

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void stacked_batch_chamfer_distance_kernel(
    const torch::Tensor node_features,
    const torch::Tensor indptr,
    torch::Tensor min_distances,
    torch::Tensor flow_idx,
    const torch::Tensor sizes,
    const int num_graphs,
    const int node_dim);

typedef std::pair<torch::Tensor, torch::Tensor> PairTensor;

PairTensor stacked_batch_chamfer_distance_cuda(
    const torch::Tensor node_features, const torch::Tensor indptr, const torch::Tensor sizes)
{
    CHECK_INPUT(node_features);
    CHECK_INPUT(indptr);
    CHECK_INPUT(sizes);

    int num_graphs = sizes.numel();
    int node_dim = node_features.size(1);
    int num_nodes = node_features.size(0);

    torch::Tensor nodewise_min_dist = torch::full(
        { num_nodes, num_graphs },
        std::numeric_limits<float>::infinity(),
        node_features.options());

    torch::Tensor nodewise_min_dist_idx = torch::arange(
        0, num_nodes, indptr.options())
        .unsqueeze(1)
        .repeat({ 1, num_graphs });

    stacked_batch_chamfer_distance_kernel(
        node_features,
        indptr,
        nodewise_min_dist,
        nodewise_min_dist_idx,
        sizes,
        num_graphs,
        node_dim
    );
    return std::make_pair(nodewise_min_dist, nodewise_min_dist_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stacked_batch_chamfer_distance_cuda", &stacked_batch_chamfer_distance_cuda, "Stacked Batch Chamfer Distance (CUDA)");
}