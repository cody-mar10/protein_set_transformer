#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ __forceinline__ void sbcd_kernel(
    const scalar_t* __restrict__ node_features,
    const int* __restrict__ indptr,
    scalar_t* __restrict__ min_distances,
    int* __restrict__ flow_idx,
    const int* __restrict__ sizes,
    const int num_graphs,
    const int node_dim)
{
    // these both range over the number of graphs
    const int x_idx = threadIdx.x;
    const int y_idx = threadIdx.y;

    if (x_idx <= num_graphs && y_idx <= num_graphs && x_idx != y_idx) {
        const int x_size = sizes[x_idx];
        const int y_size = sizes[y_idx];
        const int x_start = indptr[x_idx];
        const int y_start = indptr[y_idx];

        // loop over all nodes in graph x
        for (int i = 0; i < x_size; ++i) {
            const int x_node_offset = (x_start + i) * node_dim;
            const int output_idx = (x_start + i) * num_graphs + y_idx;

            // loop over all nodes in graph y
            for (int j = 0; j < y_size; ++j) {
                const int y_node_offset = (y_start + j) * node_dim;

                // loop over node dimensions
                scalar_t distance = 0.0;
                for (int k = 0; k < node_dim; ++k) {
                    scalar_t x = node_features[x_node_offset + k];
                    scalar_t y = node_features[y_node_offset + k];
                    scalar_t diff = x - y;
                    distance += diff * diff;
                }

                // update min distance
                if (distance < min_distances[output_idx]) {
                    min_distances[output_idx] = distance;
                    // this will make the idx an absolute index
                    // instead of relative to each graph
                    flow_idx[output_idx] = j + y_start;
                }
            }
        }
    }
}


void stacked_batch_chamfer_distance_kernel(
    const torch::Tensor node_features,
    const torch::Tensor indptr,
    torch::Tensor min_distances,
    torch::Tensor flow_idx,
    const torch::Tensor sizes,
    const int num_graphs,
    const int node_dim)
{
    const int num_blocks = 1;
    const dim3 threadsPerBlock(num_graphs, num_graphs);

    AT_DISPATCH_FLOATING_TYPES(min_distances.type(), "stacked_batch_chamfer_distance_cuda", (
        [&] {
            sbcd_kernel<scalar_t> << <num_blocks, threadsPerBlock >> > (
                node_features.data_ptr<scalar_t>(),
                indptr.data_ptr<int>(),
                min_distances.data_ptr<scalar_t>(),
                flow_idx.data_ptr<int>(),
                sizes.data_ptr<int>(),
                num_graphs,
                node_dim);
        }));
}