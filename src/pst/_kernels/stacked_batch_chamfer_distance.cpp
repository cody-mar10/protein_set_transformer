#include <torch/extension.h>
#include <tuple>
#include <vector>
#include <limits>

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>> stacked_batch_chamfer_distance_cpu(const torch::Tensor& point_sets, const torch::Tensor& set_ptr)
{
    // set_ptr[i] points to the beginning index for set i
    // set_ptr[i+1] points to the ending index for set i
    int num_sets = set_ptr.size(0) - 1;
    auto set_ptr_acc = set_ptr.accessor<long, 1>();

    float inf = std::numeric_limits<float>::infinity();
    torch::Tensor chamfer_dist = torch::full({ num_sets, num_sets }, inf, point_sets.options());
    std::vector<torch::Tensor> x_indices;
    std::vector<torch::Tensor> y_indices;

    // precompute the squared euclidean distance between all items within stacked point_sets tensor
    torch::Tensor pairwise_dist = torch::cdist(point_sets, point_sets).pow(2);

    // TODO: keeping track of min indices
    for (int set_idx_x = 0; set_idx_x < num_sets; set_idx_x++)
    {
        int start_x = set_ptr_acc[set_idx_x];
        int end_x = set_ptr_acc[set_idx_x + 1];

        for (int set_idx_y = set_idx_x + 1; set_idx_y < num_sets; set_idx_y++)
        {
            int start_y = set_ptr_acc[set_idx_y];
            int end_y = set_ptr_acc[set_idx_y + 1];
            auto x_slice = torch::indexing::Slice(start_x, end_x);
            auto y_slice = torch::indexing::Slice(start_y, end_y);
            torch::Tensor block = pairwise_dist.index({ x_slice, y_slice });

            auto x_min_data = block.min(0);
            torch::Tensor x_mins = std::get<0>(x_min_data);
            torch::Tensor x_idx = std::get<1>(x_min_data);

            auto y_min_data = block.min(1);
            torch::Tensor y_mins = std::get<0>(y_min_data);
            torch::Tensor y_idx = std::get<1>(y_min_data);

            float cdist_x = x_mins.mean().item<float>();
            float cdist_y = y_mins.mean().item<float>();
            float set_wise_dist = cdist_x + cdist_y;
            chamfer_dist[set_idx_x][set_idx_y] = set_wise_dist;
            chamfer_dist[set_idx_y][set_idx_x] = set_wise_dist;

            x_indices.push_back(x_idx);
            y_indices.push_back(y_idx);
        }
    }
    return { chamfer_dist, x_indices, y_indices };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stacked_batch_chamfer_distance_cpu", &stacked_batch_chamfer_distance_cpu, "Stacked Batch Chamfer distance (CPU)");
}
