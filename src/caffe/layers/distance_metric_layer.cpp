#include <functional>
#include <utility>
#include <vector>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"
#include "caffe/filler.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

#define DISTANCE_DATASET_NAME "data"

namespace caffe {

template <typename Dtype>
void DistanceMetricLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    CHECK(this->layer_param_.distance_metric_param().has_source())
    << "Distance matrix source must be specified.";
    string filename = this->layer_param_.distance_metric_param().source();
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        LOG(FATAL) << "Failed opening HDF5 file: " << filename;
    }
    
    const int MIN_DATA_DIM = 1;
    const int MAX_DATA_DIM = INT_MAX;
    hdf5_load_nd_dataset(file_id, DISTANCE_DATASET_NAME,
                         MIN_DATA_DIM, MAX_DATA_DIM, &distm_);
    herr_t status = H5Fclose(file_id);
    CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
}

template <typename Dtype>
void DistanceMetricLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.distance_metric_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  int num_labels = bottom[0]->count(label_axis_);
  CHECK_EQ(distm_.num(), num_labels) << "Wrong dimensions of distance matrix";
  CHECK_EQ(distm_.count(), num_labels *  num_labels) << "Wrong dimensions of distance matrix";
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // DistanceMetric is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DistanceMetricLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype distance_metric = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* distm = distm_.cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);

      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);

      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + 1,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
        
      distance_metric += distm[bottom_data_vector[0].second * num_labels + label_value];
      ++count;
    }
  }

  // LOG(INFO) << "DistanceMetric: " << distance_metric;
  top[0]->mutable_cpu_data()[0] = distance_metric / count;
  
  // DistanceMetric layer should not be used as a loss function.
}

INSTANTIATE_CLASS(DistanceMetricLayer);
REGISTER_LAYER_CLASS(DistanceMetric);

}  // namespace caffe
