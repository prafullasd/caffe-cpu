#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"
#include "caffe/filler.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

#define DISTANCE_DATASET_NAME "Distance"

namespace caffe {
template <typename Dtype>
void MyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // initialize a uniform distribution u as a state variable
  u0_.ReshapeLike(*bottom[0]);
  float len = bottom[0]->channels();
  float c = 1.0 / len;
  for (int i = 0; i < len; ++i) {
    u0_.mutable_cpu_data()[i] = Dtype(c);
  }
  CHECK(this->layer_param_.my_param().has_source())
      << "Distance matrix source must be specified.";
  // For binaryproto
  // BlobProto blob_proto;
  string filename = this->layer_param_.my_param().source();
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }
  
  //If have to do using binaryproto
  //ReadProtoFromBinaryFile(filename, &blob_proto);
  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;
  hdf5_load_nd_dataset(file_id, DISTANCE_DATASET_NAME,
                       MIN_DATA_DIM, MAX_DATA_DIM, &distm_);
  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
  //DLOG(INFO) << "Successully loaded " << blob_proto->shape(0) << " rows";
  // For binaryproto
  //distm_.FromProto(blob_proto);


}

template <typename Dtype>
void MyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void MyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* distm = distm_.cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  // u = state variable u0_
  // while now converged, update u
  // calculate v_ from u
  // calculate loss from v_ and distm
  // save state variable alpha_ as gradient
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = std::max(
        bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    loss -= log(prob);
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void MyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // use state variable alpha_ to get gradient
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    for (int i = 0; i < num; ++i) {
      int label = static_cast<int>(bottom_label[i]);
      Dtype prob = std::max(
          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
      bottom_diff[i * dim + label] = scale / prob;
    }
  }
}

INSTANTIATE_CLASS(MyLossLayer);
REGISTER_LAYER_CLASS(MyLoss);

}  // namespace caffe
