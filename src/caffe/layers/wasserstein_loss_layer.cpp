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

#define DISTANCE_DATASET_NAME "data"

namespace caffe {
template <typename Dtype>
void WassersteinLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // initialize a uniform distribution u as a state variable
  u0_.ReshapeLike(*bottom[0]);
  Dtype* u = u0_.mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  float c = 1.0 / dim;
  for (int i = 0; i < bottom[0]->count(); ++i) {
      u[i] = Dtype(c);
  }
  CHECK(this->layer_param_.wasserstein_param().has_source())
      << "Distance matrix source must be specified.";
  // For binaryproto
  // BlobProto blob_proto;
  string filename = this->layer_param_.wasserstein_param().source();
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
  
  float lambda = this->layer_param_.wasserstein_param().lambda();
  // Initialize K and KM
  K_.ReshapeLike(distm_);
    caffe_copy(K_.count(), distm_.cpu_data(), K_.mutable_cpu_data());
  caffe_scal(distm_.count(), Dtype(-lambda), K_.mutable_cpu_data());
  caffe_exp(K_.count(), K_.cpu_data(), K_.mutable_cpu_data());

  KM_.ReshapeLike(distm_);
  caffe_mul(K_.count(), K_.cpu_data(), distm_.cpu_data(), KM_.mutable_cpu_data());
}

template <typename Dtype>
void WassersteinLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  // replace with CHECK_EQ(bottom[1]->channels(), 1) OR CHECK_EQ(bottom[1]->channels(), bottom[0]->channels())
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void printVector(const char* name, int count, Dtype* vec) {
  printf("%s\n", name);
  for (int i = 0; i < count; i++) {
    printf("%f, ", float(vec[i]));
  }
  printf("\n");
}

template <typename Dtype>
void WassersteinLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* init_bottom_label = bottom[1]->mutable_cpu_data();

  const Dtype* K = K_.cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  float lambda = this->layer_param_.wasserstein_param().lambda();
  
  Blob<Dtype> label_tmp_;
  label_tmp_.ReshapeLike(u0_);
  Dtype* bottom_label = label_tmp_.mutable_cpu_data();
  if (bottom[1]->channels() == 1) {
    // init_bottom_label is n x 1.    
    for (int i =0; i < count; i++) {
      bottom_label[i] = Dtype(0);
    }
    //printVector(count, bottom_label);
    for (int i = 0; i < num; ++i){
      //printf("%f\n", init_bottom_label[i]);
      int label = static_cast<int>(init_bottom_label[i]);
      bottom_label[i*dim + label] = Dtype(1);
    }  
    //printf("Here\n");
    //printVector(count, bottom_label);
    //printf("Further\n");
  }
  else {
    //printf("Never\n");
    bottom_label = init_bottom_label;  
  }
  //printVector("labels", count, bottom_label);
  Dtype* u = u0_.mutable_cpu_data();
  // make u a uniform distribution
  float c = 1.0 / dim;
  for (int i = 0; i < count; ++i) {
      u[i] = Dtype(c);
  }
  
  Blob<Dtype> tmp_;
  tmp_.ReshapeLike(u0_);
  Dtype* tmp = tmp_.mutable_cpu_data();
  
  uint32_t sinkhorn_iter = this->layer_param_.wasserstein_param().sinkhorn_iter();
  for (int i = 0; i < sinkhorn_iter; i++) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.),
                   u, K, Dtype(0.), tmp);
    caffe_div(count, bottom_label, tmp, tmp);
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, num, dim, dim, Dtype(1.),
                   tmp, K, Dtype(0.), u);
    caffe_div(count, bottom_data, u, u);
  }

  v_.ReshapeLike(u0_);
  Dtype* v = v_.mutable_cpu_data();

  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.),
                 u, K, Dtype(0.), tmp);
  caffe_div(count, bottom_label, tmp, v);

  const Dtype* KM = KM_.cpu_data();
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, dim, Dtype(1.),
                 v, KM, Dtype(0.), tmp);
  caffe_mul(count, u, tmp, tmp);
  
  Dtype loss = 0;
  for (int i =0; i < count; i++) {
    loss += tmp[i];
  }
  //printf("%f\n", loss);
  top[0]->mutable_cpu_data()[0] = loss / num;
  
  // Compute gradient
  alpha_.ReshapeLike(u0_);
  Dtype* alpha = alpha_.mutable_cpu_data();
  caffe_log(count, u, alpha);
  //printVector("alphalogged", count, alpha); 
  caffe_scal(count, Dtype(1.0/(lambda*num)), alpha);
  //printVector("u", count, u);
  //printVector("tmp", count, tmp);
  //printVector("alpha", count, alpha);
}

template <typename Dtype>
void WassersteinLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // use state variable alpha_ to get gradient
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), alpha_.cpu_data(), bottom_diff);
  }
}

INSTANTIATE_CLASS(WassersteinLossLayer);
REGISTER_LAYER_CLASS(WassersteinLoss);

}  // namespace caffe
