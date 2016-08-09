#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/mil_data_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/hdf5.hpp" // for hdf5_load_nd_dataset

namespace caffe {

template <typename Dtype>
MILDataLayer<Dtype>::~MILDataLayer<Dtype>() {
  this->StopInternalThread();
}

/*
 *bottom: vector of blobs
 *top: vector of blobs
 */
template <typename Dtype>
void MILDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // LayerSetUp runs through the image name file and stores them.
  string label_file = this->layer_param_.mil_data_param().label_file().c_str();
  
  LOG(INFO) << "Loading labels from: "<< label_file;
  label_file_id_ = H5Fopen(label_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT); 

  CHECK_GE(label_file_id_, 0)<<"hdf5 label file not found: "<<label_file;

  LOG(INFO) << "MIL Data layer:" << std::endl;
  std::ifstream infile(this->layer_param_.mil_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.mil_data_param().source() << std::endl;

  const int channels = this->layer_param_.mil_data_param().channels();
  const int img_size = this->transform_param_.crop_size();
  const int scale = this->transform_param_.scale();
  const int images_per_batch = this->layer_param_.mil_data_param().images_per_batch();
  const int n_classes = this->layer_param_.mil_data_param().n_classes();
  const int num_scales = this->layer_param_.mil_data_param().num_scales();
  const float scale_factor = this->layer_param_.mil_data_param().scale_factor();
  mean_value_.clear();
  // copy transform_param_ content to mean_value_
  std::copy(this->transform_param_.mean_value().begin(),
      this->transform_param_.mean_value().end(),
      std::back_inserter(mean_value_));
  // if mean_value_.size() == 0, then we manually set the mean_value_ to be a vector of size channels
  // and each value to be 128.0
  if(mean_value_.size() == 0)
    // can replace below with code:
    //    mean_value_ = vector<float>(channels, 128)
    for(int i = 0; i < channels; i++)
      mean_value_.push_back((float)128.0);

  CHECK_EQ(mean_value_.size(), channels);

  LOG(INFO) << "MIL Data Layer: "<< "channels: " << channels;
  LOG(INFO) << "MIL Data Layer: "<< "img_size: " << img_size;
  LOG(INFO) << "MIL Data Layer: "<< "scale: " << scale;
  LOG(INFO) << "MIL Data Layer: "<< "n_classes: " << n_classes;
  LOG(INFO) << "MIL Data Layer: "<< "num_scales: " << num_scales; //by default, num_scales are 1(see src/proto/caffe_proto)
  LOG(INFO) << "MIL Data Layer: "<< "scale_factor: " << scale_factor;
  LOG(INFO) << "MIL Data Layer: "<< "images_per_batch: " << images_per_batch;
  for(int i = 0; i < mean_value_.size(); i++)
    LOG(INFO) << "MIL Data Layer: "<< "mean_value[" << i << "]: " << mean_value_[i];
  
  const bool prefetch_needs_rand =
      this->transform_param_.mirror() ||
      this->transform_param_.crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  
  string im_name;
  int count = 0; 
  while (infile >> im_name) {
    string full_im_name = string(this->layer_param_.mil_data_param().root_dir() + "/" + im_name + "." + this->layer_param_.mil_data_param().ext());
    image_database_.push_back(std::make_pair(im_name, full_im_name));
    count = count+1;
    // snapshot every 1000 images
    if(count % 1000 == 0)
      LOG(INFO) << "num: " << count << ", image_name: " << im_name;
  }
  num_images_ = count;
  LOG(INFO) << "Number of images: " << count;
    
  // data blob
  top[0]->Reshape(images_per_batch*num_scales, channels, img_size, img_size);
  for(int i = 0; i < this->PREFETCH_COUNT;++i) {
    this->prefetch_[i].data_.Reshape(images_per_batch*num_scales, channels, img_size, img_size);
  }
  // this->prefetch.data_.Reshape(images_per_batch*num_scales, channels, img_size, img_size);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  
  // label blob
  top[1]->Reshape(images_per_batch, n_classes, 1, 1);

  for(int i = 0; i < this->PREFETCH_COUNT;++i) {
    this->prefetch_[i].label_.Reshape(images_per_batch, n_classes, 1, 1);
  }
  // this->prefetch.label_.Reshape(images_per_batch, n_classes, 1, 1);
  /*
  for(auto& prefetch:this->prefetch_) {
    prefetch.data_.Reshape(top[0]->shape());
    prefetch.label_.Reshape(top[1]->shape());
  }
  */

  this->counter_ = 0;
}

template <typename Dtype>
unsigned int MILDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


// resize an image but keep aspect ratio:
//   origin_img: width_old * height_old -> resize the longest image side to new_size, but keep the aspect ratio 
//      if width_old > height_old, r = height_old/width_old, new_img: new_size * (r * new_size)
//      if width_old < height_old, r = width_old/height_old, new_img: (r * new_size) * new_size
//      (note that we always keep r <= 1)
cv::Mat Transform_IDL(cv::Mat cv_img, int img_size, bool do_mirror) {
    cv::Size cv_size; //(width, height): cv_img.rows = cv.height, cv_img.cols = cv.width
    if (cv_img.cols > cv_img.rows){
        cv::Size tmp(img_size, round(cv_img.rows*img_size / cv_img.cols)); // new_cols=img_size, new_rows=img_size * rows/colsi(rows<cols), after processing, cols are still less than rows
        cv_size = tmp;
    }
    else { // cols <= rows 
        cv::Size tmp(round(cv_img.cols*img_size / cv_img.rows), img_size); // after processing, cols <= rows 
        cv_size = tmp;
    }

    cv::Mat cv_resized_img;
    cv::resize(cv_img, cv_resized_img, cv_size, 0, 0, cv::INTER_LINEAR);
    
    // horizontal flip at random 
    if (do_mirror) {
        cv::flip(cv_resized_img, cv_resized_img, 1);
    }
    return cv_resized_img;
}

/*
// This function is called on prefetch thread and 
// load a batch of images
// Thread fetching the data
template <typename Dtype>
void MILDataLayer<Dtype>::InternalThreadEntry() {

  CPUTimer timer;
  timer.Start();

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data(); // can modify data using this pointer
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const int img_size = this->transform_param_.crop_size();
  const int channels = this->layer_param_.mil_data_param().channels();
  const int scale = this->transform_param_.scale();
  const bool mirror = this->transform_param_.mirror();

  const int images_per_batch = this->layer_param_.mil_data_param().images_per_batch();
  const int n_classes = this->layer_param_.mil_data_param().n_classes();
  const int num_scales = this->layer_param_.mil_data_param().num_scales();
  const float scale_factor = this->layer_param_.mil_data_param().scale_factor();
  const bool use_im_basename_for_hdf5 = this->layer_param_.mil_data_param().use_im_basename_for_hdf5();

  // zero out batch
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  // load a batch of images
  int item_id;
  for(int i_image = 0; i_image < images_per_batch; i_image++){
    // Sample which image to read
    unsigned int index = counter_; counter_ = counter_ + 1;
    const unsigned int rand_index = this->PrefetchRand(); // 
    if(this->layer_param_.mil_data_param().randomize())
      index = rand_index; // if need to randomize, then reset index to be rand_index

    // LOG(INFO) << index % this->num_images_ << ", " << this->num_images_;
    pair<std::string, std::string> p = this->image_database_[index % this->num_images_];
    string im_name = p.first;
    string full_im_name = p.second;
    
    cv::Mat cv_img = cv::imread(full_im_name, CV_LOAD_IMAGE_COLOR);
    if (!cv_img.data) {
      LOG(INFO) << "Could not open or find file " << full_im_name;
      return;
    }
    

    if (use_im_basename_for_hdf5 == true) {
      std::vector<std::string> elems = split(im_name, '/');
      std::string &basename = elems.back();
      // LOG(INFO)<<basename;
      hdf5_load_nd_dataset(this->label_file_id_, string("/labels-"+basename).c_str(), 4, 4, &this->label_blob_);
    } else {
      hdf5_load_nd_dataset(this->label_file_id_, string("/labels-"+im_name).c_str(), 4, 4, &this->label_blob_);
    }   
    // hdf5_load_nd_dataset(this->label_file_id_, string("/labels-"+im_name).c_str(), 4, 4, &this->label_blob_);
    const Dtype* label = label_blob_.mutable_cpu_data();
    // LOG(INFO) << "[width, height, channels, num] = " << label_blob_.width() << 
    //   ", " << label_blob_.height() << ", " << label_blob_.channels() << ", " << label_blob_.num();
    
    CHECK_EQ(label_blob_.width(), 1)          << "Expected width of label to be 1." ;
    CHECK_EQ(label_blob_.height(), n_classes) << "Expected height of label to be " << n_classes;
    CHECK_EQ(label_blob_.channels(), 1)       << "Expected channels of label to be 1." ;
    CHECK_EQ(label_blob_.num(), 1)            << "Expected num of label to be 1." ;

    float img_size_i = img_size;
    for(int i_scales = 0; i_scales < num_scales; i_scales++){
      // Resize such that the image is of size img_size, img_size
      item_id = i_image*num_scales + i_scales;
      // LOG(INFO) << "MIL Data Layer: scale: " << (int) round(img_size_i);
      cv::Mat cv_cropped_img = Transform_IDL(cv_img, static_cast<int>(round(img_size_i)), mirror);

      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            Dtype pixel =
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);
            top_data[((item_id * channels + c) * img_size + h)
                     * img_size + w]
                = (pixel - static_cast<Dtype>(mean_value_[c]))*scale;
          }
        }
      }
      img_size_i = std::max(static_cast<float>(1.), img_size_i*scale_factor);
    }
      
    for(int i_label = 0; i_label < n_classes; i_label++){
      top_label[i_image*n_classes + i_label] = 
        label[i_label];
    }
  }

    timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << timer.MilliSeconds() << " ms.";
}
*/

template<typename Dtype>
void MILDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer timer;
    timer.Start();
    CHECK(batch->data_.count());

    // Dtype* top_data = this->prefetch_data_.multable_cpu_data();
    // Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();
    const int img_size = this->transform_param_.crop_size();
    const int channels = this->layer_param_.mil_data_param().channels();
    const int scale = this->transform_param_.scale();
    const bool mirror = this->transform_param_.mirror();

    const int images_per_batch = this->layer_param_.mil_data_param().images_per_batch();
    const int n_classes = this->layer_param_.mil_data_param().n_classes();
    const int num_scales = this->layer_param_.mil_data_param().num_scales();
    const float scale_factor = this->layer_param_.mil_data_param().scale_factor();


    // zero out batch 
    // caffe_set(this->prefetch_data_.count(), Dtype(0), top_data)
    caffe_set(batch->data_.count(), Dtype(0), top_data);
    int item_id;
    for(int i_image = 0; i_image < images_per_batch; i_image++) {
        // sample which image to read 
        unsigned int index = counter_; counter_ = counter_ + 1;
        const unsigned int rand_index = this->PrefetchRand();
        if(this->layer_param_.mil_data_param().randomize())
            index = rand_index;

        // LOG(INFO) << index % this->num_images_ << "," << this->num_images_;
        pair<string, string> p = this->image_database_[index % this->num_images_];
        string im_name = p.first;
        string full_im_name = p.second;
        
        cv::Mat cv_img = cv::imread(full_im_name, CV_LOAD_IMAGE_COLOR);
        if(!cv_img.data) {
            LOG(ERROR) << "Could not open or find file " << full_im_name;
            return;
        }

        // REVIEW ktran: do not hardcode dataset name(or its prefix "/labels-")
        // REVIEW ktran: also do not use deep dataset name so that we donot have to modify the core caffe code 
        hdf5_load_nd_dataset(this->label_file_id_, string("/labels-"+im_name).c_str(), 4, 4, &this->label_blob_);
        const Dtype* label = label_blob_.mutable_cpu_data();

        CHECK_EQ(label_blob_.width(), 1) << "Expected width of label to be 1";
        CHECK_EQ(label_blob_.height(), n_classes) << "Expected width of label to be " << n_classes;
        CHECK_EQ(label_blob_.channels(), 1) << "Expected width of label to be 1";
        CHECK_EQ(label_blob_.num(), 1) << "Expected width of label to be 1";
        float img_size_i = img_size;
        for(int i_scales = 0; i_scales < num_scales; i_scales++) {
            // Resize such that the image is of size img_size ,img_size
            item_id = i_image*num_scales + i_scales;
            // LOG(INFO) << "MIL Data Layer:scale: " << (int) round(img_size_i);
            cv::Mat cv_cropped_img = Transform_IDL(cv_img, static_cast<int>(round(img_size_i)), mirror);
            for(int c = 0; c < channels; ++c) {
                for(int h = 0; h < cv_cropped_img.rows; ++h) {
                    for(int w = 0; w < cv_cropped_img.cols; ++w) {
                        Dtype pixel = static_cast<int>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);
                        top_data[((item_id * channels + c) * img_size + h) * img_size + w]
                            = (pixel - static_cast<Dtype>(mean_value_[c]))*scale;
                    }
                }
            }
            img_size_i = std::max(static_cast<float>(1.), img_size_i * scale_factor);
        }
        for(int i_label = 0; i_label < n_classes; i_label++) {
            top_label[i_image*n_classes + i_label] = label[i_label];
        }
    }
    
    timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << timer.MilliSeconds() << " ms.";
}


INSTANTIATE_CLASS(MILDataLayer);
REGISTER_LAYER_CLASS(MILData);

}  // namespace caffe
