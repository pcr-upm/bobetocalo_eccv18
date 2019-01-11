/** ****************************************************************************
 *  @file    HonariChannelFeatures.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/12
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <HonariChannelFeatures.hpp>
#include <NearestEncoding.hpp>
#include <trace.hpp>
#include <utils.hpp>

namespace upm {

const float BBOX_SCALE = 0.3f;

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
HonariChannelFeatures::HonariChannelFeatures
  (
  const cv::Mat &shape,
  const cv::Mat &label,
  const std::string &path,
  const std::string &database
  )
{
  map_scale = 1.0f;
  _max_diameter = 0.20f;
  _min_diameter = 0.05f;
  _robust_shape = shape.clone();
  _robust_label = label.clone();
  _sampling_pattern = {
    {0.667f,0.000f}, {0.333f,0.577f}, {-0.333f,0.577f}, {-0.667f,0.000f}, {-0.333f,-0.577f}, {0.333f,-0.577f},
    {0.433f,0.250f}, {0.000f,0.500f}, {-0.433f,0.250f}, {-0.433f,-0.250f}, {0.000f,-0.500f}, {0.433f,-0.250f},
    {0.361f,0.000f}, {0.180f,0.313f}, {-0.180f,0.313f}, {-0.361f,0.000f}, {-0.180f,-0.313f}, {0.180f,-0.313f},
    {0.216f,0.125f}, {0.000f,0.250f}, {-0.216f,0.125f}, {-0.216f,-0.125f}, {0.000f,-0.250f}, {0.216f,-0.125f},
    {0.167f,0.000f}, {0.083f,0.144f}, {-0.083f,0.144f}, {-0.167f,0.000f}, {-0.083f,-0.144f}, {0.083f,-0.144f},
    {0.096f,0.055f}, {0.000f,0.111f}, {-0.096f,0.055f}, {-0.096f,-0.055f}, {0.000f,-0.111f}, {0.096f,-0.055f},
    {0.083f,0.000f}, {0.042f,0.072f}, {-0.042f,0.072f}, {-0.083f,0.000f}, {-0.042f,-0.072f}, {0.042f,-0.072f},
    {0.000f,0.000f}};
  _encoder.reset(new NearestEncoding());
  _path = path;
  _database = database;
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
cv::Rect_<float>
HonariChannelFeatures::enlargeBbox
  (
  const cv::Rect_<float> &bbox
  )
{
  /// Enlarge bounding box
  cv::Point2f shift(bbox.width*BBOX_SCALE, bbox.height*BBOX_SCALE);
  cv::Rect_<float> enlarged_bbox = cv::Rect_<float>(bbox.x-shift.x, bbox.y-shift.y, bbox.width+(shift.x*2), bbox.height+(shift.y*2));
  /// Squared bbox required by neural networks
  enlarged_bbox.x = enlarged_bbox.x+(enlarged_bbox.width*0.5f)-(enlarged_bbox.height*0.5f);
  enlarged_bbox.width = enlarged_bbox.height;
  return enlarged_bbox;
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
tensorflow::Status
HonariChannelFeatures::imageToTensor
  (
  const cv::Mat &img,
  std::vector<tensorflow::Tensor>* output_tensors
  )
{
  /// Copy mat into a tensor
  tensorflow::Tensor img_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({img.rows,img.cols,img.channels()}));
  auto img_tensor_mapped = img_tensor.tensor<float,3>();
  const uchar *pixel_coordinates = img.ptr<uchar>();
  for (unsigned int i=0; i < img.rows; i++)
    for (unsigned int j=0; j < img.cols; j++)
      for (unsigned int k=0; k < img.channels(); k++)
        img_tensor_mapped(i,j,k) = pixel_coordinates[i*img.cols*img.channels() + j*img.channels() + k];

  /// The convention for image ops in TensorFlow is that all images are expected
  /// to be in batches, so that they're four-dimensional arrays with indices of
  /// [batch, height, width, channel]. Because we only have a single image, we
  /// have to add a batch dimension of 1 to the start with ExpandDims()
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto holder = tensorflow::ops::Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_FLOAT);
  auto expander = tensorflow::ops::ExpandDims(root.WithOpName("expander"), holder, 0);
  auto divider = tensorflow::ops::Div(root.WithOpName("normalized"), expander, {255.0f});

  /// This runs the GraphDef network definition that we've just constructed
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<std::pair<std::string,tensorflow::Tensor>> input_tensors = {{"input", img_tensor},};
  TF_RETURN_IF_ERROR(session->Run({input_tensors}, {"normalized"}, {}, output_tensors));
  return tensorflow::Status::OK();
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
std::vector<cv::Mat>
HonariChannelFeatures::tensorToMaps
  (
  const tensorflow::Tensor &img_tensor,
  const cv::Size &face_size
  )
{
  tensorflow::TTypes<float>::ConstFlat data = img_tensor.flat<float>();
  unsigned int num_channels = static_cast<unsigned int>(img_tensor.dim_size(0));
  unsigned int channel_size = static_cast<unsigned int>(img_tensor.dim_size(1));
  std::vector<cv::Mat> channels(num_channels);
  for (unsigned int i=0; i < num_channels; i++)
  {
    std::vector<float> vec(channel_size);
    for (unsigned int j=0; j < channel_size; j++)
      vec[j] = data(i*channel_size+j);
    channels[i] = cv::Mat(face_size.height, face_size.width, CV_32FC1);
    memcpy(channels[i].data, vec.data(), vec.size()*sizeof(float));
  }
  return channels;
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
HonariChannelFeatures::loadChannelsGenerator()
{
  /// Loading CNN model
  std::string trained_model = "data/" + _database + ".pb";
  tensorflow::GraphDef graph;
  tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), trained_model, &graph);
  if (not load_graph_status.ok())
    UPM_ERROR("Failed to load graph: " << trained_model);
  _session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status session_create_status = _session->Create(graph);
  if (not session_create_status.ok())
    UPM_ERROR("Failed to create session");
  if ((_database == "300w_public") or (_database == "300w_private"))
  {
    _cnn_parts[FacePartLabel::leyebrow] = {1, 119, 2, 121, 3};
    _cnn_parts[FacePartLabel::reyebrow] = {4, 124, 5, 126, 6};
    _cnn_parts[FacePartLabel::leye] = {7, 138, 139, 8, 141, 142};
    _cnn_parts[FacePartLabel::reye] = {11, 144, 145, 12, 147, 148};
    _cnn_parts[FacePartLabel::nose] = {128, 129, 130, 17, 16, 133, 134, 135, 18};
    _cnn_parts[FacePartLabel::tmouth] = {20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161};
    _cnn_parts[FacePartLabel::bmouth] = {156, 157, 23, 159, 160, 168, 167, 166};
    _cnn_parts[FacePartLabel::lear] = {101, 102, 103, 104, 105, 106};
    _cnn_parts[FacePartLabel::rear] = {112, 113, 114, 115, 116, 117};
    _cnn_parts[FacePartLabel::chin] = {107, 108, 24, 110, 111};
    _cnn_landmarks = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168};
  }
  else if (_database == "cofw")
  {
    _cnn_parts[FacePartLabel::leyebrow] = {1, 101, 3, 102};
    _cnn_parts[FacePartLabel::reyebrow] = {4, 103, 6, 104};
    _cnn_parts[FacePartLabel::leye] = {7, 9, 8, 10, 105};
    _cnn_parts[FacePartLabel::reye] = {11, 13, 12, 14, 106};
    _cnn_parts[FacePartLabel::nose] = {16, 17, 18, 107};
    _cnn_parts[FacePartLabel::tmouth] = {20, 22, 21, 108};
    _cnn_parts[FacePartLabel::bmouth] = {109, 23};
    _cnn_parts[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 101, 3, 102, 4, 103, 6, 104, 7, 9, 8, 10, 105, 11, 13, 12, 14, 106, 16, 17, 18, 107, 20, 22, 21, 108, 109, 23, 24};
  }
  else if (_database == "aflw")
  {
    _cnn_parts[FacePartLabel::leyebrow] = {1, 2, 3};
    _cnn_parts[FacePartLabel::reyebrow] = {4, 5, 6};
    _cnn_parts[FacePartLabel::leye] = {7, 101, 8};
    _cnn_parts[FacePartLabel::reye] = {11, 102, 12};
    _cnn_parts[FacePartLabel::nose] = {16, 17, 18};
    _cnn_parts[FacePartLabel::tmouth] = {20, 103, 21};
    _cnn_parts[FacePartLabel::lear] = {15};
    _cnn_parts[FacePartLabel::rear] = {19};
    _cnn_parts[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 2, 3, 4, 5, 6, 7, 101, 8, 11, 102, 12, 15, 16, 17, 18, 19, 20, 103, 21, 24};
  }
  else if (_database == "wflw")
  {
    _cnn_parts[upm::FacePartLabel::leyebrow] = {1, 134, 2, 136, 3, 138, 139, 140, 141};
    _cnn_parts[upm::FacePartLabel::reyebrow] = {6, 147, 148, 149, 150, 4, 143, 5, 145};
    _cnn_parts[upm::FacePartLabel::leye] = {7, 161, 9, 163, 8, 165, 10, 167, 196};
    _cnn_parts[upm::FacePartLabel::reye] = {11, 169, 13, 171, 12, 173, 14, 175, 197};
    _cnn_parts[upm::FacePartLabel::nose] = {151, 152, 153, 17, 16, 156, 157, 158, 18};
    _cnn_parts[upm::FacePartLabel::tmouth] = {20, 177, 178, 22, 180, 181, 21, 192, 191, 190, 189, 188};
    _cnn_parts[upm::FacePartLabel::bmouth] = {187, 186, 23, 184, 183, 193, 194, 195};
    _cnn_parts[upm::FacePartLabel::lear] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110};
    _cnn_parts[upm::FacePartLabel::rear] = {122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132};
    _cnn_parts[upm::FacePartLabel::chin] = {111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121};
    _cnn_landmarks = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 138, 139, 140, 141, 4, 143, 5, 145, 6, 147, 148, 149, 150, 151, 152, 153, 17, 16, 156, 157, 158, 18, 7, 161, 9, 163, 8, 165, 10, 167, 11, 169, 13, 171, 12, 173, 14, 175, 20, 177, 178, 22, 180, 181, 21, 183, 184, 23, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197};
  }
  else
  {
    UPM_ERROR("Database does not exist ...")
  }
}

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
std::vector<cv::Mat>
HonariChannelFeatures::generateChannels
  (
  const cv::Mat &img,
  const cv::Rect_<float> &bbox
  )
{
  /// Crop and scale face image
  cv::Size_<float> face_size = cv::Size_<float>(160,160);
  cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -bbox.x, 0, 1, -bbox.y);
  cv::warpAffine(img, face_translated, T, bbox.size());
  cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << face_size.width/bbox.width, 0, 0, 0, face_size.height/bbox.height, 0);
  cv::warpAffine(face_translated, face_scaled, S, face_size);

  /// Testing CNN model
  std::vector<tensorflow::Tensor> input_tensors;
  tensorflow::Status read_tensor_status = imageToTensor(face_scaled, &input_tensors);
  if (not read_tensor_status.ok())
    UPM_ERROR(read_tensor_status);
  const tensorflow::Tensor &input_tensor = input_tensors[0];

  std::string input_layer = "input_1:0";
  std::vector<std::string> output_layers = {"k2tfout_0:0"};
  std::vector<tensorflow::Tensor> output_tensors;
  tensorflow::Status run_status = _session->Run({{input_layer, input_tensor}}, output_layers, {}, &output_tensors);
  if (not run_status.ok())
    UPM_ERROR("Running model failed: " << run_status);

  /// Convert output tensor to probability maps
  std::vector<cv::Mat> channels = tensorToMaps(output_tensors[0], face_size);

  /// Sort and resize channels to reduce memory required
  unsigned int shape_idx = 0;
  std::vector<cv::Mat> img_channels(channels.size());
  for (const auto &cnn_part: _cnn_parts)
    for (int feature_idx : cnn_part.second)
    {
      auto found = std::find(_cnn_landmarks.begin(),_cnn_landmarks.end(),feature_idx);
      if (found == _cnn_landmarks.end())
        break;
      int pos = static_cast<int>(std::distance(_cnn_landmarks.begin(), found));
      cv::resize(channels[pos], img_channels[shape_idx], cv::Size(), 1.0f/map_scale, 1.0f/map_scale, cv::INTER_LINEAR);
      shape_idx++;
    }
  return img_channels;
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
cv::Mat
HonariChannelFeatures::extractFeatures
  (
  const std::vector<cv::Mat> &img_channels,
  const float face_height,
  const cv::Mat &rigid,
  const cv::Mat &tform,
  const cv::Mat &shape,
  float level
  )
{
  /// Compute features from a pixel coordinates difference using a pattern
  cv::Mat CR = rigid.colRange(0,2).clone();
  float diameter = ((1-level)*_max_diameter) + (level*_min_diameter);
  std::vector<cv::Point2f> freak_pattern(_sampling_pattern);
  for (cv::Point2f &point : freak_pattern)
    point *= diameter;
  std::vector<cv::Point2f> pixel_coordinates = _encoder->generatePixelSampling(_robust_shape, freak_pattern);
  _encoder->setPixelSamplingEncoding(_robust_shape, _robust_label, pixel_coordinates);
  std::vector<cv::Point2f> current_pixel_coordinates = _encoder->getProjectedPixelSampling(CR, tform, shape);

  const unsigned int num_landmarks = static_cast<unsigned int>(shape.rows);
  const unsigned int num_sampling = static_cast<unsigned int>(_sampling_pattern.size());
  cv::Mat features = cv::Mat(num_landmarks,num_sampling,CV_32FC1);
  for (unsigned int i=0; i < num_landmarks; i++)
  {
    cv::Mat channel;
    cv::resize(img_channels[i], channel, cv::Size(), map_scale, map_scale, cv::INTER_LINEAR);
    for (unsigned int j=0; j < num_sampling; j++)
    {
      int x = static_cast<int>(current_pixel_coordinates[(i*num_sampling)+j].x + 0.5f);
      int y = static_cast<int>(current_pixel_coordinates[(i*num_sampling)+j].y + 0.5f);
      x = x < 0 ? 0 : x > channel.cols-1 ? channel.cols-1 : x;
      y = y < 0 ? 0 : y > channel.rows-1 ? channel.rows-1 : y;
      features.at<float>(i,j) = channel.at<float>(y,x);
    }
  }
  return features;
};

} // namespace upm
