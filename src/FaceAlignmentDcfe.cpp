/** ****************************************************************************
 *  @file    FaceAlignmentDcfe.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2018/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignmentDcfe.hpp>
#include <trace.hpp>
#include <utils.hpp>
#include <boost/program_options.hpp>
#include "tensorflow/cc/ops/standard_ops.h"

namespace upm {

const float BBOX_SCALE = 0.3f;
const cv::Size FACE_SIZE = cv::Size(160,160);

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
FaceAlignmentDcfe::parseOptions
  (
  int argc,
  char **argv
  )
{
  /// Declare the supported program options
  FaceAlignment::parseOptions(argc, argv);
  namespace po = boost::program_options;
  po::options_description desc("FaceAlignmentDcfe options");
  UPM_PRINT(desc);
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
FaceAlignmentDcfe::train
  (
  const std::vector<upm::FaceAnnotation> &anns_train,
  const std::vector<upm::FaceAnnotation> &anns_valid
  )
{
  /// Training ERT model
  UPM_PRINT("Training ERT will be released soon ...");
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
FaceAlignmentDcfe::load()
{
  /// Loading CNN model
  UPM_PRINT("Loading CNN model");
  std::string trained_model = _path + _database + ".pb";
  std::vector<unsigned int> cnn_landmarks;
  if ((_database == "300w_public") or (_database == "300w_private") or (_database == "menpo"))
  {
    DB_PARTS[FacePartLabel::leyebrow] = {1, 119, 2, 121, 3};
    DB_PARTS[FacePartLabel::reyebrow] = {4, 124, 5, 126, 6};
    DB_PARTS[FacePartLabel::leye] = {7, 138, 139, 8, 141, 142};
    DB_PARTS[FacePartLabel::reye] = {11, 144, 145, 12, 147, 148};
    DB_PARTS[FacePartLabel::nose] = {128, 129, 130, 17, 16, 133, 134, 135, 18};
    DB_PARTS[FacePartLabel::tmouth] = {20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161};
    DB_PARTS[FacePartLabel::bmouth] = {156, 157, 23, 159, 160, 168, 167, 166};
    DB_PARTS[FacePartLabel::lear] = {101, 102, 103, 104, 105, 106};
    DB_PARTS[FacePartLabel::rear] = {112, 113, 114, 115, 116, 117};
    DB_PARTS[FacePartLabel::chin] = {107, 108, 24, 110, 111};
    _cnn_landmarks = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168};
  }
  if (_database == "cofw")
  {
    DB_PARTS[FacePartLabel::leyebrow] = {1, 101, 3, 102};
    DB_PARTS[FacePartLabel::reyebrow] = {4, 103, 6, 104};
    DB_PARTS[FacePartLabel::leye] = {7, 9, 8, 10, 105};
    DB_PARTS[FacePartLabel::reye] = {11, 13, 12, 14, 106};
    DB_PARTS[FacePartLabel::nose] = {16, 17, 18, 107};
    DB_PARTS[FacePartLabel::tmouth] = {20, 22, 21, 108};
    DB_PARTS[FacePartLabel::bmouth] = {109, 23};
    DB_PARTS[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 101, 3, 102, 4, 103, 6, 104, 7, 9, 8, 10, 105, 11, 13, 12, 14, 106, 16, 17, 18, 107, 20, 22, 21, 108, 109, 23, 24};
  }
  if (_database == "aflw")
  {
    DB_PARTS[FacePartLabel::leyebrow] = {1, 2, 3};
    DB_PARTS[FacePartLabel::reyebrow] = {4, 5, 6};
    DB_PARTS[FacePartLabel::leye] = {7, 101, 8};
    DB_PARTS[FacePartLabel::reye] = {11, 102, 12};
    DB_PARTS[FacePartLabel::nose] = {16, 17, 18};
    DB_PARTS[FacePartLabel::tmouth] = {20, 103, 21};
    DB_PARTS[FacePartLabel::lear] = {15};
    DB_PARTS[FacePartLabel::rear] = {19};
    DB_PARTS[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 2, 3, 4, 5, 6, 7, 101, 8, 11, 102, 12, 15, 16, 17, 18, 19, 20, 103, 21, 24};
  }
  tensorflow::GraphDef graph;
  tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), trained_model, &graph);
  if (not load_graph_status.ok())
    UPM_ERROR("Failed to load graph: " << trained_model);
  _session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status session_create_status = _session->Create(graph);
  if (not session_create_status.ok())
    UPM_ERROR("Failed to create session");
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
FaceAlignmentDcfe::process
  (
  cv::Mat frame,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  /// Analyze each detected face
  for (FaceAnnotation &face : faces)
  {
    /// Enlarge square bounding box
    cv::Point2f shift(face.bbox.pos.width*BBOX_SCALE, face.bbox.pos.height*BBOX_SCALE);
    cv::Rect_<float> bbox_enlarged = cv::Rect_<float>(face.bbox.pos.x-shift.x, face.bbox.pos.y-shift.y, face.bbox.pos.width+(shift.x*2), face.bbox.pos.height+(shift.y*2));
    /// Squared bbox required by neural networks
    bbox_enlarged.x = bbox_enlarged.x+(bbox_enlarged.width*0.5f)-(bbox_enlarged.height*0.5f);
    bbox_enlarged.width = bbox_enlarged.height;
    /// Scale image
    cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -bbox_enlarged.x, 0, 1, -bbox_enlarged.y);
    cv::warpAffine(frame, face_translated, T, bbox_enlarged.size());
    cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << FACE_SIZE.width/bbox_enlarged.width, 0, 0, 0, FACE_SIZE.height/bbox_enlarged.height, 0);
    cv::warpAffine(face_translated, face_scaled, S, FACE_SIZE);

    /// Testing CNN model
    std::vector<tensorflow::Tensor> input_tensors;
    tensorflow::Status read_tensor_status = imageToTensor(face_scaled, &input_tensors);
    if (not read_tensor_status.ok())
      UPM_ERROR(read_tensor_status);
    const tensorflow::Tensor &input_tensor = input_tensors[0];
//    UPM_PRINT("Input size:" << input_tensor.shape().DebugString() << ", tensor type:" << input_tensor.dtype()); // 1 x 160 x 160 x 3

    std::string input_layer = "input_1:0";
    std::vector<std::string> output_layers = {"k2tfout_0:0"};
    std::vector<tensorflow::Tensor> output_tensors;
    tensorflow::Status run_status = _session->Run({{input_layer, input_tensor}}, output_layers, {}, &output_tensors);
    if (not run_status.ok())
      UPM_ERROR("Running model failed: " << run_status);

    /// Convert output tensor to probability maps
//    UPM_PRINT("Output size:" << output_tensors[0].shape().DebugString() << ", tensor type:" << output_tensors[0].dtype()); // 1 x 68 x 25600
    std::vector<cv::Mat> channels = tensorToMaps(output_tensors[0], FACE_SIZE);
    #ifdef GAUSSIAN_FILTER
    for (cv::Mat &channel : channels)
      cv::GaussianBlur(channel, channel, cv::Size(31,31), 0, 0, cv::BORDER_REFLECT_101);
    #endif

    face.parts = FaceAnnotation().parts;
    for (const auto &db_part: DB_PARTS)
      for (int feature_idx : db_part.second)
      {
        auto found = std::find(_cnn_landmarks.begin(),_cnn_landmarks.end(),feature_idx);
        if (found == _cnn_landmarks.end())
          break;
        int pos = static_cast<int>(std::distance(_cnn_landmarks.begin(), found));
        cv::Point pt;
        cv::minMaxLoc(channels[pos], NULL, NULL, NULL, &pt);
        FaceLandmark landmark;
        landmark.feature_idx = static_cast<unsigned int>(feature_idx);
        landmark.pos.x = pt.x * (bbox_enlarged.width/FACE_SIZE.width) + bbox_enlarged.x;
        landmark.pos.y = pt.y * (bbox_enlarged.height/FACE_SIZE.height) + bbox_enlarged.y;
        landmark.visible = true;
        face.parts[db_part.first].landmarks.push_back(landmark);
      }
    /// Normalize network predictions
//    for (cv::Mat &channel : channels)
//    {
//      cv::normalize(channel, channel, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//      cv::cvtColor(channel, channel, cv::COLOR_GRAY2BGR);
//      cv::addWeighted(channel, 0.7, face_scaled, 0.5, 0.0, channel);
//      cv::imshow("weights", channel);
//      cv::waitKey(0);
//    }
  }

  /// Testing ERT model
  UPM_PRINT("Testing ERT will be released soon ...");
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
FaceAlignmentDcfe::imageToTensor
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
FaceAlignmentDcfe::tensorToMaps
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

} // namespace upm
