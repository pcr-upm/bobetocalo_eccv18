/** ****************************************************************************
 *  @file    FaceAlignmentDcfe.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2018/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_ALIGNMENT_DCFE_HPP
#define FACE_ALIGNMENT_DCFE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignment.hpp>
#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"

namespace upm {

/** ****************************************************************************
 * @class FaceAlignmentDcfe
 * @brief Class used for facial feature point detection.
 ******************************************************************************/
class FaceAlignmentDcfe: public FaceAlignment
{
public:
  FaceAlignmentDcfe(std::string path) : _path(path) {};

  ~FaceAlignmentDcfe() {};

  void
  parseOptions
    (
    int argc,
    char **argv
    );

  void
  train
    (
    const std::vector<upm::FaceAnnotation> &anns_train,
    const std::vector<upm::FaceAnnotation> &anns_valid
    );

  void
  load();

  void
  process
    (
    cv::Mat frame,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );

  tensorflow::Status
  imageToTensor
    (
    const cv::Mat &img,
    std::vector<tensorflow::Tensor>* output_tensors
    );

  std::vector<cv::Mat>
  tensorToMaps
    (
    const tensorflow::Tensor &img_tensor,
    const cv::Size &face_size
    );

private:
  std::string _path;
  std::vector<unsigned int> _cnn_landmarks;
  std::unique_ptr<tensorflow::Session> _session;
};

} // namespace upm

#endif /* FACE_ALIGNMENT_DCFE_HPP */
