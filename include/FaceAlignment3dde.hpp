/** ****************************************************************************
 *  @file    FaceAlignment3dde.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2018/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_ALIGNMENT_3DDE_HPP
#define FACE_ALIGNMENT_3DDE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignment.hpp>
#include <ShapeCascade.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class FaceAlignment3dde
 * @brief Class used for facial feature point detection.
 ******************************************************************************/
class FaceAlignment3dde: public FaceAlignment
{
public:
  FaceAlignment3dde(std::string path) : _path(path) {};

  ~FaceAlignment3dde() {};

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

private:
  std::string _path;
  std::vector<ShapeCascade> _sp;
};

} // namespace upm

#endif /* FACE_ALIGNMENT_3DDE_HPP */
