/** ****************************************************************************
 *  @file    FaceAlignment3dde.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2018/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignment3dde.hpp>
#include <trace.hpp>
#include <utils.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <fstream>

namespace upm {

const float FACE_HEIGHT = 160.0f;

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
FaceAlignment3dde::parseOptions
  (
  int argc,
  char **argv
  )
{
  /// Declare the supported program options
  FaceAlignment::parseOptions(argc, argv);
  namespace po = boost::program_options;
  po::options_description desc("FaceAlignment3dde options");
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
FaceAlignment3dde::train
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
FaceAlignment3dde::load()
{
  /// Loading shape predictors
  UPM_PRINT("Loading facial-feature predictors");
  std::vector<std::string> paths;
  boost::filesystem::path dir_path(_path + _database + "/");
  boost::filesystem::directory_iterator end_it;
  for (boost::filesystem::directory_iterator it(dir_path); it != end_it; ++it)
    paths.push_back(it->path().string());
  sort(paths.begin(), paths.end());
  UPM_PRINT("> Number of predictors found: " << paths.size());

  _sp.resize(HP_LABELS.size());
  for (const std::string &path : paths)
  {
    try
    {
      std::ifstream ifs(path);
      cereal::BinaryInputArchive ia(ifs);
      ia >> _sp[boost::lexical_cast<unsigned int>(path.substr(0,path.size()-4).substr(path.find_last_of('_')+1))] >> DB_PARTS >> DB_LANDMARKS;
      ifs.close();
    }
    catch (cereal::Exception &ex)
    {
      UPM_ERROR("Exception during predictor deserialization: " << ex.what());
    }
  }
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
FaceAlignment3dde::process
  (
  cv::Mat frame,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  /// Analyze each detected face to tell us the face part locations
  for (FaceAnnotation &face : faces)
  {
    int yaw_idx = getHeadposeIdx(FaceAnnotation().headpose.x);
    cv::Mat img;
    float scale = FACE_HEIGHT/face.bbox.pos.height;
    cv::resize(frame, img, cv::Size(), scale, scale);
    _sp[yaw_idx].process(img, scale, face, ann, _measure, _path);
  }
};

} // namespace upm
