/** ****************************************************************************
 *  @file    face_alignment_bobetocalo_eccv18_test.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2018/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

#include <trace.hpp>
#include <FaceAnnotation.hpp>
#include <FaceAlignmentDcfe.hpp>
#include <utils.hpp>

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
int
main
  (
  int argc,
  char **argv
  )
{
  // Read sample annotations
  upm::FaceAnnotation ann;
  ann.filename = "test/image_070.jpg";
  ann.bbox.pos = cv::Rect2f(195.196984225f,75.148001275f,898.385880775f,760.773923725f);
  upm::DB_PARTS[upm::FacePartLabel::leyebrow] = {1, 119, 2, 121, 3};
  upm::DB_PARTS[upm::FacePartLabel::reyebrow] = {4, 124, 5, 126, 6};
  upm::DB_PARTS[upm::FacePartLabel::leye] = {7, 138, 139, 8, 141, 142};
  upm::DB_PARTS[upm::FacePartLabel::reye] = {11, 144, 145, 12, 147, 148};
  upm::DB_PARTS[upm::FacePartLabel::nose] = {128, 129, 130, 17, 16, 133, 134, 135, 18};
  upm::DB_PARTS[upm::FacePartLabel::tmouth] = {20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161};
  upm::DB_PARTS[upm::FacePartLabel::bmouth] = {156, 157, 23, 159, 160, 168, 167, 166};
  upm::DB_PARTS[upm::FacePartLabel::lear] = {101, 102, 103, 104, 105, 106};
  upm::DB_PARTS[upm::FacePartLabel::rear] = {112, 113, 114, 115, 116, 117};
  upm::DB_PARTS[upm::FacePartLabel::chin] = {107, 108, 24, 110, 111};
  upm::DB_LANDMARKS = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168};
  std::vector<float> coords = {211.939577f,507.508412f,270.328191f,604.681633f,341.748364f,688.39419f,435.651535f,744.294051f,598.333129f,744.449497f,713.827933f,726.477666f,787.85507f,704.49905f,828.583557f,682.753602f,860.909862f,657.546242f,881.643288f,615.222814f,878.531127f,568.503989f,860.753602f,513.459889f,854.140669f,449.001586f,837.218907f,387.812517f,807.382167f,324.638265f,766.264252f,268.660681f,715.655227f,227.855287f,336.108592f,377.038169f,366.878816f,311.452074f,417.25549f,251.700872f,469.381735f,209.183138f,539.401273f,201.482089f,617.939577f,158.925493f,625.953147f,127.338367f,638.28439f,109.405397f,659.874393f,91.472428f,681.231229f,91.978439f,614.204822f,229.217875f,649.214998f,260.376712f,683.330552f,288.851678f,722.425235f,314.837078f,682.201945f,432.431205f,711.766655f,421.851162f,737.674333f,411.037138f,753.779194f,392.67588f,765.254675f,364.472943f,425.967746f,375.404366f,465.06243f,324.250468f,494.276575f,298.49905f,540.178499f,314.875939f,517.460704f,334.559296f,484.784648f,358.249437f,642.368897f,253.686056f,646.531127f,205.488873f,676.48445f,186.544696f,694.572865f,203.038169f,697.256737f,228.206668f,673.760902f,239.682148f,670.221268f,564.108592f,717.32952f,522.679787f,751.872547f,474.171714f,774.745786f,474.210575f,783.809425f,455.266398f,807.227537f,461.801609f,811.584073f,453.204306f,833.135214f,489.965684f,828.232991f,514.27842f,814.578835f,530.110437f,789.254891f,545.787009f,736.079391f,559.246045f,680.996431f,555.161537f,768.561142f,501.323767f,783.653979f,489.848286f,796.802125f,477.984192f,811.350905f,461.995915f,796.802125f,477.984192f,783.653979f,489.848286f,768.561142f,501.323767f};
  for (int cont=0; cont < upm::DB_LANDMARKS.size(); cont++)
  {
    unsigned int feature_idx = upm::DB_LANDMARKS[cont];
    float x = coords[(2*cont)+0];
    float y = coords[(2*cont)+1];
    bool visible = true;
    if (feature_idx < 1)
      continue;
    for (const auto &part : upm::DB_PARTS)
      if (std::find(part.second.begin(),part.second.end(),feature_idx) != part.second.end())
      {
        upm::FaceLandmark landmark;
        landmark.feature_idx = feature_idx;
        landmark.pos = cv::Point2f(x,y);
        landmark.visible = visible;
        ann.parts[part.first].landmarks.push_back(landmark);
        break;
      }
  }
  cv::Mat frame = cv::imread(ann.filename, cv::IMREAD_COLOR);
  if (frame.empty())
    return EXIT_FAILURE;

  // Set face detected position
  std::vector<upm::FaceAnnotation> faces(1);
  faces[0].bbox.pos = ann.bbox.pos;

  /// Load face components
  boost::shared_ptr<upm::FaceComposite> composite(new upm::FaceComposite());
  boost::shared_ptr<upm::FaceAlignment> fa(new upm::FaceAlignmentDcfe("data/"));
  composite->addComponent(fa);

  /// Parse face component options
  composite->parseOptions(argc, argv);
  composite->load();

  // Process frame
  double ticks = processFrame(frame, composite, faces, ann);
  UPM_PRINT("FPS = " << cv::getTickFrequency()/ticks);

  // Evaluate results
  boost::shared_ptr<std::ostream> output(&std::cout, [](std::ostream*){});
  composite->evaluate(output, faces, ann);

  // Draw results
  boost::shared_ptr<upm::Viewer> viewer(new upm::Viewer);
  viewer->init(0, 0, "face_alignment_bobetocalo_eccv18_test");
  showResults(viewer, ticks, 0, frame, composite, faces, ann);

  UPM_PRINT("End of face_alignment_bobetocalo_eccv18_test");
  return EXIT_SUCCESS;
};
