/** ****************************************************************************
 *  @file    GaussianChannelFeatures.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <GaussianChannelFeatures.hpp>
#include <NearestEncoding.hpp>

namespace upm {

const float BBOX_SCALE = 0.5f;

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
GaussianChannelFeatures::GaussianChannelFeatures
  (
  const cv::Mat &shape,
  const cv::Mat &label
  )
{
  _max_diameter = 0.20f;
  _min_diameter = 0.05f;
  _robust_shape = shape.clone();
  _robust_label = label.clone();
  _channel_sigmas = {0.0000000f, 0.0017813f, 0.0023753f, 0.0035633f, 0.0053446f, 0.0077206f, 0.0106900f, 0.0142533f};
  _sampling_pattern = {
    {0.667f,0.000f}, {0.333f,0.577f}, {-0.333f,0.577f}, {-0.667f,0.000f}, {-0.333f,-0.577f}, {0.333f,-0.577f},
    {0.433f,0.250f}, {0.000f,0.500f}, {-0.433f,0.250f}, {-0.433f,-0.250f}, {0.000f,-0.500f}, {0.433f,-0.250f},
    {0.361f,0.000f}, {0.180f,0.313f}, {-0.180f,0.313f}, {-0.361f,0.000f}, {-0.180f,-0.313f}, {0.180f,-0.313f},
    {0.216f,0.125f}, {0.000f,0.250f}, {-0.216f,0.125f}, {-0.216f,-0.125f}, {0.000f,-0.250f}, {0.216f,-0.125f},
    {0.167f,0.000f}, {0.083f,0.144f}, {-0.083f,0.144f}, {-0.167f,0.000f}, {-0.083f,-0.144f}, {0.083f,-0.144f},
    {0.096f,0.055f}, {0.000f,0.111f}, {-0.096f,0.055f}, {-0.096f,-0.055f}, {0.000f,-0.111f}, {0.096f,-0.055f},
    {0.083f,0.000f}, {0.042f,0.072f}, {-0.042f,0.072f}, {-0.083f,0.000f}, {-0.042f,-0.072f}, {0.042f,-0.072f},
    {0.000f,0.000f}};
  _channel_per_sampling = {
    7, 7, 7, 7, 7, 7,
    6, 6, 6, 6, 6, 6,
    5, 5, 5, 5, 5, 5,
    4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1,
    0};

  /// Generate pixel locations relative to robust shape
  _encoder.reset(new NearestEncoding());
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
GaussianChannelFeatures::enlargeBbox
  (
  const cv::Rect_<float> &bbox
  )
{
  /// Enlarge bounding box
  cv::Point2f shift(bbox.width*BBOX_SCALE, bbox.height*BBOX_SCALE);
  cv::Rect_<float> enlarged_bbox = cv::Rect_<float>(bbox.x-shift.x, bbox.y-shift.y, bbox.width+(shift.x*2), bbox.height+(shift.y*2));
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
std::vector<cv::Mat>
GaussianChannelFeatures::generateChannels
  (
  const cv::Mat &img,
  const cv::Rect_<float> &bbox
  )
{
  /// Crop face image
  cv::Mat face_scaled, T = (cv::Mat_<float>(2,3) << 1, 0, -bbox.x, 0, 1, -bbox.y);
  cv::warpAffine(img, face_scaled, T, bbox.size());

  /// Several gaussian channel images
  const unsigned int num_channels = static_cast<unsigned int>(_channel_sigmas.size());
  std::vector<cv::Mat> img_channels(num_channels);
  cv::cvtColor(face_scaled, img_channels[0], cv::COLOR_BGR2GRAY);
  for (unsigned int j=1; j < num_channels; j++)
    cv::GaussianBlur(img_channels[0], img_channels[j], cv::Size(), _channel_sigmas[j]*bbox.height);
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
GaussianChannelFeatures::extractFeatures
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
  const unsigned int num_sampling = static_cast<unsigned int>(_channel_per_sampling.size());
  cv::Mat features = cv::Mat(num_landmarks,num_sampling,CV_32FC1);
  for (unsigned int i=0; i < num_landmarks; i++)
  {
    for (unsigned int j=0; j < num_sampling; j++)
    {
      int x = static_cast<int>(current_pixel_coordinates[(i*num_sampling)+j].x + 0.5f);
      int y = static_cast<int>(current_pixel_coordinates[(i*num_sampling)+j].y + 0.5f);
      x = x < 0 ? 0 : x;
      y = y < 0 ? 0 : y;
      unsigned int channel = _channel_per_sampling[j];
      x = x > img_channels[channel].cols-1 ? img_channels[channel].cols-1 : x;
      y = y > img_channels[channel].rows-1 ? img_channels[channel].rows-1 : y;
      features.at<float>(i,j) = img_channels[channel].at<uchar>(y,x);
    }
  }
  return features;
};

} // namespace upm
