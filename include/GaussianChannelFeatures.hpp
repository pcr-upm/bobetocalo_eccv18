/** ****************************************************************************
 *  @file    GaussianChannelFeatures.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef GAUSSIAN_CHANNEL_FEATURES_HPP
#define GAUSSIAN_CHANNEL_FEATURES_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ChannelFeatures.hpp>
#include <FaceAnnotation.hpp>
#include <FeaturesRelativeEncoding.hpp>
#include <serialization.hpp>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class GaussianChannelFeatures
 * @brief Extract pixel features from several gaussian channels.
 ******************************************************************************/
class GaussianChannelFeatures : public ChannelFeatures
{
public:
  GaussianChannelFeatures() {};

  GaussianChannelFeatures
    (
    const cv::Mat &shape,
    const cv::Mat &label
    );

  virtual
  ~GaussianChannelFeatures() {};

  cv::Rect_<float>
  enlargeBbox
    (
    const cv::Rect_<float> &bbox
    );

  void
  loadChannelsGenerator() {};

  std::vector<cv::Mat>
  generateChannels
    (
    const cv::Mat &img,
    const cv::Rect_<float> &bbox
    );

  void
  loadFeaturesDescriptor() {};

  cv::Mat
  extractFeatures
    (
    const std::vector<cv::Mat> &img_channels,
    const float face_height,
    const cv::Mat &rigid,
    const cv::Mat &tform,
    const cv::Mat &shape,
    float level
    );

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & _max_diameter & _min_diameter & _robust_shape & _robust_label & _channel_sigmas & _sampling_pattern & _channel_per_sampling & _encoder;
  };

protected:
  float _max_diameter, _min_diameter;
  cv::Mat _robust_shape, _robust_label;
  std::vector<float> _channel_sigmas;
  std::vector<cv::Point2f> _sampling_pattern;
  std::vector<unsigned int> _channel_per_sampling;
  std::shared_ptr<FeaturesRelativeEncoding> _encoder;
};

} // namespace upm

CEREAL_REGISTER_TYPE(upm::GaussianChannelFeatures);
CEREAL_REGISTER_POLYMORPHIC_RELATION(upm::ChannelFeatures, upm::GaussianChannelFeatures);

#endif /* GAUSSIAN_CHANNEL_FEATURES_HPP */
