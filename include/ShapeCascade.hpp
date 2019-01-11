/** ****************************************************************************
 *  @file    ShapeCascade.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef SHAPE_CASCADE_HPP
#define SHAPE_CASCADE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <utils.hpp>
#include <FaceAnnotation.hpp>
#include <ModernPosit.h>
#include <ChannelFeatures.hpp>
#include <HonariChannelFeatures.hpp>
#include <LearningAlgorithm.hpp>
#include <transformation.hpp>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <serialization.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <fstream>

namespace upm {

enum class ChannelFeature { honari };
enum class InitialShape { honari };
enum class InitialHonariMode { maps };
enum class RuntimeMode { train, test };
const unsigned int MAX_NUM_LEVELS = 20;

/** ****************************************************************************
 * @class ShapeCascade
 * @brief Use cascade of regression trees that can localize facial landmarks.
 * Predict several update vectors that best agrees with the image data.
 ******************************************************************************/
class ShapeCascade
{
public:
  /// Test predictor
  ShapeCascade() {};

  ~ShapeCascade() {};

  void
  process
    (
    const cv::Mat &img,
    const float &scale,
    FaceAnnotation &face,
    const FaceAnnotation &ann,
    const ErrorMeasure &measure,
    const std::string &path
    )
  {
    /// Resize bounding box annotation
    FaceBox box_scaled = face.bbox;
    box_scaled.pos.x *= scale;
    box_scaled.pos.y *= scale;
    box_scaled.pos.width  *= scale;
    box_scaled.pos.height *= scale;

    /// Map shape from normalized space into the image dimension
    const unsigned int num_landmarks = _robust_shape.rows;
    cv::Mat utform = unnormalizingTransform(box_scaled.pos, _shape_size);

    /// Load Honari channels required to obtain probability metric
    std::shared_ptr<HonariChannelFeatures> hcf(_hcf);
    hcf->loadChannelsGenerator();
    cv::Rect_<float> prob_bbox = hcf->enlargeBbox(box_scaled.pos);
    std::vector<cv::Mat> prob_channels = hcf->generateChannels(img, prob_bbox);
    /// Load feature channels into memory
    std::shared_ptr<ChannelFeatures> cf(_cf);
    cf->loadChannelsGenerator();
    cf->loadFeaturesDescriptor();
    cv::Rect_<float> feat_bbox = cf->enlargeBbox(box_scaled.pos);
    std::vector<cv::Mat> feat_channels;
    for (const cv::Mat &channel : prob_channels)
      feat_channels.push_back(channel.clone());
    /// Set bbox according to cropped channel features
    cv::Point2f feat_scale = cv::Point2f(feat_channels[0].cols/feat_bbox.width, feat_channels[0].rows/feat_bbox.height);
    feat_scale *= hcf->map_scale;
    feat_bbox = cv::Rect_<float>(box_scaled.pos.x-feat_bbox.x, box_scaled.pos.y-feat_bbox.y, box_scaled.pos.width, box_scaled.pos.height);
    feat_bbox.x *= feat_scale.x;
    feat_bbox.y *= feat_scale.y;
    feat_bbox.width *= feat_scale.x;
    feat_bbox.height *= feat_scale.y;
    cv::Mat feat_utform = unnormalizingTransform(feat_bbox, _shape_size);

    /// Run algorithm several times using different initializations
    const unsigned int num_initial_shapes = static_cast<int>(_initial_shapes.size());
    std::vector<cv::Mat> current_shapes(num_initial_shapes), current_labels(num_initial_shapes), current_rigids(num_initial_shapes);
    for (unsigned int i=0; i < num_initial_shapes; i++)
    {
      cv::RNG rnd = cv::RNG();
      FaceAnnotation initial_face = generateInitialHonariShape(path, RuntimeMode::test, prob_bbox, scale, hcf->map_scale, prob_channels, box_scaled.pos, rnd);
      const cv::Mat ntform = normalizingTransform(box_scaled.pos, _shape_size);
      current_shapes[i] = cv::Mat::zeros(num_landmarks,3,CV_32FC1);
      current_labels[i] = cv::Mat::zeros(num_landmarks,1,CV_32FC1);
      facePartsToShape(initial_face.parts, ntform, scale, current_shapes[i], current_labels[i]);
    }

    for (unsigned int i=0; i < _forests.size(); i++)
    {
      for (unsigned int j=0; j < num_initial_shapes; j++)
      {
        /// Global similarity transform that maps 'robust_shape' to 'current_shape'
        float level = static_cast<float>(i) / static_cast<float>(MAX_NUM_LEVELS);
        current_rigids[j] = findSimilarityTransform(_robust_shape, current_shapes[j], current_labels[j]);
        cv::Mat features = cf->extractFeatures(feat_channels, box_scaled.pos.height, current_rigids[j], feat_utform, current_shapes[j], level);
        /// Compute residuals according to feature values
        for (const std::vector<impl::RegressionTree> &forest : _forests[i])
          for (const impl::RegressionTree &tree : forest)
            addResidualToShape(tree.leafs[tree.predict(features)].residual, current_shapes[j]);
      }
    }
    /// Facial feature location obtained
    bestEstimation(current_shapes, current_labels, utform, scale, ann, measure, face);
  };

  static FaceAnnotation
  generateInitialHonariShape
    (
    const std::string &path,
    const RuntimeMode &runtime_mode,
    const cv::Rect_<float> &prob_bbox,
    const float &scale,
    const float &map_scale,
    const std::vector<cv::Mat> &prob_channels,
    const cv::Rect_<float> &bbox,
    cv::RNG &rnd
    )
  {
    /// Robust correspondences
    const std::vector<unsigned int> mask = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    /// Load 3D face shape
    std::vector<cv::Point3f> world_all;
    std::vector<unsigned int> index_all;
    ModernPosit::loadWorldShape(path + "../../../headpose/posit/data/", DB_LANDMARKS, world_all, index_all);
    /// Intrinsic parameters
    cv::Rect_<float> bbox_cnn = cv::Rect_<float>(0, 0, prob_bbox.width, prob_bbox.height);
    double focal_length = static_cast<double>(bbox_cnn.width) * 1.5;
    cv::Point2f face_center = (bbox_cnn.tl() + bbox_cnn.br()) * 0.5f;
    cv::Mat cam_matrix = (cv::Mat_<float>(3,3) << focal_length,0,face_center.x, 0,focal_length,face_center.y, 0,0,1);
    /// Extrinsic parameters
    cv::Mat rot_matrix, trl_matrix;
    FaceAnnotation face_cnn;
    unsigned int shape_idx = 0;
    for (const auto &db_part: DB_PARTS)
      for (int feature_idx : db_part.second)
      {
        cv::Mat channel;
        cv::resize(prob_channels[shape_idx], channel, cv::Size(), map_scale, map_scale, cv::INTER_LINEAR);
        cv::Point lnd;
        float gaussian_kernel = (32.0f/map_scale)+1;
        cv::GaussianBlur(channel, channel, cv::Size(gaussian_kernel,gaussian_kernel), 0, 0, cv::BORDER_REFLECT_101);
        cv::minMaxLoc(channel, NULL, NULL, NULL, &lnd);
        FaceLandmark landmark;
        landmark.feature_idx = static_cast<unsigned int>(feature_idx);
        landmark.pos.x = lnd.x*(prob_bbox.width/channel.cols);
        landmark.pos.y = lnd.y*(prob_bbox.height/channel.rows);
        landmark.visible = true;
        face_cnn.parts[db_part.first].landmarks.push_back(landmark);
        shape_idx++;
      }
    /// Avoid outliers using RANSAC + POSIT
    const float MIN_METRIC = 0.3f;
    const unsigned int MAX_RANSAC_ITERS = 25;
    std::vector<cv::Mat> rot_matrices(MAX_RANSAC_ITERS), trl_matrices(MAX_RANSAC_ITERS);
    std::vector<float> metrics(MAX_RANSAC_ITERS, std::numeric_limits<float>::min());
    for (unsigned int i=0; i < MAX_RANSAC_ITERS; i++)
    {
      std::vector<unsigned int> inliers(mask);
      if (i > 0)
        inliers.erase(inliers.begin() + static_cast<int>(i%mask.size()));
      std::vector<cv::Point3f> world_pts;
      std::vector<cv::Point2f> image_pts;
      ModernPosit::setCorrespondences(world_all, index_all, face_cnn, inliers, world_pts, image_pts);
      ModernPosit::run(world_pts, image_pts, cam_matrix, 100, rot_matrices[i], trl_matrices[i]);
      cv::Mat rot_vector;
      cv::Rodrigues(rot_matrices[i], rot_vector);
      std::vector<cv::Point2f> image_all_proj;
      cv::projectPoints(world_all, rot_vector, trl_matrices[i], cam_matrix, cv::Mat(), image_all_proj);
      FaceAnnotation initial_face;
      for (const auto &db_part : DB_PARTS)
        for (int feature_idx : db_part.second)
        {
          FaceLandmark landmark;
          landmark.feature_idx = feature_idx;
          unsigned int shape_idx = std::distance(index_all.begin(), std::find(index_all.begin(),index_all.end(),feature_idx));
          landmark.pos.x = (prob_bbox.x + image_all_proj[shape_idx].x) / scale;
          landmark.pos.y = (prob_bbox.y + image_all_proj[shape_idx].y) / scale;
          landmark.visible = true;
          initial_face.parts[db_part.first].landmarks.push_back(landmark);
        }
      std::vector<float> values = getProbabilityMetric(initial_face.parts, prob_channels, prob_bbox, scale, map_scale);
      metrics[i] = 100.0f * static_cast<float>(std::accumulate(values.begin(),values.end(),0.0)) / values.size();
      if ((i == 0) and (metrics[i] > MIN_METRIC))
        break;
    }
    unsigned int best_i = std::distance(metrics.begin(), std::max_element(metrics.begin(),metrics.end()));
    rot_matrix = rot_matrices[best_i].clone();
    trl_matrix = trl_matrices[best_i].clone();
    if (runtime_mode == RuntimeMode::train)
    {
      cv::Point3f headpose = ModernPosit::rotationMatrixToEuler(rot_matrix);
      headpose += cv::Point3f(rnd.uniform(-20.0f,20.0f),rnd.uniform(-10.0f,10.0f),rnd.uniform(-10.0f,10.0f));
      rot_matrix = ModernPosit::eulerToRotationMatrix(headpose);
    }
    /// Project 3D shape into 2D landmarks
    cv::Mat rot_vector;
    cv::Rodrigues(rot_matrix, rot_vector);
    std::vector<cv::Point2f> image_all_proj;
    cv::projectPoints(world_all, rot_vector, trl_matrix, cam_matrix, cv::Mat(), image_all_proj);
    cv::Point3f headpose = ModernPosit::rotationMatrixToEuler(rot_matrix);
    FaceAnnotation initial_face;
    for (const auto &db_part : DB_PARTS)
      for (int feature_idx : db_part.second)
      {
        FaceLandmark landmark;
        landmark.feature_idx = feature_idx;
        unsigned int shape_idx = std::distance(index_all.begin(),std::find(index_all.begin(), index_all.end(), feature_idx));
        landmark.pos.x = (prob_bbox.x + image_all_proj[shape_idx].x) / scale;
        landmark.pos.y = (prob_bbox.y + image_all_proj[shape_idx].y) / scale;
        landmark.visible = not((db_part.first == FacePartLabel::lear and headpose.x < -30.0f) or
                               (db_part.first == FacePartLabel::rear and headpose.x > 30.0f) or
                              ((db_part.first == FacePartLabel::leye or db_part.first == FacePartLabel::leyebrow) and headpose.x < -45.0f) or
                              ((db_part.first == FacePartLabel::reye or db_part.first == FacePartLabel::reyebrow) and headpose.x > 45.0f));
        initial_face.parts[db_part.first].landmarks.push_back(landmark);
      }
    return initial_face;
  };

  static void
  bestEstimation
    (
    const std::vector<cv::Mat> &shapes,
    const std::vector<cv::Mat> &labels,
    const cv::Mat &tform,
    const float scale,
    const FaceAnnotation &ann,
    const ErrorMeasure &measure,
    FaceAnnotation &face
    )
  {
    unsigned int best_idx = 0;
    float err, best_err = std::numeric_limits<float>::max();
    const unsigned int num_initials = shapes.size();
    for (unsigned int i=0; i < num_initials; i++)
    {
      FaceAnnotation current;
      shapeToFaceParts(shapes[i], labels[i], tform, scale, current.parts);
      std::vector<unsigned int> indices;
      std::vector<float> errors;
      getNormalizedErrors(current, ann, measure, indices, errors);
      err = std::accumulate(errors.begin(),errors.end(),0.0) / static_cast<float>(errors.size());
      if (err < best_err)
      {
        best_err = err;
        best_idx = i;
      }
    }
    shapeToFaceParts(shapes[best_idx], labels[best_idx], tform, scale, face.parts);
  };

  static std::vector<float>
  getProbabilityMetric
    (
    const std::vector<FacePart> &parts,
    const std::vector<cv::Mat> &prob_channels,
    const cv::Rect_<float> &prob_bbox,
    const float &scale,
    const float &map_scale
    )
  {
    const unsigned int num_landmarks = prob_channels.size();
    std::vector<cv::Mat> img_channels(num_landmarks);
    for (unsigned int i=0; i < num_landmarks; i++)
      cv::resize(prob_channels[i], img_channels[i], cv::Size(), map_scale, map_scale, cv::INTER_LINEAR);

    /// Transform current shape coordinates to CNN channel size
    cv::Mat tform = normalizingTransform(prob_bbox, img_channels[0].size());
    cv::Mat shape = cv::Mat::zeros(num_landmarks,3,CV_32FC1);
    cv::Mat label = cv::Mat::zeros(num_landmarks,1,CV_32FC1);
    facePartsToShape(parts, tform, scale, shape, label);

    /// Return probability value for each landmark
    std::vector<float> values;
    for (unsigned int i=0; i < shape.rows; i++)
      if (label.at<float>(i,0) == 1.0f)
      {
        int x = static_cast<int>(shape.at<float>(i,0) + 0.5f);
        int y = static_cast<int>(shape.at<float>(i,1) + 0.5f);
        x = x < 0 ? 0 : x > img_channels[i].cols-1 ? img_channels[i].cols-1 : x;
        y = y < 0 ? 0 : y > img_channels[i].rows-1 ? img_channels[i].rows-1 : y;
        values.push_back(img_channels[i].at<float>(y,x));
      }
    return values;
  };

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & _shape_size & _robust_shape & _robust_label & _initial_shapes & _initial_labels & _forests & _initial_mode & _feature_mode & _feats_convergence_iter & _cf & _hcf;
  };

private:
  cv::Size2f _shape_size;
  cv::Mat _robust_shape;
  cv::Mat _robust_label;
  std::vector<cv::Mat> _initial_shapes;
  std::vector<cv::Mat> _initial_labels;
  std::vector<LearningAlgorithm::EnsembleTrees> _forests;
  InitialShape _initial_mode;
  ChannelFeature _feature_mode;
  int _feats_convergence_iter;
  std::shared_ptr<ChannelFeatures> _cf;
  std::shared_ptr<HonariChannelFeatures> _hcf;
};

} // namespace upm

#endif /* SHAPE_CASCADE_HPP */
