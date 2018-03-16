#ifndef A_DET
#define A_DET

#include <iostream>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

#include "Thirdparty/Apriltag2/apriltag_src/apriltag.h"
#include "Thirdparty/Apriltag2/apriltag_src/tag36h11.h"
#include "Thirdparty/Apriltag2/apriltag_src/tag36h10.h"
#include "Thirdparty/Apriltag2/apriltag_src/tag36artoolkit.h"
#include "Thirdparty/Apriltag2/apriltag_src/tag25h9.h"
#include "Thirdparty/Apriltag2/apriltag_src/tag25h7.h"
#include "Thirdparty/Apriltag2/apriltag_src/common/getopt.h"

namespace April{
class tag_data
{
public:
    // The decoded ID of the tag
    int id;

    // How many error bits were corrected? Note: accepting large numbers of
    // corrected errors leads to greatly increased false positive rates.
    // NOTE: As of this implementation, the detector cannot detect tags with
    // a hamming distance greater than 2.
    int hamming;

    // A measure of the quality of tag localization: measures the
    // average contrast of the pixels around the border of the
    // tag. refine_pose must be enabled, or this field will be zero.
    float goodness;

    // A measure of the quality of the binary decoding process: the
    // average difference between the intensity of a data bit versus
    // the decision threshold. Higher numbers roughly indicate better
    // decodes. This is a reasonable measure of detection accuracy
    // only for very small tags-- not effective for larger tags (where
    // we could have sampled anywhere within a bit cell and still
    // gotten a good detection.)
    float decision_margin;

    // The center of the detection in image pixel coordinates.
    double c[2];

    // The corners of the tag in image pixel coordinates. These always
    // wrap counter-clock wise around the tag.
    double p[4][2];
};


class april_detector
{
public:
    april_detector(apriltag_detector_t * in_opt);
    ~april_detector();
    void clear();
    std::vector<tag_data> detection(const cv::Mat& gray);
    void detection_show(zarray_t * detections,const cv::Mat & frame);
    cv::Mat image_segmentation(const cv::Mat &gray, const std::vector<tag_data> &tag_vec);

    
private:
    apriltag_family_t *aprilopt;
    apriltag_detector_t* april_det_opt;
    void get_mat_type(const cv::Mat & M);
   
};
}
#endif