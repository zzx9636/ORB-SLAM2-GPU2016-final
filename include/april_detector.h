#ifndef A_DET
#define A_DET

#include <iostream>
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
class april_detector
{
public:
    april_detector(apriltag_detector_t * in_opt);
    ~april_detector();
    void clear();
    void detection(cv::Mat gray);
    void detection_show(zarray_t * detections,cv::Mat & frame);

    
private:
    apriltag_family_t *aprilopt;
    apriltag_detector_t* april_det_opt;
   
};
}
#endif