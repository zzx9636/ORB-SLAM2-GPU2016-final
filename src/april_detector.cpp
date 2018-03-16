#include "../include/april_detector.h"

using namespace cv;
using namespace April;

april_detector::april_detector(apriltag_detector_t * in_opt)
{
    
    aprilopt=tag36h11_create(); 
    std::cout<<"Set default april tag family 36h11"<<std::endl;

    april_det_opt=apriltag_detector_create();
    apriltag_detector_add_family(april_det_opt,aprilopt);
    if(in_opt == NULL)
    {
        std::cout<<"Start default detector setting"<<std::endl;
        //apriltag_detector_add_family(april_det_opt,aprilopt);
        //default parameters
        april_det_opt->quad_decimate=1.0;
        april_det_opt->quad_sigma=0.0;
        april_det_opt->nthreads=4;
        april_det_opt->debug=0;
        april_det_opt->refine_edges=0;
        april_det_opt->refine_decode=0;
        april_det_opt->refine_pose=0;
        std::cout<<"Default Apriltag detector configuration set"<<std::endl;
    }else{
        april_det_opt->quad_decimate=in_opt->quad_decimate;
        april_det_opt->quad_sigma=in_opt->quad_sigma;
        april_det_opt->nthreads=in_opt->nthreads;
        april_det_opt->debug=in_opt->debug;
        april_det_opt->refine_edges=in_opt->refine_edges;
        april_det_opt->refine_decode=in_opt->refine_decode;
        april_det_opt->refine_pose=in_opt->refine_pose;
        std::cout<<"Imported Apriltag detector configuration set"<<std::endl;
    }

    //cv::namedWindow("April tags Detecion");
}



void april_detector::clear()
{
    if(aprilopt!=NULL)
        tag36h11_destroy(aprilopt);
    
    if(april_det_opt!=NULL)
        apriltag_detector_destroy(april_det_opt);

    //cv::destroyWindow("April tags Detecion");
}

april_detector::~april_detector()
{
    this->clear();
}

 std::vector<tag_data> april_detector::detection(const cv::Mat& gray)
{
    //get_mat_type(gray);
    image_u8_t im = { .width = gray.cols,
    .height = gray.rows,
    .stride = gray.cols,
    .buf = gray.data
    };
    zarray_t *detections = apriltag_detector_detect(april_det_opt, &im);
    std::cout << zarray_size(detections) << " tags detected" << std::endl;

    //detection_show(detections,gray);
    //copy detected tag information from detectors
    std::vector<tag_data> tags_vec(zarray_size(detections));
    for (int i = 0; i < zarray_size(detections); i++) 
    {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        tags_vec[i].id=det->id;
        tags_vec[i].hamming=det->hamming;
        tags_vec[i].goodness=det->goodness;
        tags_vec[i].decision_margin=det->decision_margin;
        std::copy(std::begin(det->c),std::end(det->c),std::begin(tags_vec[i].c));
        std::copy(std::begin(det->p),std::end(det->p),std::begin(tags_vec[i].p));
        //std::cout<<"Tag "<<i<<"'s center is ["<<tags_vec[i].c[0]<<","<<tags_vec[i].c[1]<<"]"<<std::endl;
    }
    zarray_destroy(detections);
    return tags_vec;
}

void april_detector::detection_show(zarray_t * detections,const cv::Mat & frame)
{
     for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[1][0], det->p[1][1]),
                     Scalar(0, 0xff, 0), 2);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0, 0, 0xff), 2);
            line(frame, Point(det->p[1][0], det->p[1][1]),
                     Point(det->p[2][0], det->p[2][1]),
                     Scalar(0xff, 0, 0), 2);
            line(frame, Point(det->p[2][0], det->p[2][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0xff, 0, 0), 2);

            std::stringstream ss;
            ss << det->id;
            String text = ss.str();
            int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
            double fontscale = 1.0;
            int baseline;
            Size textsize = getTextSize(text, fontface, fontscale, 2,
                                            &baseline);
            putText(frame, text, Point(det->c[0]-textsize.width/2,
                                       det->c[1]+textsize.height/2),
                    fontface, fontscale, Scalar(0xff, 0x99, 0), 2);
        }

        imshow("April tags Detecion", frame);
        waitKey(1);
}


  cv::Mat april_detector::image_segmentation(const cv::Mat& gray, const std::vector<tag_data> &tag_vec)
  {
      if(tag_vec.size()==0) //no tag detected
        return gray;

    /*
    cv::Mat tag_mask=Mat(gray.rows,gray.cols,CV_8UC1);
    
    for(int i=0; i<tag_mask.cols; i++){
        for(int j=0; j<tag_mask.rows; j++)
            tag_mask.at<uchar>(Point(i,j)) = uchar(255);
    }
    */
    std::vector<Point> ROI_Poly;
    std::vector<Point> ROI_Vertices;

    cv::Mat seg_img=gray;
    for(unsigned int i=0; i<tag_vec.size(); i++ ){
        ROI_Vertices.clear();
        ROI_Poly.clear();
        ROI_Vertices.push_back(Point((tag_vec[i]).p[0][0],(tag_vec[i]).p[0][1]));
        ROI_Vertices.push_back(Point((tag_vec[i]).p[1][0],(tag_vec[i]).p[1][1]));
        ROI_Vertices.push_back(Point((tag_vec[i]).p[2][0],(tag_vec[i]).p[2][1]));
        ROI_Vertices.push_back(Point((tag_vec[i]).p[3][0],(tag_vec[i]).p[3][1]));
        approxPolyDP(ROI_Vertices, ROI_Poly, 1.0, true);
        fillConvexPoly(seg_img, &ROI_Poly[0], ROI_Poly.size(), Scalar(255),4, 0);                 
    }

    //cv::Mat seg_img=Mat(gray.rows,gray.cols,CV_8UC1);
    //gray.copyTo(seg_img,tag_mask);
    
    return seg_img;
    //return tag_mask;
  }

  void april_detector::get_mat_type(const cv::Mat & M)
  {
    std::string r;
    int type=M.type();
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');
      printf("Matrix: %s %dx%d \n", r.c_str(), M.cols, M.rows );
  }
