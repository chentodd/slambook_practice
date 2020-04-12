#include <bits/stdc++.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <DBoW3/DBoW3.h>

int main(int argc, char** argv)
{
    std::cout << "reading database" << std::endl;
    DBoW3::Vocabulary vocab("./vocabulary.yaml.gz");
    if (vocab.empty())
    {
        std::cerr << "Vocabulary does not exist." << std::endl;
        return -1;
    }

    std::cout << "reading images..." << std::endl;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; ++i)
    {
        std::string path = "./data/" + std::to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    std::cout << "detecting ORB features..." << std::endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (cv::Mat& image : images)
    {
        std::vector<cv::KeyPoint> keypoint;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoint, descriptor);
        descriptors.push_back(descriptor);
    }

    std::cout << "comparing images with images" << std::endl;
    for (int i = 0; i < images.size(); ++i)
    {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        for (int j = i; j < images.size(); ++j)
        {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            double score = vocab.score(v1, v2);
            std::cout << "image " << i << "vs image " << j << " : " << score << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "comparing images with database" << std::endl;
    DBoW3::Database db(vocab, false, 0);
    for (int i = 0; i < descriptors.size(); ++i)
        db.add(descriptors[i]);
    for (int i = 0; i < descriptors.size(); ++i)
    {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4);
        std::cout << "searching for image " << i << " returns " << ret << std::endl << std::endl;
    }
    std::cout << "done." << std::endl;
    return 0;
}