#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file> [bins] [threshold]\n";
        return 1;
    }

    std::string videoPath = argv[1];
    int n = (argc >= 3) ? std::stoi(argv[2]) : 8;
    double T = (argc >= 4) ? std::stod(argv[3]) : 0.5;
    if (n < 4) n = 4;
    if (n > 256) n = 256;
    if (T < 0.0) T = 0.0;
    if (T > 1.0) T = 1.0;

    // Prepare output folder
    std::filesystem::create_directories("output/lab1");

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file " << videoPath << std::endl;
        return 1;
    }

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (cap.read(frame)) {
        frames.push_back(frame.clone());
    }
    cap.release();

    int frameCount = frames.size();
    if (frameCount == 0) {
        std::cerr << "Error: No frames read from video." << std::endl;
        return 1;
    }

    std::vector<cv::Mat> histList(frameCount);
    std::vector<int> sceneLabel(frameCount);
    float range[] = {0.0f, 256.0f};
    const float* histRange = { range };
    int histSize = n;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < frameCount; ++i) {
        cv::Mat histB, histG, histR;
        int ch = 0;
        cv::calcHist(&frames[i], 1, &ch, cv::Mat(), histB, 1, &histSize, &histRange);
        ch = 1;
        cv::calcHist(&frames[i], 1, &ch, cv::Mat(), histG, 1, &histSize, &histRange);
        ch = 2;
        cv::calcHist(&frames[i], 1, &ch, cv::Mat(), histR, 1, &histSize, &histRange);

        cv::normalize(histB, histB, 1.0, 0.0, cv::NORM_L1);
        cv::normalize(histG, histG, 1.0, 0.0, cv::NORM_L1);
        cv::normalize(histR, histR, 1.0, 0.0, cv::NORM_L1);

        cv::Mat combined(3 * histSize, 1, CV_32F);
        histB.copyTo(combined.rowRange(0, histSize));
        histG.copyTo(combined.rowRange(histSize, 2 * histSize));
        histR.copyTo(combined.rowRange(2 * histSize, 3 * histSize));
        histList[i] = combined;
    }

    int sceneCount = 1;
    sceneLabel[0] = sceneCount;
    for (int i = 1; i < frameCount; ++i) {
        double dist = cv::compareHist(histList[i-1], histList[i], cv::HISTCMP_BHATTACHARYYA);
        if (dist > T) {
            sceneCount++;
        }
        sceneLabel[i] = sceneCount;
    }

    for (int i = 0; i < frameCount; ++i) {
        std::string labelText = "Scene " + std::to_string(sceneLabel[i]);
        cv::putText(frames[i], labelText, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0,0,0), 3);
        cv::putText(frames[i], labelText, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(255,255,255), 1);

        std::ostringstream filename;
        filename << "output/lab1/frame_" << std::setw(3) << std::setfill('0') << i
                 << "_scene" << sceneLabel[i] << ".png";
        cv::imwrite(filename.str(), frames[i]);
    }

    std::cout << "✅ " << frameCount << " frames processed and saved to output/lab1/.\n";
    return 0;
}
