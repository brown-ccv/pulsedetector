#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>

#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::videostab;

#define arg(name) cmd.get<string>(name)
#define argb(name) cmd.get<bool>(name)
#define argi(name) cmd.get<int>(name)
#define argf(name) cmd.get<float>(name)
#define argd(name) cmd.get<double>(name)


const int HORIZONTAL_BORDER_CROP = 20; // In pixels. Crops the border to reduce the black borders from stabilization being too noticeable.

// This video stabilization smooths the global trajectory using a sliding average window
// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }

    double x;
    double y;
    double a; // angle
};


void printHelp()
{
    cout << "Video stabilizer. Read a video and save the stabilized version to output directory.\n"
                "Usage: videostab <file_path> <output_dir> [arguments]\n\n"
                "Arguments:\n"
                "  --nkps=<int_number>\n"
                "      Number of keypoints to find in each frame. The default is 200.\n"
                "  -r, --radius=<int_number>\n"
                "      Set smoothing sliding window radius (in seconds). The default is 5.\n"
                "  --min-dis=<int_number>\n"
                "      Minimum distance between keypoints. The default is 30.\n"
                "  --save-side2side\n"
                "      Save side2side video to disk.\n\n"
                "  -q, --quiet\n"
                "      Don't show side2side video frames.\n\n"
                "  -h, --help\n"
                "      Print help.\n\n";
}

int main(int argc, char **argv)
{

    const char *keys =
            "{ @1                       |           | }"
            "{ @2                       |           | }"
            "{  nkps                    | 0        | }"
            "{ r  radius                | 0 | }"
            "{  min-dis                 | 0 | }"
            "{  save-side2side          |  | }"
            "{ q quiet                  |  | }"
            "{ h help                   |  | }";
    CommandLineParser cmd(argc, argv, keys);

    // parse command arguments
    if (argb("help"))
    {
        printHelp();
        return 0;
    }

   // check if source video is specified
    string inputPath = arg(0);
    if (inputPath.empty())
        throw runtime_error("specify video file path");

    string outputPath = arg(1);
    if (outputPath.empty())
        throw runtime_error("specify output directory");


    bool quietMode = argb("quiet");
    bool side2side = argb("save-side2side");
    float smoothingRadius = argf("radius");
    int nkps = argi("nkps");
    int minDis = argi("min-dis");
    float kpsQuality = 0.01;

    cout << "Running videostab.cpp with arguments: \n"
    << "nkps: " << nkps << "\nradius: " << smoothingRadius << "\nmin-dis: " << minDis << endl;

    if (side2side)
        cout << "Saving side2side \n";
    if (quietMode)
        cout << "Running quiet mode \n";

    // For further analysis
    ofstream out_transform(outputPath + "/prev_to_cur_T.txt");
    ofstream out_trajectory(outputPath + "/trajectory.txt");
    ofstream out_smoothed_trajectory(outputPath + "/smooth_trajectory.txt");
    ofstream out_new_transform(outputPath + "/new_prev_to_cur_T.txt");
    string vid_fout = outputPath + "/stabilized.mov";
    string side2side_fout = outputPath + "/side2side.mov";


    Ptr<VideoFileSource> source = makePtr<VideoFileSource>(inputPath);
    cout << "frame count (rough): " << source->count() << endl;
    double fps = source->fps();

    Mat cur, cur_grey;
    Mat prev, prev_grey;

    prev = source->nextFrame();
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    vector <TransformParam> prev_to_cur_transform; // previous to current

    int k=1;
    int max_frames = source->count();
    Mat last_T;

    while(true) {
        cur = source->nextFrame();

        if(cur.data == NULL) {
            break;
        }

        cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

        // vector from prev to cur
        vector <Point2f> prev_corner, cur_corner;
        vector <Point2f> prev_corner2, cur_corner2;
        vector <uchar> status;
        vector <float> err;

        goodFeaturesToTrack(prev_grey, prev_corner, nkps, kpsQuality, minDis);
        calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

        // weed out bad matches
        for(size_t i=0; i < status.size(); i++) {
            if(status[i]) {
                prev_corner2.push_back(prev_corner[i]);
                cur_corner2.push_back(cur_corner[i]);
            }
        }

        // translation + rotation only
        Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

        // in rare cases no transform is found. We'll just use the last known good transform.
        if(T.data == NULL) {
            last_T.copyTo(T);
        }

        T.copyTo(last_T);

        // decompose T
        double dx = T.at<double>(0,2);
        double dy = T.at<double>(1,2);
        double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

        prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        out_transform << k << " " << dx << " " << dy << " " << da << endl;

        cur.copyTo(prev);
        cur_grey.copyTo(prev_grey);

        cout << "Frame: " << k << "/" << max_frames << " - nkps: " << prev_corner2.size() << endl;
        k++;
    }

    // Step 2 - Accumulate the transformations to get the image trajectory
    cout << "Step 2" << endl;

    // Accumulated frame to frame transform
    double a = 0;
    double x = 0;
    double y = 0;

    vector <Trajectory> trajectory; // trajectory at all frames

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        trajectory.push_back(Trajectory(x,y,a));

        out_trajectory << (i+1) << " " << x << " " << y << " " << a << endl;
    }

    // Step 3 - Smooth out the trajectory using an averaging window
    cout << "Step 3" << endl;

    vector <Trajectory> smoothed_trajectory; // trajectory at all frames


    int radius = int(smoothingRadius*fps);
    for(size_t i=0; i < trajectory.size(); i++) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_a = 0;
        int count = 0;


        for(int j=-radius; j <= radius; j++) {
            if(i+j >= 0 && i+j < trajectory.size()) {
                sum_x += trajectory[i+j].x;
                sum_y += trajectory[i+j].y;
                sum_a += trajectory[i+j].a;

                count++;
            }
        }

        double avg_a = sum_a / count;
        double avg_x = sum_x / count;
        double avg_y = sum_y / count;

        smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));

        out_smoothed_trajectory << (i+1) << " " << avg_x << " " << avg_y << " " << avg_a << endl;
    }

    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    cout << "Step 4" << endl;

    vector <TransformParam> new_prev_to_cur_transform;

    // Accumulated frame to frame transform
    a = 0;
    x = 0;
    y = 0;

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        // target - current
        double diff_x = smoothed_trajectory[i].x - x;
        double diff_y = smoothed_trajectory[i].y - y;
        double diff_a = smoothed_trajectory[i].a - a;

        double dx = prev_to_cur_transform[i].dx + diff_x;
        double dy = prev_to_cur_transform[i].dy + diff_y;
        double da = prev_to_cur_transform[i].da + diff_a;

        new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        out_new_transform << (i+1) << " " << dx << " " << dy << " " << da << endl;
    }

    // Step 5 - Apply the new transformation to the video
    source->reset();
    Mat T(2,3,CV_64F);

    int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct

    cout << "Step 5" << endl;
    k=0;
    VideoWriter writer;
    VideoWriter side2sideWriter;

    while(k < max_frames-1) { // don't process the very last frame, no valid transform
        cur = source->nextFrame();

        if(cur.data == NULL) {
            cout << "Null capture" << endl;
            continue;
        }

        T.at<double>(0,0) = cos(new_prev_to_cur_transform[k].da);
        T.at<double>(0,1) = -sin(new_prev_to_cur_transform[k].da);
        T.at<double>(1,0) = sin(new_prev_to_cur_transform[k].da);
        T.at<double>(1,1) = cos(new_prev_to_cur_transform[k].da);

        T.at<double>(0,2) = new_prev_to_cur_transform[k].dx;
        T.at<double>(1,2) = new_prev_to_cur_transform[k].dy;

        Mat cur2;

        warpAffine(cur, cur2, T, cur.size());

        cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

        // Resize cur2 back to cur size, for better side by side comparison
        resize(cur2, cur2, cur.size());

        // write to file
        if (!writer.isOpened())
            writer.open(vid_fout, VideoWriter::fourcc('m','p','4','v'),
                        fps, cur2.size());

        writer << cur2;

        if (side2side)
        {
            // Now draw the original and stabilized side by side for coolness
            Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());

            cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
            cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

            // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
            if(canvas.cols > 1920) {
                resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
            }


            // write to file
            if (!side2sideWriter.isOpened())
                side2sideWriter.open(side2side_fout, VideoWriter::fourcc('m','p','4','v'),
                            fps, canvas.size());

            side2sideWriter << canvas;

            if (!quietMode){
                imshow("before and after", canvas);
                char key = static_cast<char>(waitKey(3));
                if (key == 27) { cout << endl; break; }
            }
        }

        k++;
    }

    source.release();
    writer.release();
    return 0;
}