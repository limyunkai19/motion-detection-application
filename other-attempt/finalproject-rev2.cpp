#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// BIN_SIZE should be power of 2
#define BIN_SIZE 32;
const double PI = acos(-1);

short max(short a, int b){
    return max(int(a), b);
}

short max(int a, short b){
    return max(b, a);
}

struct QuantizedBin{
    short quantity, lambda;
    Vec3b color;
    vector<short> frame_index;

    QuantizedBin(){}
    QuantizedBin(Vec3b color){
        lambda = 0;
        frame_index.reserve(1);
        frame_index.push_back(-1);
        quantity = 0;
        this->color = color;
    }

    bool operator< (const QuantizedBin x) const {
        return this->quantity < x.quantity;
    }
};

class CodeBook{
  private:
    static const short sbin = BIN_SIZE;
    static const short nbin = 256/sbin;
    static const short tbin = nbin*nbin*nbin;

    short cur;
    bool isSorted;
    vector<QuantizedBin> colorBin;
    // colorBinIdx track the where the colorBin is after sorting
    vector<short> colorBinIdx;

    void binSort(){
        // sort reverse order
        sort(colorBin.rbegin(), colorBin.rend());

        for(int i = 0; i < colorBin.size(); i++){
            colorBinIdx[colorToIdx(colorBin[i].color)] = i;
        }
        isSorted = true;
    }
    int colorToIdx(Vec3b color){
        int b = color[0]/sbin, g = color[1]/sbin, r = color[2]/sbin;
        return nbin*nbin*b + nbin*g + r;
    }

  public:
    CodeBook(){
        cur = 0;

        colorBin.resize(tbin);
        colorBinIdx.resize(tbin);
        for(int i = 0; i < tbin; i++){
            int bin = i;
            int r = bin%nbin; bin /= nbin;
            int g = bin%nbin; bin /= nbin;
            int b = bin%nbin;

            colorBin[i] = QuantizedBin(Vec3b(b*sbin+sbin/2, g*sbin+sbin/2, r*sbin+sbin/2));
            colorBinIdx[i] = i;
        }
        isSorted = true;
    }

    void update(Vec3b color){
        isSorted = false;
        int idx = colorBinIdx[colorToIdx(color)];

        // check colorBin[idx].color ==  color
        colorBin[idx].quantity++;
        colorBin[idx].lambda = max(colorBin[idx].lambda, cur-colorBin[idx].frame_index.back());
        colorBin[idx].frame_index.push_back(cur);
        cur++;
    }

    Vec3b get_color(){
        if(!isSorted)
            binSort();
        return colorBin[0].color;
    }

    bool compare_color(Vec3b color){
        Vec3b dominant = this->get_color();

        const int thres = 40;
        bool match = true;
        for(int i = 0; i < 3; i++){
            if(abs(dominant[i]-color[i]) > thres)
                match = false;
        }
        if(match)
            return true;

        if(!isSorted)
            binSort();

        // variable !!!
        for(int i = 1; i < 10; i++){
            colorBin[i].lambda = max(colorBin[i].lambda, cur-colorBin[i].frame_index.back());
            if(colorBin[i].lambda > 300)
                continue;
            // a dynamic backgruond code word
            bool match = true;
            for(int j = 0; j < 3; j++){
                if(abs(colorBin[i].color[j]-color[j]) > thres)
                    match = false;
            }
            if(match)
                return true;
        }
        return false;
    }
};

class BackgroundModel{
  private:
    // every pixel has a CodeBook
    int rows, cols;
    vector<CodeBook> pixelCodeBook;

  public:
    BackgroundModel(int rows, int cols){
        this->rows = rows;
        this->cols = cols;

        pixelCodeBook.assign(rows*cols, CodeBook());
    }

    void learn(Mat frame){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                // mat.at<>(r, c);
                pixelCodeBook[i*cols+j].update(frame.at<Vec3b>(i, j));
            }
        }
    }

    Mat get_background_model(){
        Mat background(rows, cols, CV_8UC3);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                // mat.at<>(r, c);
                background.at<Vec3b>(i, j) = pixelCodeBook[i*cols+j].get_color();
            }
        }
        return background;
    }

    Mat get_foreground_mask(Mat frame){
        Mat foreground(rows, cols, CV_8UC1);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                // mat.at<>(r, c);
                // compare color return true(1) false(0), multiple 255 to get binary foreground image
                foreground.at<uchar>(i, j) = 255 *
                    !pixelCodeBook[i*cols+j].compare_color(frame.at<Vec3b>(i, j));
            }
        }
        return foreground;
    }
};

struct ForegroundObject{
    Point center;
    int area, x, y, h, w;
    double velocity_x, velocity_y;
    vector<Point> path;

    ForegroundObject(Point center, int area, int x, int y, int h, int w){
        velocity_x = velocity_y = 0;
        this->center = center;
        this->area = area;
        this->x = x;
        this->y = y;
        this->h = h;
        this->w = w;
    }

    void draw(Mat &img){
        rectangle(img, Point(x, y), Point(x+w, y+h), Scalar(0, 0, 255), 1);
        circle(img, center, 2, Scalar(255, 0, 0), -1);
        arrowedLine(img, center, Point(center.x + velocity_x*7, center.y + velocity_y*7),
            Scalar(255, 255, 0), 2, CV_AA, 0, 0.2);

        // draw path
        for(int i = 0; i < (int)path.size()-1; i++){
            line(img, path[i], path[i+1], Scalar(255, 255, 0), 1, CV_AA);
        }
    }
};

class TrajectoryModel{
  private:
    vector<ForegroundObject> objects;

  public:
    void computeVelocity(vector<ForegroundObject> &curObjects){
        // for each curObject, try to match with a object
        for(int i = 0; i < curObjects.size(); i++){
            for(int j = 0; j < objects.size(); j++){
                if( abs(objects[j].center.x + objects[j].velocity_x - curObjects[i].center.x) < 9 &&
                    abs(objects[j].center.y + objects[j].velocity_y - curObjects[i].center.y) < 9){
                    // matched expected center

                    double v_x = curObjects[i].center.x - objects[j].center.x;
                    double v_y = curObjects[i].center.y - objects[j].center.y;

                    // check direction, angle should be less than 120 degree
                    double angle_rad = atan2(v_x, v_y)-atan2(objects[j].velocity_x, objects[j].velocity_y);
                    if(abs(angle_rad) > 3*PI/4)
                        continue;

                    // update velocity
                    curObjects[i].velocity_x = 0.8*objects[j].velocity_x + 0.2*v_x;
                    curObjects[i].velocity_y = 0.8*objects[j].velocity_y + 0.2*v_y;
                    // update path
                    curObjects[i].path = objects[j].path;
                    curObjects[i].path.push_back(curObjects[i].center);
                    break;
                }
            }
        }

        objects = curObjects;
    }
};

void training(VideoCapture &vid, BackgroundModel &background){
    cout << "---------------------- Training Phase ------------------------" << endl << endl;
    cout << "  Training Progress    Processing Speed          ETA" << endl;

    long long t0 = cv::getTickCount(), t1;
    int total_frame = vid.get(CV_CAP_PROP_FRAME_COUNT);
    double fps = vid.get(CV_CAP_PROP_FPS);

    int f = 0;
    while(vid.grab()){
        Mat frame;
        vid.retrieve(frame);

        background.learn(frame);
        imshow("Learning Progress", frame);
        waitKey(1);
        f++;

        if(f%10 == 0){
            double cur_fps = 10.0/(cv::getTickCount()-t1)*cv::getTickFrequency();
            int eta = (total_frame-f)/cur_fps;
            printf("\r         %d %%  \t\t   %0.2lf FPS \t     %d min %d sec         ",
                f*100/total_frame, cur_fps, eta/60, eta%60);
            fflush(stdout);
            t1 = cv::getTickCount();
        }
    }

    vid.release();
    imshow("Learned Background", background.get_background_model());

    cout << endl << endl;
    cout << "Training Phase Completed\nTotal Elapse Time: ";
    int elapsetime = (int)((cv::getTickCount()-t0)/cv::getTickFrequency());
    printf("%d min %d sec \n\n", elapsetime/60, elapsetime%60);

    cout << "Press Any Key to Continue" << endl;
    waitKey(0);
}

void testing(VideoCapture &vid, BackgroundModel &background){
    cout << "----------------------- Testing Phase ------------------------" << endl << endl;
    cout << "   Testing Progress    Processing Speed          ETA" << endl;

    int total_frame = vid.get(CV_CAP_PROP_FRAME_COUNT);
    double fps = vid.get(CV_CAP_PROP_FPS);
    long long t0 = cv::getTickCount(), t1;
    TrajectoryModel trajectoryModel;
    int f = 0;
    while(vid.grab()){
        Mat frame;
        vid.retrieve(frame);
        imshow("Original Video", frame);

        Mat foregroundMask = background.get_foreground_mask(frame);
        imshow("Forground Mask", 255-foregroundMask);

        Mat labels, stats, centroids;
        Mat strucElement = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
        morphologyEx(foregroundMask, foregroundMask, MORPH_CLOSE, strucElement);
        morphologyEx(foregroundMask, foregroundMask, MORPH_OPEN, strucElement);
        morphologyEx(foregroundMask, foregroundMask, MORPH_OPEN, strucElement);
        strucElement = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
        morphologyEx(foregroundMask, foregroundMask, MORPH_CLOSE, strucElement);
        imshow("Morphology Transform", 255-foregroundMask);
        int nLabels = connectedComponentsWithStats(foregroundMask, labels, stats, centroids);

        vector<ForegroundObject> objects;
        for(int i = 1; i < nLabels; i++){
            if(stats.at<int>(i, CC_STAT_AREA) < 20)
                continue;

            int x = stats.at<int>(i, CC_STAT_LEFT);
            int y = stats.at<int>(i, CC_STAT_TOP);
            int w = stats.at<int>(i, CC_STAT_WIDTH);
            int h = stats.at<int>(i, CC_STAT_HEIGHT);

            ForegroundObject curObject(Point(centroids.at<double>(i, 0), centroids.at<double>(i, 1)),
                stats.at<int>(i, CC_STAT_AREA), x, y, h, w);

            objects.push_back(curObject);
        }

        trajectoryModel.computeVelocity(objects);

        for(int i = 0; i < objects.size(); i++)
            objects[i].draw(frame);

        imshow("Original Video with Bounding Box", frame);

        double sec = (cv::getTickCount()-t0)/cv::getTickFrequency();
        if(sec < 1.0/fps)
            waitKey(1000.0/fps - sec*1000 + 1);
        else
            waitKey(1);

        f++;
        if(f%10 == 0){
            double cur_fps = 10.0/(cv::getTickCount()-t1)*cv::getTickFrequency();
            int eta = (total_frame-f)/cur_fps;
            printf("\r         %d %%  \t\t   %0.2lf FPS \t     %d min %d sec         ",
                f*100/total_frame, cur_fps, eta/60, eta%60);
            fflush(stdout);
            t1 = cv::getTickCount();
        }
    }

    cout << endl << endl;
    cout << "Testing Phase Completed\nTotal Elapse Time: ";
    int elapsetime = (int)((cv::getTickCount()-t0)/cv::getTickFrequency());
    printf("%d min %d sec \n\n", elapsetime/60, elapsetime%60);

}

int main(int argc, char *argv[]){
    String filename = "parking.mp4";
    if(argc > 1){
        filename = argv[1];
    }
    VideoCapture vid(filename);
    if(!vid.isOpened() || !vid.grab())
        return -1;

    int v_height = vid.get(CV_CAP_PROP_FRAME_HEIGHT);
    int v_width = vid.get(CV_CAP_PROP_FRAME_WIDTH);
    int total_frame = vid.get(CV_CAP_PROP_FRAME_COUNT);
    double fps = vid.get(CV_CAP_PROP_FPS);

    cout << endl;
    cout << "--------------------- Video Information ----------------------" << endl;
    cout << "Video Name: " << filename << endl;
    cout << "Video Frame: " << total_frame << endl;
    cout << "Video FPS: " << fps << endl;
    cout << "Video Length: "; printf("%d min %d sec\n", (int)(total_frame/fps/60), (int)(total_frame/fps)%60);
    cout << "Video Dimension: " << v_width << " x " << v_height << endl;
    cout << endl;

    BackgroundModel background(v_height, v_width);

    training(vid, background);
    cout << endl << endl;

    vid.open(filename);
    if(!vid.isOpened() || !vid.grab())
        return -1;

    testing(vid, background);

    return 0;
}