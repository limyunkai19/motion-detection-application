#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Vec3b rgb2lab(Vec3b rgbColor){
    Mat rgb(1, 1, CV_8UC3, Scalar(rgbColor[0], rgbColor[1], rgbColor[2])), lab;
    cvtColor(rgb, lab, CV_BGR2Lab);

    return lab.at<Vec3b>(0, 0);
}

struct QuantizedBin{
    short quantity, lambda, last;
    Vec3b color;

    QuantizedBin(){
        lambda = 0;
        last = -1;
    }
    QuantizedBin(Vec3b color){
        lambda = 0;
        last = -1;
        quantity = 0;
        this->color = color;
    }

    bool operator< (const QuantizedBin x) const {
        return this->quantity < x.quantity;
    }
};

class CodeBook{
  private:
    int cur;
    bool isSorted;
    vector<QuantizedBin> colorBin;
    // colorBinIdx track the where the colorBin is after sorting
    vector<int> colorBinIdx;

    void binSort(){
        // sort reverse order
        sort(colorBin.rbegin(), colorBin.rend());

        for(int i = 0; i < colorBin.size(); i++){
            colorBinIdx[colorToIdx(colorBin[i].color)] = i;
        }
        isSorted = true;
    }

    int colorToIdx(Vec3b color){
        int l = color[0]/64, a = color[1]/8, b = color[2]/8;
        return 32*32*l + 32*a + b;
    }

  public:
    CodeBook(){
        cur = 0;

        colorBin.resize(4*32*32);
        colorBinIdx.resize(4*32*32);
        for(int i = 0; i < 4*32*32; i++){
            int bin = i;
            int b = bin%32; bin /= 32;
            int a = bin%32; bin /= 32;
            int l = bin;

            colorBin[i] = QuantizedBin(Vec3b(l*64+32, a*8+4, b*8+4));
            colorBinIdx[i] = i;
        }
        isSorted = true;
    }

    void update(Vec3b color){
        isSorted = false;
        int idx = colorBinIdx[colorToIdx(color)];

        // check colorBin[idx].color ==  color
        colorBin[idx].quantity++;
        colorBin[idx].lambda = max(colorBin[idx].lambda, cur-colorBin[idx].last);
        colorBin[idx].last = cur;
        cur++;
    }

    Vec3b get_color(){
        if(!isSorted)
            binSort();
        return colorBin[0].color;
    }

    // same = 1, diff = 0
    bool compare_color(Vec3b x, Vec3b y){
        if(abs(x[0]-y[0]) > 64)
            return false;
        if(abs(x[1]-y[1]) > 4)
            return false;
        if(abs(x[2]-y[2]) > 4)
            return false;
        return true;
    }

    bool is_foreground(Vec3b color){
        if(!isSorted)
            binSort();

        Vec3b dominant = colorBin[0].color;

        if(compare_color(dominant, color)){
            return false;
        }

        // variable !!!
        for(int i = 1; i < 10; i++){
            colorBin[i].lambda = max(colorBin[i].lambda, cur-colorBin[i].last);
            if(colorBin[i].lambda > 300)
                continue;
            if(compare_color(colorBin[i].color, color)){
                return false;
            }
        }
        return true;
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
    void learn(Mat rgbframe){
        Mat frame;
        cvtColor(rgbframe, frame, CV_BGR2Lab);

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
        cvtColor(background, background, CV_Lab2BGR);
        return background;
    }

    Mat get_foreground_mask(Mat rgbframe){
        Mat frame;
        cvtColor(rgbframe, frame, CV_BGR2Lab);

        Mat foreground(rows, cols, CV_8UC1);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                // mat.at<>(r, c);
                // is foreground return true(1) false(0), multiple 255 to get binary foreground image
                foreground.at<uchar>(i, j) = 255 *
                    pixelCodeBook[i*cols+j].is_foreground(frame.at<Vec3b>(i, j));
            }
        }
        return foreground;
    }
};

struct ForegroundObject{
    Point center;
    int area, x, y, h, w;
    double velocity_x, velocity_y;

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
        circle(img, center, 2, Scalar(255, 255, 0), -1);
        arrowedLine(img, center, Point(center.x + velocity_x*7, center.y + velocity_y*7),
            Scalar(255, 255, 0), 2, CV_AA, 0, 0.2);
    }
};

class ProjectileModel{
  private:
    vector<ForegroundObject> objects;

  public:
    void computeVelocity(vector<ForegroundObject> &curObjects){
        // for each curObject, try to match with a object
        for(int i = 0; i < curObjects.size(); i++){
            for(int j = 0; j < objects.size(); j++){
                if( abs(curObjects[i].center.x - objects[j].center.x) < 9 &&
                    abs(curObjects[i].center.y - objects[j].center.y) < 9){
                    //matched

                    double v_x = curObjects[i].center.x - objects[j].center.x;
                    double v_y = curObjects[i].center.y - objects[j].center.y;

                    // update velocity
                    curObjects[i].velocity_x = 0.8*objects[j].velocity_x + 0.2*v_x;
                    curObjects[i].velocity_y = 0.8*objects[j].velocity_y + 0.2*v_y;
                    break;
                }
            }
        }

        objects = curObjects;
    }
};

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
    long long t0 = cv::getTickCount(), t1;

    cout << endl;
    cout << "--------------------- Video Information ----------------------" << endl;
    cout << "Video Name: " << filename << endl;
    cout << "Video Frame: " << total_frame << endl;
    cout << "Video FPS: " << fps << endl;
    cout << "Video Length: "; printf("%d min %d sec\n", (int)(total_frame/fps/60), (int)(total_frame/fps)%60);
    cout << "Video Dimension: " << v_width << " x " << v_height << endl;
    cout << endl;

    cout << "---------------------- Training Phase ------------------------" << endl << endl;

    cout << "  Training Progress    Processing Speed          ETA" << endl;

    BackgroundModel background(v_height, v_width);

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
            printf("\r         %d %%  \t\t   %0.2lf FPS \t     %d min %d sec       ",
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


    cout << endl << endl;
    cout << "----------------------- Testing Phase ------------------------" << endl << endl;

    vid.open(filename);
    if(!vid.isOpened() || !vid.grab())
        return -1;

    // VideoWriter outputVideo;
    // outputVideo.open("mask.mp4", -1, 30, Size(v_width, v_height), false);

    ProjectileModel projectileModel;
    while(vid.grab()){
        Mat frame;
        vid.retrieve(frame);
        imshow("Original Video", frame);

        Mat foregroundMask = background.get_foreground_mask(frame);
        imshow("Forground Mask", foregroundMask);

        // outputVideo << foregroundMask;

        const int kernal1 = 1, kernal3 = 3;
        medianBlur(foregroundMask, foregroundMask, kernal3);
        threshold(foregroundMask, foregroundMask, 100, 255, THRESH_BINARY);
        imshow("Forground Mask Median Blur", foregroundMask);
        Mat element3 = getStructuringElement(MORPH_ELLIPSE, Size(2*kernal3+1, 2*kernal3+1),
                           Point(kernal3, kernal3));
        Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2*kernal1+1, 2*kernal1+1),
                           Point(kernal1, kernal1));
        dilate(foregroundMask, foregroundMask, element3);
        erode(foregroundMask, foregroundMask, element1);
        dilate(foregroundMask, foregroundMask, element1);
        // dilate(foregroundMask, foregroundMask, element3);
        imshow("Forground Mask Dilate", foregroundMask);

        GaussianBlur(foregroundMask, foregroundMask, Size(kernal3, kernal3), 0, 0);
        threshold(foregroundMask, foregroundMask, 100, 255, THRESH_BINARY);
        imshow("Forground Mask Gaussian Blur", foregroundMask);

        foregroundMask = foregroundMask;

        Mat labels, stats, centroids;
        int nLabels = connectedComponentsWithStats(foregroundMask, labels, stats, centroids);

        vector<ForegroundObject> objects;
        for(int i = 1; i < nLabels; i++){
            if(stats.at<int>(i, CC_STAT_AREA) < 50)
                continue;

            int x = stats.at<int>(i, CC_STAT_LEFT);
            int y = stats.at<int>(i, CC_STAT_TOP);
            int w = stats.at<int>(i, CC_STAT_WIDTH);
            int h = stats.at<int>(i, CC_STAT_HEIGHT);

            ForegroundObject curObject(Point(centroids.at<double>(i, 0), centroids.at<double>(i, 1)),
                stats.at<int>(i, CC_STAT_AREA), x, y, h, w);

            objects.push_back(curObject);
        }

        projectileModel.computeVelocity(objects);

        for(int i = 0; i < objects.size(); i++)
            objects[i].draw(frame);

        imshow("Original Video with Bounding Box", frame);
        waitKey(100);
    }


    return 0;
}