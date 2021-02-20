#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <bits/stdc++.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>


std::string node_name = "detect_target";    // デフォルトノード名
std::string camera_topic_name = "/cv_camera/image_raw";
int counter = 0;
double camera_offset = 0.51;                // 駆動軸からカメラまでの距離 [m]

class ImageConverter
{
    ros::NodeHandle nh_;
    ros::Publisher angle_pub_;              // 角度
    ros::Publisher dist_pub_;               // 距離
    ros::Publisher valid_pub_;              // 有効／無効
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
public:
    ImageConverter() : it_(nh_)
    {
        nh_.getParam(node_name + "/offset", camera_offset);     // offsetパラメータの取得
        nh_.getParam(node_name + "/camera", camera_topic_name); // カメラ画像のノード名
        std::cout << "[" + node_name + "] カメラオフセット=" << camera_offset << std::endl;
        // 入力ビデオフィードのサブスクライブと出力ビデオフィードのパブリッシュ
        image_sub_ = it_.subscribe(camera_topic_name, 1, &ImageConverter::imageCb, this);
        image_pub_ = it_.advertise(node_name + "/monitor", 1);
        angle_pub_ = nh_.advertise<std_msgs::Float32>(node_name + "/angle", 1);
        dist_pub_ = nh_.advertise<std_msgs::Float32>(node_name + "/distance", 1);
        valid_pub_ = nh_.advertise<std_msgs::Bool>(node_name + "/valid", 1);
    }
    ~ImageConverter()
    {
        //cv::destroyWindow(OPENCV_WINDOW); // デスクトップ版ではないので使えない
    }

    // 入力画像サブスクライバコールバック
    void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
        ros::Time ros_start = ros::Time::now(); // 時間計測開始

        cv_bridge::CvImagePtr cv_mono;
        cv_bridge::CvImagePtr cv_dst;
        cv_bridge::CvImage img_bridge;  // 中間イメージ
        sensor_msgs::Image img_msg;     // 送信用メッセージ
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv_mono = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception& e)
        {
            std::cout << "[" + node_name + "] 画像を取得できませんでした";
            return;
        }

        // 画像をカットする
        int width_c = cv_mono->image.cols;
        int height_c = cv_mono->image.rows / 3;
        cv::Mat cut_image;
        cut_image = cv::Mat(cv_mono->image, cv::Rect(0, height_c, width_c, height_c));
        // モニタ用に 3 チャネル画像を作成
        cv::Mat color_src01;
        cv::cvtColor(cut_image, color_src01, cv::COLOR_GRAY2BGR);
        // 大津の二値化
        cv::Mat binary;
        cv::threshold(cut_image, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        // ラベリング
        cv::Mat labelImage(binary.size(), CV_32S);   // 結果画像
        cv::Mat statres;                                // Left, Top, Width, Height, Area, Max
        cv::Mat centroidsres;                           // 重心座標
        int nLabels = cv::connectedComponentsWithStats(binary, labelImage, statres, centroidsres, 8, 4); // ラベリング

        struct _singleTarget {
            int index = 0;
            int x = 0;
            int y = 0;
            int height = 0;
            int width = 0;
            double cx = 0.0;
            double cy = 0.0;
            double area = 0.0;
            double score = 0.0;
        };
        struct _singleTarget targets[nLabels];
        int cc = 0;  // 候補の数カウンタ

        // 検出したラベルからターゲットを絞り込む
        for (int i = 1; i < nLabels; ++i) {
            // 矩形座標
            int *rects = statres.ptr<int>(i);
            int rx = rects[cv::ConnectedComponentsTypes::CC_STAT_LEFT];
            int ry = rects[cv::ConnectedComponentsTypes::CC_STAT_TOP];
            int rheight = rects[cv::ConnectedComponentsTypes::CC_STAT_HEIGHT];
            int rwidth = rects[cv::ConnectedComponentsTypes::CC_STAT_WIDTH];
            // 重心
            double *centers = centroidsres.ptr<double>(i);
            double cx = centers[0];
            double cy = centers[1];
            // 面積
            int *areas = statres.ptr<int>(i);
            int s = areas[cv::ConnectedComponentsTypes::CC_STAT_AREA];
            // 縦横 4 ドット未満なら対象外
            if (rwidth < 4 || rheight < 4) {
                continue;
            }
            // フェレ径を用いて円形度を算出
            double circularity = s * 4.0 / (rwidth * rheight * M_PI);
            if (circularity > 1.0) {
                circularity = 1.0 / circularity; 
            }
            // 縦横比を加味
            double flatting = double(rheight) / rwidth;
            if (flatting > 1) {
                flatting = 1.0 / flatting;
            }
            circularity *= flatting;
            // 上下方向中心からの距離
            double half = binary.rows / 2.0;
            double level = 1.0 - std::abs((cy - half) / half);
            circularity *= level;

            // 閾値処理
            cv::Scalar color = cv::Scalar(0, 0, 255);   // デフォルト色：赤
            if (circularity > 0.6) {
                color = cv::Scalar(255, 0, 0);  // 青
                targets[cc].index = i;
                targets[cc].score = circularity;
                targets[cc].x = rx;
                targets[cc].y = ry;
                targets[cc].width = rwidth;
                targets[cc].height = rheight;
                targets[cc].cx = cx;
                targets[cc].cy = cy;
                targets[cc].area = s;
                cc++;
            }
            // 矩形描画
            cv::rectangle(color_src01, cv::Rect(rx, ry, rwidth, rheight), color, 1);
        }

        // ペア検出ループ
        double max_score = 0.0;
        int element1 = -1;
        int element2 = -1;

        for (int p1 = 0; p1 < cc - 1; p1++) {
            for (int p2 = p1 + 1; p2 < cc; p2++) {
                double score = targets[p1].score * targets[p2].score;
                score = 1.0;    // テスト用。本番では削除*******************************************************
                // 水平度からスコアを補正
                if (targets[p1].cx == targets[p2].cx) {
                    std::cout << "[" + node_name + "] x座標が同一です. p1=" << targets[p1].cx << ", p2=" << targets[p2].cx << std::endl;
                    continue;   // x 座標が同一なら対象外
                }
                double angle = 1.0 - std::abs((targets[p1].cy - targets[p2].cy) * 5.0 / (targets[p1].cx - targets[p2].cx));
                score *= angle;
                // 面積比からスコアを補正
                double a_ratio = targets[p1].area / targets[p2].area;
                if (a_ratio > 1.0) {
                    a_ratio = 1.0 / a_ratio;
                }
                score *= a_ratio;
                // 距離からスコアを補正
                double span_r = std::abs((targets[p1].cx - targets[p2].cx) / ((targets[p1].width + targets[p2].width) * 2.0));
                if (span_r > 1.0) {
                    span_r = 1.0 / span_r;
                }
                score *= span_r;

                // 閾値処理
                if (score < 0.6) {
                    continue;
                }

                if (score > max_score) {
                    max_score = score;  // 最大スコアを更新
                    element1 = p1;
                    element2 = p2;
                }
            }
        }
        if (element1 == -1 || element2 == -1){
            std_msgs::Bool valid;
            valid.data = false;
            valid_pub_.publish(valid);  // 無効をパブリッシュ
            std::cout << "[" + node_name + "] ターゲットが見つかりません。" << std::endl;
        }
        else{
            double span = distance(targets[element1].cx, targets[element1].cy, targets[element2].cx, targets[element2].cy);
            double direction = (targets[element1].cx + targets[element2].cx) / 2.0;

            std_msgs::Float32 distance;
            distance.data = convert_to_distance(span);  // ターゲット間隔を距離に変換
            std_msgs::Float32 angle;
            angle.data = convert_to_angle(direction, color_src01.cols);    // ターゲットx座標を角度に変換
            angle.data = adjust_angle(angle.data, distance.data, camera_offset);    // カメラ基準の角度を車軸基準の角度に変換

            angle_pub_.publish(angle);      // 角度をパブリッシュ
            dist_pub_.publish(distance);    // 距離をパブリッシュ
            std_msgs::Bool valid;
            valid.data = true;
            valid_pub_.publish(valid);      // 有効をパブリッシュ

            // 中心線の描画
            cv::line(color_src01, cv::Point(direction, 0), cv::Point(direction, color_src01.rows), cv::Scalar(0, 255, 0));
            // ターゲットの矩形を描画
            cv::rectangle(color_src01, cv::Rect(targets[element1].x, targets[element1].y, targets[element1].width, targets[element1].height), cv::Scalar(0, 255, 0), 2);
            cv::rectangle(color_src01, cv::Rect(targets[element2].x, targets[element2].y, targets[element2].width, targets[element2].height), cv::Scalar(0, 255, 0), 2);
        }

        // Mat を sensro_msgs::Image に変換してパブリッシュする
        std_msgs::Header header;            // 空のヘッダ
        counter++;                          // カウンタをインクリメント
        header.seq = counter;
        header.stamp = ros::Time::now();    // タイムスタンプ
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, color_src01);    // Mat から cv_bridge::CvImage を作成
        img_bridge.toImageMsg(img_msg);     // cv_bridge::CvImage を sensor_msgs::Image に変換
        image_pub_.publish(img_msg);

        // 処理時間の測定
        ros::Duration ros_duration =  ros::Time::now() - ros_start;
        //std::cout << "処理時間= " << (double)ros_duration.sec + ((double)ros_duration.nsec / 1000000000.0) << std::endl;
    }

    // 2 点の距離を計算する
    double distance(double x1, double y1, double x2, double y2) {
        return sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0));
    }

    // x 座標の値を角度に変換する
    double convert_to_angle(double x, int width) {
        return (x - (width / 2.0)) / 404.02;
    }

    // ターゲットの間隔を距離に変換する
    double convert_to_distance(double span) {
        double theta = span / (404.02 * 2.0);
        return 0.1 / std::tan(theta);
    }

    // カメラから見た角度を、車軸基準の角度に変換する
    double adjust_angle(double angle, double distance, double offset) {
        double d = distance * std::tan(angle);
        return std::atan(d / (distance + offset));
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, node_name);   // ノードの初期化
    node_name = ros::this_node::getName();      // ノード名の取得

    ImageConverter ic;  // 画像変換インスタンス

    std::cout << "[" + node_name + "] ノードを起動しました。" << std::endl;
    ros::spin();
    std::cout << "[" + node_name + "] ノードを終了しました。" << std::endl;
    return 0;
}

