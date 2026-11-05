// rtabmap_bridge: RTAB-Map's own stereo odometry, stereo depth, loop closure and persistent map, driven over a local TCP socket by rtabmap_lab/live.py.
//
// Python -> bridge (little endian, [u32 type][u32 length][payload]):
//   1 INIT   f64 fx fy cx cy baseline | i32 width height | i32 cloud_stride | f32 min_range max_range | i32 disp_every | i32 lost_reset | f32 max_depth_std
//   2 FRAME  i32 id | f64 stamp | left BGR (w*h*3) | right gray (w*h)      both RECTIFIED, same size as INIT
//   3 QUIT   save the database and exit
// bridge -> Python:
//   101 STATUS    per frame, sent by the odometry thread right after the odometry (pose, features, inliers...). The map update runs on its own thread, as in RTAB-Map's own
//                 applications, so a slow map update never makes the odometry miss pairs; its results (node added, loop closure, timings) ride on the next STATUS, once.
//   102 DISPARITY i32 rows cols | u8 rows*cols (disparity * 4, 0 = no match)
//   103 NODE      i32 id | f32 pose[12] (optimised, base frame) | i32 n | n * (f32 x y z in the LEFT OPTICAL frame, u8 b g r)
//   104 GRAPH     i32 nposes | nposes * (i32 id, f32 pose[12]) | i32 nlinks | nlinks * (i32 from, i32 to, i32 type)
//   105 LOG       text
//   106 READY     i32 nodes already in the database
//   107 DEPTH     i32 node id | i32 frame id (the pair that made the node, -1 for a node loaded from the database) | i32 rows cols | u16 depth in millimetres (0 = none) rows*cols | u8 b g r rows*cols*3 | u8 right picture, gray, rows*cols   the node's pictures and its dense depth: the TSDF, and any other depth source run on the same pair
// Poses are RTAB-Map's base frame: x forward, y left, z up (right handed). The camera (optical) frame hangs off it with the local transform
// CameraModel::opticalRotation(), which is what makes a level, upright camera give a z-up map.
#include <winsock2.h>
#include <ws2tcpip.h>

#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/StereoCameraModel.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/Link.h>
#include <rtabmap/core/util2d.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/utilite/UFile.h>

#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace rtabmap;

struct Buf {
    std::vector<uint8_t> d;
    template <class T> void put(T v) { const uint8_t *p = reinterpret_cast<const uint8_t *>(&v); d.insert(d.end(), p, p + sizeof(T)); }
    void bytes(const void *p, size_t n) { const uint8_t *q = static_cast<const uint8_t *>(p); d.insert(d.end(), q, q + n); }
    void pose(const Transform &t) {
        if (t.isNull()) { for (int i = 0; i < 12; ++i) put<float>(0.f); return; }
        for (int i = 0; i < 12; ++i) put<float>(t.data()[i]);
    }
};

static SOCKET g_sock = INVALID_SOCKET;

static bool recvAll(void *p, size_t n) {
    char *c = static_cast<char *>(p);
    while (n) { int r = recv(g_sock, c, (int)(n > (1 << 20) ? (1 << 20) : n), 0); if (r <= 0) return false; c += r; n -= (size_t)r; }
    return true;
}
static std::mutex g_sendMu;
static bool sendMsg(uint32_t type, const Buf &b) {
    std::lock_guard<std::mutex> lk(g_sendMu);
    uint32_t h[2] = {type, (uint32_t)b.d.size()};
    const char *c = reinterpret_cast<const char *>(h);
    size_t n = 8;
    while (n) { int r = send(g_sock, c, (int)n, 0); if (r <= 0) return false; c += r; n -= (size_t)r; }
    c = reinterpret_cast<const char *>(b.d.data());
    n = b.d.size();
    while (n) { int r = send(g_sock, c, (int)(n > (1 << 20) ? (1 << 20) : n), 0); if (r <= 0) return false; c += r; n -= (size_t)r; }
    return true;
}
static void sendLog(const std::string &s) { Buf b; b.bytes(s.data(), s.size()); sendMsg(105, b); }

struct Cfg { double fx, fy, cx, cy, baseline; int w, h, stride; float minRange, maxRange; int dispEvery, lostReset; float maxDepthStd; };

// Block-matching disparity is noisy: 1 px of error at 6 m is +-1.4 m with a 42 mm baseline. Three cleanups before the picture is used for the map and the TSDF (the odometry does not use it):
//   speckles (small islands that jump from their surroundings), pixels that disagree with their 5x5 median by more than 1 px, and pixels whose expected depth noise sigma_z = z^2 * 0.3 px / (fx * B) is above max_depth_std.
static cv::Mat cleanDisparity(cv::Mat d, const Cfg &c, double fx, double baseline) {
    cv::Mat d16;
    cv::Mat valid = d > 0.5f;
    cv::Mat pos(d.size(), CV_32F, cv::Scalar(0));
    d.copyTo(pos, valid);
    pos.convertTo(d16, CV_16S, 16.0);
    cv::filterSpeckles(d16, 0, 120, 16);
    d16.convertTo(d, CV_32F, 1.0 / 16.0);
    cv::Mat med; cv::medianBlur(d, med, 5);
    cv::Mat bad = cv::abs(d - med) > 1.0f;
    d.setTo(0, bad);
    if (c.maxDepthStd > 0.f) {
        const float dmin = (float)std::sqrt(fx * baseline * 0.3 / c.maxDepthStd);
        d.setTo(0, d < dmin);
    }
    return d;
}

// RTAB-Map's own stereo matching (util2d::disparityFromStereoImages, the StereoBM/Stereo parameters), as float pixels; <= 0 = no match
static cv::Mat disparity(const cv::Mat &leftGray, const cv::Mat &rightGray, const ParametersMap &params, const Cfg &c, double fx, double baseline) {
    cv::Mat d = util2d::disparityFromStereoImages(leftGray, rightGray, params);
    if (d.type() == CV_16SC1) { cv::Mat f; d.convertTo(f, CV_32F, 1.0 / 16.0); d = f; }
    else if (d.type() != CV_32FC1) { cv::Mat f; d.convertTo(f, CV_32F); d = f; }
    return cleanDisparity(d, c, fx, baseline);
}

static void sendDisparity(const cv::Mat &disp) {
    cv::Mat u8;
    disp.convertTo(u8, CV_8U, 4.0);                                   // 0..63 px at 1/4 px steps
    Buf b; b.put<int32_t>(u8.rows); b.put<int32_t>(u8.cols);
    for (int r = 0; r < u8.rows; ++r) b.bytes(u8.ptr(r), (size_t)u8.cols);
    sendMsg(102, b);
}

// points of one node in the left optical frame, from the disparity: z = fx * baseline / d
static void sendNode(int id, const Transform &pose, const cv::Mat &disp, const cv::Mat &leftBGR, double fx, double fy, double cx, double cy, double baseline, const Cfg &c) {
    Buf pts; int n = 0;
    cv::Mat color = leftBGR;
    if (color.channels() == 1) cv::cvtColor(leftBGR, color, cv::COLOR_GRAY2BGR);
    for (int v = 0; v < disp.rows; v += c.stride) {
        const float *dr = disp.ptr<float>(v);
        for (int u = 0; u < disp.cols; u += c.stride) {
            float d = dr[u];
            if (!(d > 0.5f)) continue;
            float z = (float)(fx * baseline / d);
            if (z < c.minRange || z > c.maxRange) continue;
            pts.put<float>((float)((u - cx) * z / fx)); pts.put<float>((float)((v - cy) * z / fy)); pts.put<float>(z);
            const cv::Vec3b px = color.at<cv::Vec3b>(v, u);
            pts.put<uint8_t>(px[0]); pts.put<uint8_t>(px[1]); pts.put<uint8_t>(px[2]);
            ++n;
        }
    }
    Buf b; b.put<int32_t>(id); b.pose(pose); b.put<int32_t>(n); b.bytes(pts.d.data(), pts.d.size());
    sendMsg(103, b);
}

// dense depth of a node (z = fx * baseline / disparity, millimetres) with its colour picture: what the TSDF in Python integrates
static void sendDepth(int id, int frameId, const cv::Mat &disp, const cv::Mat &leftBGR, const cv::Mat &rightRaw, double fx, double baseline, const Cfg &c) {
    cv::Mat color = leftBGR;
    if (color.channels() == 1) cv::cvtColor(leftBGR, color, cv::COLOR_GRAY2BGR);
    cv::Mat depth(disp.rows, disp.cols, CV_16UC1, cv::Scalar(0));
    for (int v = 0; v < disp.rows; ++v) {
        const float *dr = disp.ptr<float>(v);
        uint16_t *out = depth.ptr<uint16_t>(v);
        for (int u = 0; u < disp.cols; ++u) {
            float d = dr[u];
            if (!(d > 0.5f)) continue;
            float z = (float)(fx * baseline / d);
            if (z >= c.minRange && z <= c.maxRange) out[u] = (uint16_t)(z * 1000.f);
        }
    }
    Buf b; b.put<int32_t>(id); b.put<int32_t>(frameId); b.put<int32_t>(depth.rows); b.put<int32_t>(depth.cols);
    for (int r = 0; r < depth.rows; ++r) b.bytes(depth.ptr(r), (size_t)depth.cols * 2);
    for (int r = 0; r < color.rows; ++r) b.bytes(color.ptr(r), (size_t)color.cols * 3);
    cv::Mat rg = rightRaw.channels() == 1 ? rightRaw : cv::Mat();
    if (rg.empty()) cv::cvtColor(rightRaw, rg, cv::COLOR_BGR2GRAY);
    for (int r = 0; r < rg.rows; ++r) b.bytes(rg.ptr(r), (size_t)rg.cols);
    sendMsg(107, b);
}

static void sendGraph(Rtabmap &rtabmap) {
    std::map<int, Transform> poses; std::multimap<int, Link> links;
    rtabmap.getGraph(poses, links, true, false);
    Buf b; b.put<int32_t>((int32_t)poses.size());
    for (auto &p : poses) { b.put<int32_t>(p.first); b.pose(p.second); }
    int nl = 0; Buf lb;
    for (auto &l : links) {
        if (l.second.from() == l.second.to()) continue;
        lb.put<int32_t>(l.second.from()); lb.put<int32_t>(l.second.to()); lb.put<int32_t>((int32_t)l.second.type()); ++nl;
    }
    b.put<int32_t>(nl); b.bytes(lb.d.data(), lb.d.size());
    sendMsg(104, b);
}

// nodes already in a loaded database: their clouds, so a saved map shows up again
static int replayDatabase(Rtabmap &rtabmap, const ParametersMap &params, const Cfg &c) {
    std::map<int, Transform> poses; std::multimap<int, Link> links; std::map<int, Signature> sigs;
    rtabmap.getGraph(poses, links, true, true, &sigs, true, false, false, false, false, false);
    int sent = 0;
    for (auto &s : sigs) {
        auto it = poses.find(s.first);
        if (it == poses.end() || it->second.isNull()) continue;
        SensorData &sd = s.second.sensorData();
        sd.uncompressData();
        if (sd.imageRaw().empty() || sd.depthOrRightRaw().empty() || sd.stereoCameraModels().empty()) continue;
        cv::Mat gray = sd.imageRaw().channels() == 1 ? sd.imageRaw() : cv::Mat();
        if (gray.empty()) cv::cvtColor(sd.imageRaw(), gray, cv::COLOR_BGR2GRAY);
        const StereoCameraModel &m = sd.stereoCameraModels()[0];
        cv::Mat d = disparity(gray, sd.depthOrRightRaw(), params, c, m.left().fx(), m.baseline());
        sendNode(s.first, it->second, d, sd.imageRaw(), m.left().fx(), m.left().fy(), m.left().cx(), m.left().cy(), m.baseline(), c);
        sendDepth(s.first, -1, d, sd.imageRaw(), sd.depthOrRightRaw(), m.left().fx(), m.baseline(), c);
        ++sent;
    }
    if (sent) sendGraph(rtabmap);
    return (int)sigs.size();
}

struct MapJob { SensorData data; Transform pose; cv::Mat cov, leftBGR; int id = 0; bool valid = false; };
struct MapResult { int added = 0, loopId = 0, proxId = 0, nodes = 0, nLoops = 0; float mapMs = 0.f, dispMs = 0.f; Transform corr = Transform::getIdentity(); };
static std::mutex g_jobMu, g_resMu;
static std::condition_variable g_jobCv;
static MapJob g_job;
static bool g_stop = false;
static MapResult g_res;

// the map update, on its own thread: RTAB-Map's loop closure detection, graph optimisation and database, plus the dense depth of each new node
static void mapLoop(Rtabmap &rtabmap, const ParametersMap &params, const Cfg &c) {
    int nMap = 0, nLoops = 0;
    for (;;) {
        MapJob job;
        {
            std::unique_lock<std::mutex> lk(g_jobMu);
            g_jobCv.wait(lk, [] { return g_job.valid || g_stop; });
            if (!g_job.valid) return;
            job = std::move(g_job); g_job.valid = false;
        }
        UTimer t;
        bool added = false;
        try { added = rtabmap.process(job.data, job.pose, job.cov); } catch (const std::exception &e) { sendLog(std::string("rtabmap: ") + e.what()); }
        const float mapMs = (float)(t.ticks() * 1000.0);
        const int loopId = rtabmap.getLoopClosureId(), proxId = rtabmap.getStatistics().proximityDetectionId();
        if (loopId > 0 || proxId > 0) { ++nLoops; sendLog("loop closure: node " + std::to_string(rtabmap.getStatistics().refImageId()) + " with node " + std::to_string(loopId > 0 ? loopId : proxId) + (loopId > 0 ? " (appearance)" : " (proximity)")); }
        float dispMs = 0.f; cv::Mat disp;
        const bool wantDisp = (c.dispEvery > 0 && nMap % c.dispEvery == 0) || added;
        if (wantDisp) {
            cv::Mat gray; cv::cvtColor(job.leftBGR, gray, cv::COLOR_BGR2GRAY);
            UTimer td; disp = disparity(gray, job.data.depthOrRightRaw(), params, c, c.fx, c.baseline); dispMs = (float)(td.ticks() * 1000.0);
            if (c.dispEvery > 0 && nMap % c.dispEvery == 0) sendDisparity(disp);
        }
        if (added) {
            const int nodeId = rtabmap.getLastLocationId();
            auto &opt = rtabmap.getLocalOptimizedPoses();
            auto it = opt.find(nodeId);
            sendNode(nodeId, it != opt.end() ? it->second : rtabmap.getMapCorrection() * job.pose, disp, job.leftBGR, c.fx, c.fy, c.cx, c.cy, c.baseline, c);
            sendDepth(nodeId, job.id, disp, job.leftBGR, job.data.depthOrRightRaw(), c.fx, c.baseline, c);
        }
        if (added || loopId > 0 || proxId > 0) sendGraph(rtabmap);
        {
            std::lock_guard<std::mutex> lk(g_resMu);
            g_res.added |= added ? 1 : 0; if (loopId > 0) g_res.loopId = loopId; if (proxId > 0) g_res.proxId = proxId;
            g_res.mapMs = mapMs; g_res.dispMs = dispMs; g_res.corr = rtabmap.getMapCorrection();
            g_res.nodes = (int)rtabmap.getLocalOptimizedPoses().size(); g_res.nLoops = nLoops;
        }
        ++nMap;
    }
}

int main(int argc, char *argv[]) {
    int port = 5710; std::string db, ini; bool localize = false;
    ParametersMap params = Parameters::getDefaultParameters();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--db" && i + 1 < argc) db = argv[++i];
        else if (a == "--config" && i + 1 < argc) ini = argv[++i];
        else if (a == "--localize") localize = true;
    }
    if (!ini.empty()) Parameters::readINI(ini, params);
    for (int i = 1; i + 1 < argc; ++i) {                                  // --Group/Name value: any RTAB-Map parameter, as in the RTAB-Map tools
        std::string a = argv[i];
        if (a.rfind("--", 0) == 0 && params.count(a.substr(2))) params[a.substr(2)] = argv[++i];
    }
    ULogger::setType(ULogger::kTypeConsole);
    ULogger::setLevel(ULogger::kWarning);
    params[Parameters::kMemIncrementalMemory()] = localize ? "false" : "true";
    if (localize) params[Parameters::kMemInitWMWithAllNodes()] = "true";

    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET ls = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((u_short)port); inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    if (bind(ls, (sockaddr *)&addr, sizeof(addr)) || listen(ls, 1)) { std::printf("cannot listen on 127.0.0.1:%d\n", port); return 2; }
    std::printf("rtabmap_bridge %s listening on 127.0.0.1:%d db=%s %s\n", RTABMAP_VERSION, port, db.empty() ? "(memory)" : db.c_str(), localize ? "localization only" : "mapping"); std::fflush(stdout);
    g_sock = accept(ls, nullptr, nullptr);
    setsockopt(g_sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));

    uint32_t h[2];
    if (!recvAll(h, 8) || h[0] != 1) { std::printf("expected INIT\n"); return 3; }
    std::vector<uint8_t> p(h[1]); recvAll(p.data(), p.size());
    Cfg c{}; size_t o = 0;
    auto get = [&](auto &v) { std::memcpy(&v, p.data() + o, sizeof(v)); o += sizeof(v); };
    get(c.fx); get(c.fy); get(c.cx); get(c.cy); get(c.baseline); get(c.w); get(c.h); get(c.stride); get(c.minRange); get(c.maxRange); get(c.dispEvery); get(c.lostReset); get(c.maxDepthStd);

    Rtabmap rtabmap;
    rtabmap.init(params, db);
    std::unique_ptr<Odometry> odom(Odometry::create(params));
    StereoCameraModel model("rtabmap_lab", c.fx, c.fy, c.cx, c.cy, c.baseline, CameraModel::opticalRotation(), cv::Size(c.w, c.h));
    int existing = 0;
    try { existing = replayDatabase(rtabmap, params, c); } catch (const std::exception &e) { sendLog(std::string("could not show the nodes of the loaded database: ") + e.what()); }
    { Buf b; b.put<int32_t>(existing); sendMsg(106, b); }

    int lostCount = 0, nResets = 0, nFrames = 0; Transform lastGood = Transform::getIdentity(), sinceGood = Transform::getIdentity();
    std::thread mapper(mapLoop, std::ref(rtabmap), std::cref(params), std::cref(c));
    for (;;) {
        if (!recvAll(h, 8)) break;
        std::vector<uint8_t> msg(h[1]);
        if (h[1] && !recvAll(msg.data(), msg.size())) break;
        if (h[0] == 3) break;
        if (h[0] != 2) continue;
        int32_t id; double stamp;
        std::memcpy(&id, msg.data(), 4); std::memcpy(&stamp, msg.data() + 4, 8);
        const size_t npx = (size_t)c.w * c.h;
        const bool hasGyro = msg.size() == 12 + npx * 4 + 24;           // optional: rotation since the previous frame from an IMU (rotation vector, rad, optical frame)
        if (msg.size() != 12 + npx * 4 && !hasGyro) { sendLog("frame of the wrong size, skipped"); continue; }
        Transform gyroBase;                                                // the same rotation in RTAB-Map's base frame (x forward, y left, z up)
        if (hasGyro) {
            double rv[3]; std::memcpy(rv, msg.data() + 12 + npx * 4, 24);
            cv::Mat R; cv::Rodrigues(cv::Mat(3, 1, CV_64F, rv), R);
            const Transform Ropt(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), 0, R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), 0,
                                 R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), 0);
            const Transform opt = CameraModel::opticalRotation();
            gyroBase = opt * Ropt * opt.inverse();
            sinceGood = sinceGood * gyroBase;
        }
        cv::Mat leftBGR(c.h, c.w, CV_8UC3, msg.data() + 12), right(c.h, c.w, CV_8UC1, msg.data() + 12 + npx * 3), gray;
        cv::cvtColor(leftBGR, gray, cv::COLOR_BGR2GRAY);
        UTimer total;
        SensorData data(gray, right.clone(), model, id, stamp);
        OdometryInfo info;
        Transform pose;
        try { pose = odom->process(data, &info); } catch (const std::exception &e) { sendLog(std::string("odometry: ") + e.what()); }
        const float odomMs = (float)(total.ticks() * 1000.0);
        if (!pose.isNull()) {
            lastGood = pose; lostCount = 0; sinceGood = Transform::getIdentity();
            MapJob job;
            job.data = data; job.pose = pose; job.id = id; job.valid = true; job.leftBGR = leftBGR.clone();
            job.cov = info.reg.covariance.empty() ? cv::Mat::eye(6, 6, CV_64FC1) * 0.01 : info.reg.covariance;
            { std::lock_guard<std::mutex> lk(g_jobMu); g_job = std::move(job); }   // the newest wins: the map update works at its own rate (Rtabmap/DetectionRate), older pairs are not queued
            g_jobCv.notify_one();
        } else if (++lostCount >= c.lostReset) {                            // the odometry has been lost for a while: start it again where it last was
            // turned by the gyro's rotation since then: the rig usually loses tracking WHILE turning, and restarting at the old heading put every later wall at the wrong angle
            const Transform restart = hasGyro ? lastGood * sinceGood : lastGood;
            odom->reset(restart); lostCount = 0; ++nResets;
            if (hasGyro) { lastGood = restart; sinceGood = Transform::getIdentity(); }
            sendLog("odometry lost for " + std::to_string(c.lostReset) + " frames: restarted at the last good pose" + (hasGyro ? " turned by the gyro" : ""));
        }
        MapResult r;
        { std::lock_guard<std::mutex> lk(g_resMu); r = g_res; g_res.added = g_res.loopId = g_res.proxId = 0; g_res.mapMs = g_res.dispMs = 0.f; }
        Buf b;
        b.put<int32_t>(id); b.put<double>(stamp); b.put<int32_t>(pose.isNull() ? 1 : 0); b.pose(pose); b.pose(r.corr);
        b.put<int32_t>(info.features); b.put<int32_t>(info.reg.inliers); b.put<int32_t>(info.reg.matches); b.put<int32_t>(info.localMapSize);
        b.put<float>(odomMs); b.put<float>(r.mapMs); b.put<float>(r.dispMs);
        b.put<int32_t>(r.added); b.put<int32_t>(r.loopId); b.put<int32_t>(r.proxId);
        b.put<int32_t>(r.nodes); b.put<int32_t>(r.nLoops); b.put<int32_t>(nResets);
        b.put<float>(info.distanceTravelled);
        if (!sendMsg(101, b)) break;
        ++nFrames;
    }
    { std::lock_guard<std::mutex> lk(g_jobMu); g_stop = true; }
    g_jobCv.notify_one();
    mapper.join();
    int nLoops; { std::lock_guard<std::mutex> lk(g_resMu); nLoops = g_res.nLoops; }
    std::printf("closing: %d frames, %d loop closures, %d odometry restarts%s\n", nFrames, nLoops, nResets, db.empty() ? "" : ", saving the database");
    try { rtabmap.close(!localize); } catch (...) {}
    closesocket(g_sock); WSACleanup();
    return 0;
}
