// sudo apt-get install libcpprest-dev
// g++ -std=c++11  motion_server.cpp -o motion_server -lcpprest -lssl -lcrypto -lboost_system
// test:
// curl -X POST http://192.168.2.100:34568 -H "Content-Type: application/json" -d '{"moveToCartesian": [0.5, 0, 0.3]}'

#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <iomanip>
#include <mutex>
#include "common.h"

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;

class RobotHandler {
    franka::Robot robot;
    franka::Gripper gripper;
    std::mutex robot_mutex;  // libfranka is not thread-safe; serialize all robot calls
    double default_motion_time_;
public:
    // kIgnore lets robot.control() run on a non-PREEMPT_RT kernel; trade-off
    // is missed-deadline warnings and slightly jerky cubic trajectories.
    RobotHandler(const std::string& ip_addr, double default_motion_time)
        : robot(ip_addr, franka::RealtimeConfig::kIgnore),
          gripper(ip_addr),
          default_motion_time_(default_motion_time) {
        // Clear any pre-existing error state left over from a previous run
        try {
            robot.automaticErrorRecovery();
        } catch (const franka::Exception& e) {
            // No error to recover from — this is fine
            std::cerr << "Startup error recovery (may be benign): " << e.what() << std::endl;
        }
        // Loosen reflex thresholds: kIgnore trajectories are noisier than RT
        // and trip Desk's defaults. Acceleration thresholds set near hardware
        // limits (87 Nm torque, 100 N force) to ignore transient spikes;
        // nominal thresholds left moderate so steady-state contact is caught.
        try {
            robot.setCollisionBehavior(
                {{87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0}},
                {{87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0}},
                {{30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0}},
                {{30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0}},
                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                {{50.0, 50.0, 50.0, 50.0, 50.0, 50.0}},
                {{50.0, 50.0, 50.0, 50.0, 50.0, 50.0}}
            );
        } catch (const franka::Exception& e) {
            std::cerr << "[ctor] setCollisionBehavior failed: " << e.what() << std::endl;
        }
    }

    // kIgnore-compatible cubic joint trajectory to home. Fallback for when
    // goHome() (RT-only) throws. Long tf keeps wrist angular velocity low
    // enough to avoid Cartesian force spikes during the homing swing.
    void goHomeJoints(double tf = 10.0) {
        const std::array<double, 7> q_home = {
            0.0, -M_PI_4, 0.0, -3.0 * M_PI_4, 0.0, M_PI_2, M_PI_4
        };
        std::array<double, 7> q_initial{};
        double time = 0.0;

        robot.control(
            [&time, &q_initial, q_home, tf]
            (const franka::RobotState& state, franka::Duration period) -> franka::JointPositions {
                time += period.toSec();
                if (time == 0.0) {
                    q_initial = state.q;
                }
                std::array<double, 7> q_target = q_initial;
                double t2 = pow(time, 2);
                double t3 = pow(time, 3);
                double tf2 = pow(tf, 2);
                double tf3 = pow(tf, 3);
                for (int i = 0; i < 7; ++i) {
                    double delta = q_home[i] - q_initial[i];
                    double a2 = 3.0 * delta / tf2;
                    double a3 = -2.0 * delta / tf3;
                    q_target[i] = q_initial[i] + a2 * t2 + a3 * t3;
                }
                franka::JointPositions output = q_target;
                if (time >= tf) {
                    std::cout << "[goHomeJoints] reached home pose" << std::endl;
                    return franka::MotionFinished(output);
                }
                return output;
            }
        );
    }

    void initialize() {
        // Try RT-only goHome() first, fall back to kIgnore-friendly
        // goHomeJoints() if it throws. If both fail, the rest of the server
        // still works -- the robot just stays where it was.
        bool homed = false;
        try {
            goHome(robot);
            homed = true;
        } catch (const franka::Exception& e) {
            std::cerr << "[initialize] goHome skipped: " << e.what() << std::endl;
        }
        if (!homed) {
            // Clear Reflex mode before the fallback; otherwise robot.control()
            // rejects with "command not possible in the current mode (Reflex)".
            try {
                robot.automaticErrorRecovery();
                std::cerr << "[initialize] cleared Reflex state, retrying..." << std::endl;
            } catch (const franka::Exception& e) {
                std::cerr << "[initialize] pre-fallback recovery failed: "
                          << e.what() << std::endl;
            }
            std::cerr << "[initialize] Falling back to goHomeJoints "
                      << "(cubic joint trajectory under kIgnore)..." << std::endl;
            try {
                goHomeJoints();
                homed = true;
            } catch (const franka::Exception& e) {
                std::cerr << "[initialize] goHomeJoints also failed: "
                          << e.what() << std::endl;
                try {
                    robot.automaticErrorRecovery();
                } catch (...) {}
                std::cerr << "[initialize] Continuing without homing." << std::endl;
            }
        }
        try {
            gripper.homing();
        } catch (const franka::Exception& e) {
            std::cerr << "[initialize] gripper.homing skipped: " << e.what() << std::endl;
        }

        franka::Model model = robot.loadModel();
        const franka::RobotState& robot_state = robot.readOnce();
        std::array<double, 16> initial_pose = robot_state.O_T_EE;
        for(int i = 0; i < 16; i++)
        {
            std::cout << initial_pose[i] << ", ";
            if (i % 4 == 3)
            {
                std::cout << std::endl;
            }
        }
        getRotationAngles();
    }


    std::tuple<double, double, double> getRotationAngles(const std::array<double, 16>* custom_pose = nullptr, bool verbose = true) {
        std::array<double, 16> pose;
        if (custom_pose) {
            pose = *custom_pose;
        } else {
            const franka::RobotState& robot_state = robot.readOnce();
            pose = robot_state.O_T_EE;
        }

        double Beta = atan2(-pose[2], sqrt(pow(pose[0], 2) + pow(pose[1], 2)));
        double Alpha = 0.0;
        double Gamma = 0.0;
        double cosBeta = cos(Beta);
        if (abs(cosBeta) < 1e-5) {
            if (Beta > 0){
                Beta = M_PI_2;
                Gamma = atan2(pose[4], pose[5]);
            }else{
                Beta = -M_PI_2;
                Gamma = -atan2(pose[4], pose[5]);
            }
        }else{
            Alpha = atan2(pose[1]/cosBeta, pose[0]/cosBeta);
            Gamma = atan2(pose[6]/cosBeta, pose[10]/cosBeta);
        }

        if (verbose) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Alpha(Z): " << Alpha << ", Beta(Y): " << Beta << ", Gamma(X): " << Gamma << " radians" << std::endl;
            // in degrees
            std::cout << "Alpha(Z): " << Alpha * 180 / M_PI << ", Beta(Y): " << Beta * 180 / M_PI << ", Gamma(X): " << Gamma * 180 / M_PI << " degrees" << std::endl;
            // xyz coordinates
            std::cout << "X: " << pose[12] << ", Y: " << pose[13] << ", Z: " << pose[14] << " meters" <<  std::endl;
        }
        return std::make_tuple(Alpha, Beta, Gamma);
    }


    bool isValidCartesianPose(const std::vector<float>& numbers) {
        const double sphere_radius = 0.855;
        double xf = numbers[0];
        double yf = numbers[1];
        double zf = numbers[2];
        if (pow(xf, 2) + pow(yf, 2) + pow(zf, 2) > pow(sphere_radius, 2)){
            throw std::runtime_error("The desired position is outside the workspace.");
            return false;
        }
        if (zf < 0.015){
            throw std::runtime_error("The desired position is too close to the table.");
            return false;
        }
        return true;
    }

    bool isValidJointPose(const std::array<double, 7>& q) {
        // Franka Emika Panda joint limits (radians)
        static const std::array<double, 7> q_min = {
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
        };
        static const std::array<double, 7> q_max = {
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
        };
        for (int i = 0; i < 7; ++i) {
            if (q[i] < q_min[i] || q[i] > q_max[i]) {
                std::cerr << "Joint " << (i + 1) << " target " << q[i]
                          << " rad outside ["
                          << q_min[i] << ", " << q_max[i] << "]" << std::endl;
                return false;
            }
        }
        return true;
    }

    // Cubic joint trajectory to a 7-DOF target. Accepts 7 joint angles
    // (rad); optional 8th value is tf, clamped to >= default_motion_time_.
    // Returns measured final q.
    std::vector<double> moveToJointPose(const std::vector<float> &numbers) {
        std::lock_guard<std::mutex> lock(robot_mutex);
        std::vector<double> final_q(7, 0.0);
        if (numbers.size() < 7) {
            throw std::runtime_error(
                "moveToJointPose requires at least 7 joint angles (rad).");
        }
        std::array<double, 7> q_target_raw;
        for (int i = 0; i < 7; ++i) {
            q_target_raw[i] = static_cast<double>(numbers[i]);
        }
        if (!isValidJointPose(q_target_raw)) {
            throw std::runtime_error(
                "Joint target outside Franka joint limits.");
        }
        double tf = default_motion_time_;
        if (numbers.size() >= 8) {
            tf = static_cast<double>(numbers[7]);
            if (tf < default_motion_time_) {
                tf = default_motion_time_;
            }
        }
        std::array<double, 7> q_initial{};
        double time = 0.0;
        const std::array<double, 7> q_target = q_target_raw;

        try {
            robot.control(
                [&time, &q_initial, q_target, tf]
                (const franka::RobotState& state, franka::Duration period)
                    -> franka::JointPositions {
                    time += period.toSec();
                    if (time == 0.0) {
                        q_initial = state.q;
                    }
                    std::array<double, 7> q_cur = q_initial;
                    double t2 = pow(time, 2);
                    double t3 = pow(time, 3);
                    double tf2 = pow(tf, 2);
                    double tf3 = pow(tf, 3);
                    for (int i = 0; i < 7; ++i) {
                        double delta = q_target[i] - q_initial[i];
                        double a2 = 3.0 * delta / tf2;
                        double a3 = -2.0 * delta / tf3;
                        q_cur[i] = q_initial[i] + a2 * t2 + a3 * t3;
                    }
                    franka::JointPositions output = q_cur;
                    if (time >= tf) {
                        std::cout << time
                                  << "sec : End of joint motion ........."
                                  << std::endl;
                        return franka::MotionFinished(output);
                    }
                    return output;
                }
            );
        } catch (const franka::Exception &ex) {
            std::cerr << "franka::Exception during moveToJointPose: "
                      << ex.what() << std::endl;
            try {
                robot.automaticErrorRecovery();
            } catch (const franka::Exception &re) {
                throw std::runtime_error(
                    std::string("collision_recovery_failed: original=")
                    + ex.what() + " recovery=" + re.what());
            }
            throw std::runtime_error(
                std::string("collision_recovery: ") + ex.what());
        }

        franka::RobotState st = robot.readOnce();
        for (int i = 0; i < 7; ++i) {
            final_q[i] = st.q[i];
        }
        return final_q;
    }

    // moveToCartesian accepts three float numbers in a vector
    std::vector<double> moveToCartesian(const std::vector<float> &numbers)
    {
        std::lock_guard<std::mutex> lock(robot_mutex);
        std::vector<double> final_coords(6, 0.0);
        if (numbers.size() < 3)
        {
            throw std::runtime_error("Please provide at least 3 float numbers (x,y,z,t).");
            return final_coords;
        }
        if (!isValidCartesianPose(numbers))
        {
            return final_coords;
        }
        double xf = numbers[0];
        double yf = numbers[1];
        double zf = numbers[2];
        std::array<double, 16> initial_pose;
        std::array<double, 16> final_pose;
        double time = 0.0;
        double tf = 0.0;
        double Alpha = 0.0, Beta = 0.0, Gamma = 0.0l;
        double deltaAlpha = 0.0, deltaBeta = 0.0, deltaGamma = 0.0;
        double Alphaf = 0.0, Betaf = 0.0, Gammaf = 0.0;
        bool is_rotation = false;
        if (numbers.size() < 4)
        {
            tf = default_motion_time_;
        }
        else if (numbers.size() >= 4)
        {
            tf = numbers[3];
            if (tf < default_motion_time_)
            {
                tf = default_motion_time_;
            }
        }

        if (numbers.size() >= 5)
        {
            deltaAlpha = numbers[4];
            // must be between -90 to 90 degrees
            if (deltaAlpha < -90 || deltaAlpha > 90)
            {
                deltaAlpha = 0.0;
                std::cerr << "Error: deltaAlpha must be between -90 and 90 degrees." << std::endl;
            }
            deltaAlpha = deltaAlpha * M_PI / 180;  // in radians
            is_rotation = true;
        }

        if (numbers.size() >= 6)
        {
            deltaBeta = numbers[5];
            // must be between -90 and 90 degrees
            if (deltaBeta < -90 || deltaBeta > 90)
            {
                deltaBeta = 0.0;
                std::cerr << "Error: deltaBeta must be between -90 and 90 degrees." << std::endl;
            }
            deltaBeta = deltaBeta * M_PI / 180;  // in radians
        }
        if (numbers.size() >= 7)
        {
            deltaGamma = numbers[6];
            // must be between -90 and 90 degrees
            if (deltaGamma < -90 || deltaGamma > 90)
            {
                deltaGamma = 0.0;
                std::cerr << "Error: deltaGamma must be between -90 and 90 degrees." << std::endl;
            }
            deltaGamma = deltaGamma * M_PI / 180;  // in radians
        }

        try
        {
            auto trajectory_callback = [this, xf, yf, zf, tf, is_rotation, deltaAlpha, deltaBeta, deltaGamma,
                                        &time, &Alpha, &Beta, &Gamma, &Alphaf, &Betaf, &Gammaf, &initial_pose, &final_pose](
                                           const franka::RobotState &robot_state,
                                           franka::Duration period) -> franka::CartesianPose
            {
                time += period.toSec();

                if (time == 0.0)
                {
                    // Read the initial pose to start the motion from in the first time step.
                    initial_pose = robot_state.O_T_EE;
                    for(int i = 0; i < 16; i++)
                    {
                        std::cout << initial_pose[i] << ", ";
                        if (i % 4 == 3)
                        {
                            std::cout << std::endl;
                        }
                    }
                    if (is_rotation){
                        std::tuple<double, double, double> angles = getRotationAngles(&initial_pose, false);
                        Alpha = std::get<0>(angles);
                        Beta = std::get<1>(angles);
                        Gamma = std::get<2>(angles);

                        Alphaf = Alpha + deltaAlpha;
                        Betaf = Beta + deltaBeta;
                        Gammaf = Gamma + deltaGamma;
                    }
                }

                // cubic polynomial trajectory
                franka::CartesianPose pose_desired = initial_pose;
                double t2 = pow(time, 2);
                double t3 = pow(time, 3);
                double tf2 = pow(tf, 2);
                double tf3 = pow(tf, 3);

                double x0 = pose_desired.O_T_EE[12];
                double a2 = 3 * (xf - x0) / tf2;
                double a3 = -2 * (xf - x0) / tf3;
                double xt = x0 + a2 * t2 + a3 * t3;
                double vx = 2 * a2 * time + 3 * a3 * t2;

                double y0 = pose_desired.O_T_EE[13];
                a2 = 3 * (yf - y0) / tf2;
                a3 = -2 * (yf - y0) / tf3;
                double yt = y0 + a2 * t2 + a3 * t3;
                double vy = 2 * a2 * time + 3 * a3 * t2;

                double z0 = pose_desired.O_T_EE[14];
                a2 = 3 * (zf - z0) / tf2;
                a3 = -2 * (zf - z0) / tf3;
                double zt = z0 + a2 * t2 + a3 * t3;
                double vz = 2 * a2 * time + 3 * a3 * t2;
                bool stop = fabs(vx) < 0.0001 && fabs(vy) < 0.0001 && fabs(vz) < 0.0001 && time > 1.0;

                pose_desired.O_T_EE[12] = xt;
                pose_desired.O_T_EE[13] = yt;
                pose_desired.O_T_EE[14] = zt;

                if(is_rotation){
                    a2 = 3 * (Alphaf - Alpha) / tf2;
                    a3 = -2 * (Alphaf - Alpha) / tf3;
                    double Alphat = Alpha + a2 * t2 + a3 * t3;
                    double vAlpha = 2 * a2 * time + 3 * a3 * t2;

                    a2 = 3 * (Betaf - Beta) / tf2;
                    a3 = -2 * (Betaf - Beta) / tf3;
                    double Betat = Beta + a2 * t2 + a3 * t3;
                    double vBeta = 2 * a2 * time + 3 * a3 * t2;

                    a2 = 3 * (Gammaf - Gamma) / tf2;
                    a3 = -2 * (Gammaf - Gamma) / tf3;
                    double Gammat = Gamma + a2 * t2 + a3 * t3;
                    double vGamma = 2 * a2 * time + 3 * a3 * t2;

                    // Rotation matrix (ZYX Euler: Alpha=Z, Beta=Y, Gamma=X)
                    pose_desired.O_T_EE[0] = cos(Alphat) * cos(Betat);
                    pose_desired.O_T_EE[1] = sin(Alphat) * cos(Betat);
                    pose_desired.O_T_EE[2] = -sin(Betat);
                    pose_desired.O_T_EE[4] = cos(Alphat) * sin(Betat) * sin(Gammat) - sin(Alphat) * cos(Gammat);
                    pose_desired.O_T_EE[5] = sin(Alphat) * sin(Betat) * sin(Gammat) + cos(Alphat) * cos(Gammat);
                    pose_desired.O_T_EE[6] = cos(Betat) * sin(Gammat);
                    pose_desired.O_T_EE[8] = cos(Alphat) * sin(Betat) * cos(Gammat) + sin(Alphat) * sin(Gammat);
                    pose_desired.O_T_EE[9] = sin(Alphat) * sin(Betat) * cos(Gammat) - cos(Alphat) * sin(Gammat);
                    pose_desired.O_T_EE[10] = cos(Betat) * cos(Gammat);
                    stop = stop && fabs(vAlpha) < 0.0001 && fabs(vBeta) < 0.0001 && fabs(vGamma) < 0.0001;
                }

                if (time >= tf || stop)
                {
                    std::cout << std::endl << time << "sec : End of motion ............." << std::endl;
                    final_pose = pose_desired.O_T_EE;
                    return franka::MotionFinished(pose_desired);
                }

                return pose_desired;
            };

            robot.control(trajectory_callback);
        }
        catch (const franka::Exception &ex)
        {
            std::cerr << "franka::Exception during motion: " << ex.what() << std::endl;

            // Unlock the robot so it can accept new commands.
            // The Python side is responsible for any recovery motion after this.
            try {
                std::cerr << "Attempting automatic error recovery..." << std::endl;
                robot.automaticErrorRecovery();
                std::cerr << "Robot unlocked. Returning 400 to caller." << std::endl;
            } catch (const franka::Exception& recovery_ex) {
                std::cerr << "Error recovery failed: " << recovery_ex.what() << std::endl;
                throw std::runtime_error(
                    std::string("collision_recovery_failed: original=") + ex.what() +
                    " recovery=" + recovery_ex.what());
            }

            // "collision_recovery:" prefix is parsed by the Python client.
            throw std::runtime_error(std::string("collision_recovery: ") + ex.what());
        }

        final_pose = robot.readOnce().O_T_EE;
        // Reuse the cached pose to skip a second readOnce inside getRotationAngles.
        std::tuple<double, double, double> final_angles = getRotationAngles(&final_pose);
        final_coords[0] = final_pose[12];
        final_coords[1] = final_pose[13];
        final_coords[2] = final_pose[14];
        final_coords[3] = std::get<0>(final_angles) * 180.0 / M_PI;  // Alpha (Z) in degrees
        final_coords[4] = std::get<1>(final_angles) * 180.0 / M_PI;  // Beta (Y) in degrees
        final_coords[5] = std::get<2>(final_angles) * 180.0 / M_PI;  // Gamma (X) in degrees
        return final_coords;
    }

    std::string closeGripper(const std::vector<float> &numbers) {
        std::lock_guard<std::mutex> lock(robot_mutex);
        try {
            // GUI sends `{"closeGripper": []}`; indexing the empty vector is
            // UB, so default to 0.01 m so the fingers squeeze small objects.
            double grasping_width = numbers.empty() ? 0.01 : static_cast<double>(numbers[0]);
            franka::GripperState gripper_state = gripper.readOnce();
            if (gripper_state.max_width < grasping_width) {
                return "Object is too large for the current fingers on the gripper: " + std::to_string(gripper_state.max_width);
            }
            // 40 N force + wide epsilon so small objects (~0.5-6 cm) clamp
            // firmly regardless of exact width.
            if (!gripper.grasp(grasping_width, 0.1, 40.0, 0.05, 0.05)) {
                return "Failed to grasp object.";
            }
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
            gripper_state = gripper.readOnce();
            if (!gripper_state.is_grasped) {
                return "Object lost.";
            }
        } catch (franka::Exception const& e) {
            return e.what();
        }
        return "Object grasped successfully.";
    }

    std::string openGripper(const std::vector<float> &numbers)
    {
        std::lock_guard<std::mutex> lock(robot_mutex);
        try
        {
            franka::GripperState gripper_state = gripper.readOnce();
            // Empty-vector guard: GUI sends `{"openGripper": []}`.
            double speed = numbers.empty() ? 0.1 : static_cast<double>(numbers[0]);
            std::cout << "Grasped object, will release it now." << std::endl;
            gripper.move(gripper_state.max_width, speed);
            gripper.stop();
        }
        catch (franka::Exception const &e)
        {
            return e.what();
        }
        return "Gripper opened successfully.";
    }

    // Current EE pose: [x, y, z, alpha_deg, beta_deg, gamma_deg].
    std::vector<double> readState()
    {
        std::lock_guard<std::mutex> lock(robot_mutex);
        const franka::RobotState& state = robot.readOnce();
        std::array<double, 16> pose = state.O_T_EE;
        auto [alpha, beta, gamma] = getRotationAngles(&pose, false);
        return {pose[12], pose[13], pose[14],
                alpha * 180.0 / M_PI,
                beta  * 180.0 / M_PI,
                gamma * 180.0 / M_PI};
    }

    // Current joint angles [q1..q7] in radians.
    std::vector<double> readJointState()
    {
        std::lock_guard<std::mutex> lock(robot_mutex);
        const franka::RobotState& state = robot.readOnce();
        std::vector<double> q(7);
        for (int i = 0; i < 7; ++i) {
            q[i] = state.q[i];
        }
        return q;
    }
};

class RobotRestAPI {
public:
    RobotRestAPI(utility::string_t url, const std::string& robot_ip, double default_motion_time)
        : m_listener(url), robotHandler(robot_ip, default_motion_time) {
        m_listener.support(methods::POST, std::bind(&RobotRestAPI::handle_post, this, std::placeholders::_1));
        robotHandler.initialize();
    }

    pplx::task<void> open() { return m_listener.open(); }
    pplx::task<void> close() { return m_listener.close(); }

private:
    void handle_post(http_request request) {
        request
            .extract_json()
            .then([](json::value body)
                  {
                      std::map<std::string, std::vector<float>> data;
                      for (const auto& item : body.as_object())
                      {
                          if (!item.second.is_array())
                          {
                              throw std::runtime_error("All values must be arrays.");
                          }
                          std::vector<float> numbers;
                          for (const auto& num : item.second.as_array())
                          {
                              if (!num.is_number())
                              {
                                  throw std::runtime_error("All array elements must be numbers.");
                              }
                              numbers.push_back(static_cast<float>(num.as_double()));
                          }
                          data[item.first] = numbers;
                      }
                      return data;
                  })
            .then([this](std::map<std::string, std::vector<float>> data)
                  {
                json::value response;
                for (const auto& item : data)
                {
                    const std::string& key = item.first;
                    const std::vector<float>& numbers = item.second;

                    if (key == "moveToCartesian") {
                        std::vector<double> final_coords = robotHandler.moveToCartesian(numbers);
                        for (int i = 0; i < 6; ++i) {
                            response[key][i] = json::value::number(final_coords[i]);
                        }
                    } else if (key == "moveToJointPose") {
                        std::vector<double> final_q = robotHandler.moveToJointPose(numbers);
                        for (int i = 0; i < 7; ++i) {
                            response[key][i] = json::value::number(final_q[i]);
                        }
                    } else if (key == "closeGripper") {
                        std::string message = robotHandler.closeGripper(numbers);
                        response[key] = json::value::string(message);
                    } else if (key == "openGripper") {
                        std::string message = robotHandler.openGripper(numbers);
                        response[key] = json::value::string(message);
                    } else if (key == "readState") {
                        std::vector<double> state = robotHandler.readState();
                        for (int i = 0; i < 6; ++i) {
                            response[key][i] = json::value::number(state[i]);
                        }
                    } else if (key == "readJointState") {
                        std::vector<double> q = robotHandler.readJointState();
                        for (int i = 0; i < 7; ++i) {
                            response[key][i] = json::value::number(q[i]);
                        }
                    } else {
                        std::cout << "Invalid command: " << key << std::endl;
                        response["Response"] = json::value::string("Invalid command.");
                    }
                }
                return response; })
            .then([=](json::value response)
                  { request.reply(status_codes::OK, response); })
            .then([=](pplx::task<void> t)
                  {
                try {
                    t.get();
                }
                catch (const std::exception &e) {
                    request.reply(status_codes::BadRequest, json::value::string(e.what()));
                } });
    }

    http_listener m_listener;
    RobotHandler robotHandler;
};


#include <iostream>

int main(int argc, char* argv[]) {
    std::string port = "34568";
    std::string address = "http://0.0.0.0:" + port;
    utility::string_t utility_address = utility::conversions::to_string_t(address);
    std::string robot_ip = "192.168.2.100";
    double default_motion_time = 1.0;

    auto print_usage = [&](std::ostream& os) {
        os << "Usage: " << argv[0] << " [--robot-ip <ip>] "
           << "[--default-motion-time <seconds>] [-h|--help]\n"
           << "  --robot-ip <ip>                Franka robot IP (default: 192.168.2.100)\n"
           << "  --default-motion-time <sec>    Default/min trajectory duration (default: 1.0)\n";
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(std::cout);
            return 0;
        } else if (arg == "--robot-ip") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --robot-ip requires a value.\n";
                print_usage(std::cerr);
                return 1;
            }
            robot_ip = argv[++i];
        } else if (arg == "--default-motion-time") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --default-motion-time requires a value.\n";
                print_usage(std::cerr);
                return 1;
            }
            try {
                default_motion_time = std::stod(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: --default-motion-time must be a number, got '"
                          << argv[i] << "'.\n";
                print_usage(std::cerr);
                return 1;
            }
            if (default_motion_time <= 0.0) {
                std::cerr << "Error: --default-motion-time must be > 0, got "
                          << default_motion_time << ".\n";
                print_usage(std::cerr);
                return 1;
            }
        } else {
            std::cerr << "Error: unknown argument '" << arg << "'.\n";
            print_usage(std::cerr);
            return 1;
        }
    }

    RobotRestAPI api(utility_address, robot_ip, default_motion_time);

    try {
        api.open().wait();
        std::cout << "Listening at: " << address << std::endl;
        std::cout << "Press ENTER to exit." << std::endl;
        std::string line;
        std::getline(std::cin, line);
        api.close().wait();
    }
    catch (std::exception const & e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
