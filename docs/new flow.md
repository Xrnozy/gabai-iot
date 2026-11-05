flowchart TB
    subgraph PI["RASPBERRY PI"]
        L["Left Camera"]
        R["Right Camera"]
        I["IMU"]
        SYNC["Timestamp Synchronization"]

        L --> SYNC
        R --> SYNC
        I --> SYNC

        SYNC --> KLT["KLT Feature Detection and Tracking"]
        KLT --> KLTQ["Forward-Backward Tracking Validation"]
        SYNC --> TX["Lightweight Data Transmission"]
        KLTQ --> TX

        TX["Transmit Left + Right Frames, IMU + KLT Tracks"]
    end

    subgraph GPU["GPU SERVER"]
        TX --> BUF["Synchronized Stereo Frame Buffer"]

        BUF --> FS["FoundationStereo + TensorRT"]
        FS --> DEPTH["Metric Depth Map"]
        FS --> CONF["Depth Confidence Map"]

        BUF --> KLTVAL["KLT Track Quality Monitoring"]
        KLTVAL --> VALID["Tracking and Motion Consistency Validation"]

        BUF --> ODOM["RTAB-Map Stereo Visual Odometry"]
        I --> IMUF["IMU Orientation / Gravity Assistance"]
        IMUF --> ODOM

        ODOM --> POSE["Camera Pose + Odometry Quality"]

        DEPTH --> DVALID["Depth Validation and Filtering"]
        CONF --> DVALID
        POSE --> DVALID

        DVALID --> RGBD["RGB + Metric Depth + Confidence"]
    end

    subgraph SLAM["RTAB-MAP SLAM"]
        RGBD --> NODE["RGB-D Mapping Input"]
        POSE --> NODE

        NODE --> FEATURES["Visual Feature Processing"]
        NODE --> GRAPH["Pose Graph"]
        FEATURES --> LOOP["Loop Closure Detection"]
        LOOP --> GRAPH
        GRAPH --> OPT["Graph Optimization"]
    end

    subgraph EGO["EGOCENTRIC PERCEPTION"]
        RGBD --> DET["Object Detection"]
        DET --> TRACK["Object Tracking and ID Association"]
        POSE --> TRACK

        TRACK --> DEPTH3D["3D Object Position Estimation"]
        CONF --> DEPTH3D
        DEPTH3D --> ASSOC["Persistent Object Association"]

        ASSOC --> SCENE["Egocentric Scene Graph"]
        SCENE --> REL["Spatial Relationships"]
        REL --> REL1["Objects on Tables"]
        REL --> REL2["Objects Near Chairs / Desks"]
        REL --> REL3["Object-to-Object Relationships"]
    end

    subgraph OUTPUT["OUTPUT"]
        OPT --> MAP["Optimized Global Map"]
        MAP --> CLOUD["Persistent 3D Point Cloud"]
        MAP --> TRAJ["Corrected Camera Trajectory"]

        ASSOC --> OBJECTMAP["Persistent 3D Object Map"]
        SCENE --> SCENEOUT["Egocentric Scene Understanding"]
    end