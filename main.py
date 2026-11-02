"""The project's entry point: everything is loaded first (depth model, hand and object models, Boxer), then the source and the viewer start. Same options as gabai/tools/rerun_pipeline.py.

    py main.py --source live --mode stereo_imu --model h
"""
from gabai.tools.rerun_pipeline import main

if __name__ == "__main__":
    main()
