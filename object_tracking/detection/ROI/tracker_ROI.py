import cv2

class tracker_ROI:

    np = __import__("numpy")

    ''' MIL Tracker: Better accuracy than BOOSTING tracker but does a poor job of reporting failure. (minimum OpenCV 3.0.0)
    KCF Tracker: Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well. (minimum OpenCV 3.1.0)
    CSRT Tracker: Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than KCF but slightly slower. (minimum OpenCV 3.4.2)
    MedianFlow Tracker: Does a nice job reporting failures; however, if there is too large of a jump in motion, such as fast moving objects, or objects that change quickly in their appearance, the model will fail. (minimum OpenCV 3.0.0)
    TLD Tracker: I’m not sure if there is a problem with the OpenCV implementation of the TLD tracker or the actual algorithm itself, but the TLD tracker was incredibly prone to false-positives. I do not recommend using this OpenCV object tracker. (minimum OpenCV 3.0.0)
    MOSSE Tracker: Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed. (minimum OpenCV 3.4.1)
    GOTURN Tracker: The only deep learning-based object detector included in OpenCV. It requires additional model files to run. My initial experiments showed it was a bit of a pain to use even though it reportedly handles viewing changes well. (minimum OpenCV 3.2.0) '''


    tracker_name = "CSRT"
    tracking_area = (100,100)

    # ROI selection variables
    roi_start_event = cv2.EVENT_LBUTTONDOWN
    window_name = "Main"


    #Technical variables(avoid changing)
    roi_selected = False
    event_detected = False
    point_ROI = None
    points = []


    OPENCV_OBJECT_TRACKERS = {
		"CSRT": cv2.TrackerCSRT_create,
		"KCF": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"MIL": cv2.TrackerMIL_create,
		"TLD": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create}



    def __init__(self, tracker_name = tracker_name, window_name = window_name, tracking_area = tracking_area):
        self.tracker = self.OPENCV_OBJECT_TRACKERS[tracker_name]()

        self.window_name = window_name
        self.tracking_area = tracking_area

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.event_handler)


    def event_handler(self, event, x, y, flags, params):
            if event == self.roi_start_event:
                self.event_detected = True
                self.selected_points = (x, y)
            
                roi = cv2.selectROI("ROI", self.new_frame)
                cv2.destroyWindow("ROI")

                self.roi_selected = True

                self.tracking_area = (roi[2], roi[3])

                point_roi = ((roi[0] + (roi[2]/2)) , (roi[1] + (roi[3]/2)))
                self.point_ROI = self.np.array([[ point_roi ]], dtype=self.np.float32)

                self.roi = roi


    def run(self, old_points, new_frame, old_frame):
        self.new_frame = new_frame

        if self.roi_selected:
            if self.roi is not None:
                ok = self.tracker.init(new_frame, self.roi)
                new_points = self.point_ROI
                self.roi = None

                return new_points, 1, 0

            status, new_points = self.tracker.update(new_frame)

            new_points = self.np.array([[new_points[0], new_points[1]]], dtype=self.np.float32)

            if status == 1:error = 0
            else: error = 1
                
            return new_points, status, error
        else:
            return old_points, 0, 0
