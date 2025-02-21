# This script is designed to synchronize a motorsports control lap with live laps (from youtube or m3u8)
# so that coaches can visualize how their drivers are doing in real time.
# When there are no live races being streamed, the script can be used for comparing past 
# laps for post-race analysis and driver feedback.
# 
# The main synchronizes the laps at the start/finish line so that the videos start at the same place


import cv2 as cv
import streamlink
import time

CONTROL_PATH = 'lando.mp4'
LIVE_PATH_OR_LINK = 'max.mp4'
CONTROL_MARKER_PATH = 'control_marker.jpg'
# live marker can use control's marker in some cases if there is a close match
LIVE_MARKER_PATH = 'live_marker.jpg'


def get_youtube_stream(url):
    '''Get the live video stream URL using streamlink.'''
    streams = streamlink.streams(url)
    if 'best' in streams:
        return streams['best'].url
    else:
        raise Exception("No livestream found")


def detect_marker(frame, marker_template, threshold=0.99):
    """Detect the marker in a frame using template matching."""
    result = cv.matchTemplate(frame, marker_template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    #print(max_val)
    if max_val >= threshold:
        return max_val
    return None


def synchronize(video, marker_template):
    """Synchronize the start of the lap for control video."""
    marker_frame = None

    # Find the first marker frame in the video
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if detect_marker(frame, marker_template):
            marker_frame = int(video.get(cv.CAP_PROP_POS_FRAMES)) - 1
            break

    if marker_frame is None:
        raise Exception("Marker not found in control video")

    # Set the video to the marker frame
    video.set(cv.CAP_PROP_POS_FRAMES, marker_frame)

    return marker_frame


def main():
    control = cv.VideoCapture(CONTROL_PATH)

    # if link is from youtbe - uncomment next line
    #LIVE_PATH_OR_LINK = get_youtube_stream(LIVE_PATH_OR_LINK)
    live = cv.VideoCapture(LIVE_PATH_OR_LINK)

    # Marker image (customize per track)
    control_marker = cv.imread(CONTROL_MARKER_PATH)
    if control_marker is None:
        raise Exception("Control Marker template not found")
    live_marker = cv.imread(LIVE_MARKER_PATH)
    if live_marker is None:
        raise Exception("Live Marker template not found")

    # Find start/finish frame on control lap
    control_marker_frame = synchronize(control, control_marker)
    print('finish sync')

    width = int(control.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(control.get(cv.CAP_PROP_FRAME_HEIGHT))

    view_mode = 0
    live_marker_detected = False
    last_detect_time = 0
    max_val = 0
    peak_frame = None
    lap_delay = 62  # Customize for each track

    while True:
        ret1, frame1 = control.read()
        ret2, frame2 = live.read()

        if not ret2:  # Stop if live video ends
            break
        
        if not ret1:  # Restart the control video if it ends
            control.set(cv.CAP_PROP_POS_FRAMES, control_marker_frame)
            ret1, frame1 = control.read()

        # Detect marker in live video to handle new laps
        #
        # We only want to search for the start/finish line after
        # the drivers have completed most of the lap
        current_time = time.time()
        if current_time - last_detect_time > lap_delay:
            scale_factor = 1      # scale as little as possible w/o slowing down livestream, will affect threshold
            small_frame2 = cv.resize(frame2, None, fx=scale_factor, fy=scale_factor)
            small_marker = cv.resize(live_marker, None, fx=scale_factor, fy=scale_factor)
            current_val = detect_marker(small_frame2, small_marker, 0.97)
            #print(current_val)

            #current_val = detect_marker(frame2, live_marker, 0.8)

            if current_val:
                live_marker_detected = True

                if current_val >= max_val:
                    max_val = current_val
                    peak_frame = frame2
                elif current_val < max_val:
                    live_marker = peak_frame
                    print("New lap detected in live video. Restarting control lap.")
                    control.set(cv.CAP_PROP_POS_FRAMES, control_marker_frame)
                    ret1, frame1 = control.read()
                    
                    last_detect_time = current_time  # Update the last detection time
                    max_val = 0
                    live_marker_detected = False
            
            # extra branch in case final frame doesn't meet threshold
            elif live_marker_detected:
                live_marker = peak_frame
                print("New lap detected in live video. Restarting control lap.")
                control.set(cv.CAP_PROP_POS_FRAMES, control_marker_frame)
                ret1, frame1 = control.read()
                
                last_detect_time = current_time  # Update the last detection time
                max_val = 0
                live_marker_detected = False
        
        # Convert frame1 to grayscale
        frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        # Convert grayscale frame1 to a 3-channel image to blend with frame2
        frame1_gray_colored = cv.cvtColor(frame1_gray, cv.COLOR_GRAY2BGR)

        # Resize frame2 to match frame1's size (if needed)
        frame2 = cv.resize(frame2, (width, height))

        # Display side-by-by comparison
        if view_mode == 0:
            display_frame = cv.hconcat([frame1, frame2])
        # Display overlaid video
        else:
            display_frame = cv.addWeighted(frame1_gray_colored, 0.6, frame2, 1.0, 0)

        # Show the frame
        cv.imshow('Video Overlay', display_frame)

        # Keyboard controls
        key = cv.waitKey(20) & 0xFF
        if key == ord('d'):  # Exit on pressing 'd'
            break
        elif key == ord('t'):  # Toggle view mode on pressing 't'
            view_mode = 1 - view_mode  # Toggle between 0 and 1

    control.release()
    live.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()