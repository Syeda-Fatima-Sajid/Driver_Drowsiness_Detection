import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.set_page_config(
    page_title="WebRTC Test",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Webcam Test")
st.write("Testing Streamlit WebRTC on Community Cloud")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

webrtc_streamer(
    key="webcam",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

st.success("If webcam starts successfully, WebRTC is working.")
