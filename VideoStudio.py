from io import BytesIO

import streamlit as st
from PIL import Image
from duckdb.experimental.spark.sql.functions import struct

if 'video' not in st.session_state:
    st.session_state.video = None
    st.session_state.content_type = None


def img_to_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_video(image: Image.Image, cfg_scale: float = 1.8, motion_bucket_id: int = 127) -> tuple[bytes, str]:
    """
    Accepts image and generation parameters. Returns video data and content type.
    """
    import requests

    generate_response = requests.post(
        f"https://api.stability.ai/v2beta/image-to-video",
        headers={
            "authorization": f"Bearer {st.secrets['STABILITY_API_KEY']}"
        },
        files={
            "image": BytesIO(img_to_bytes(image))
        },
        data={
            # "seed": 0,
            "cfg_scale": cfg_scale,
            "motion_bucket_id": motion_bucket_id
        },
    )
    generate_response.raise_for_status()
    task = generate_response.json()
    generation_id = task['id']
    while True:
        result_response = requests.get(
            f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}",
            headers={
                "authorization": f"Bearer {st.secrets['STABILITY_API_KEY']}",
                "accept": "video/*"
            }
        )
        if result_response.status_code == 200:
            break
        elif result_response.status_code == 202:
            continue
        else:
            result_response.raise_for_status()
    return result_response.content, result_response.headers['content-type']


def main():
    st.title("Video Studio")
    with st.sidebar:
        with st.form("Generate Video from Image"):
            image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
            cfg_scale = st.slider("CFG Scale", min_value=0.0, max_value=10.0, value=1.8)
            motion_bucket_id = st.slider("Motion Bucket ID", min_value=1, max_value=255, value=127)
            submit = st.form_submit_button("Generate")
    if submit and image is not None:
        image = Image.open(image)
        with st.spinner("Generating video..."):
            st.session_state.video, st.session_state.content_type = image_to_video(
                image, cfg_scale, motion_bucket_id
            )
    if st.session_state.video is not None:
        st.video(st.session_state.video)
        st.download_button("Download", st.session_state.video, mime=st.session_state.content_type)


if __name__ == "__main__":
    main()
