import base64

import requests
import streamlit as st
from PIL import Image
from io import BytesIO

if 'im' not in st.session_state:
    st.session_state.im = None


def img_to_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("Image Studio")


def generate_image_stable_diffusion(prompt: str, model: str, width, height, negative_prompt, im_input, strength):
    api_key = st.secrets["STABILITY_API_KEY"]
    from fractions import Fraction
    frac = Fraction(width, height)
    aspect_ratio = f"{frac.numerator}:{frac.denominator}"
    body = {
        "prompt": prompt,
        "output_format": "png",
        "model": model,
        "negative_prompt": negative_prompt,
    }
    files = {"none": ''}
    if im_input is not None:
        files['image'] = img_to_bytes(im_input)
        body['mode'] = 'image-to-image'
        # files['image'] = base64.b64encode(img_to_bytes(im_input)).decode('utf-8')
        body['strength'] = strength
    else:
        body['aspect_ratio'] = aspect_ratio
    resp = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "authorization": f"Bearer {api_key}",
            "accept": "image/*"
        },
        files=files,
        data=body,
    )
    if resp.status_code != 200:
        print(resp.content)
        breakpoint()
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def generate_image_bfl(prompt: str, model: str, width, height, negative_prompt):
    import requests

    url = f"https://api.bfl.ml/v1/{model}"

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "prompt_upsampling": False,
        "seed": 42,
        "safety_tolerance": 6,
        "output_format": "png",
        "negative_prompt": negative_prompt
    }
    headers = {
        "Content-Type": "application/json",
        "X-Key": st.secrets["BFL_API_KEY"]
    }

    response = requests.post(url, json=payload, headers=headers)

    response.raise_for_status()

    id = response.json()["id"]

    url = "https://api.bfl.ml/v1/get_result"

    querystring = {"id": id}

    status = "Pending"

    while status == "Pending":
        response = requests.get(url, params=querystring)
        response.raise_for_status()
        task = response.json()
        status = task["status"]

    print(task)
    response = requests.get(task["result"]["sample"])
    response.raise_for_status()

    return Image.open(BytesIO(response.content))


def img_to_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_image_fal(prompt, model, width, height, negative_prompt) -> Image.Image:
    _, __ = model, negative_prompt
    submit_resp = requests.post(
        "https://queue.fal.run/fal-ai/flux-pro/v1.1",
        json={
            "prompt": prompt,
            "image_size": {
                "width": width,
                "height": height
            },
            "enable_safety_checker": False,
            "output_format": "png",
            "safety_tolerance": "6",
        },
        headers={
            "Authorization": f"Key {st.secrets['FAL_API_KEY']}"
        },
    )
    # breakpoint()

    submit_resp.raise_for_status()
    request_id = submit_resp.json()["request_id"]
    status = None
    while status != "COMPLETED":
        status_resp = requests.get(
            f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}/status",
            headers={
                "Authorization": f"Key {st.secrets['FAL_API_KEY']}"
            }
        )
        # breakpoint()
        status_resp.raise_for_status()
        status_json = status_resp.json()
        status = status_json["status"]
        import time
        # time.sleep(1)
    result_resp = requests.get(
        f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}",
        headers={
            "Authorization": f"Key {st.secrets['FAL_API_KEY']}"
        }
    )
    # breakpoint()

    result_resp.raise_for_status()
    result = result_resp.json()
    images = result["images"]
    image = images[0]
    image_url = image["url"]
    return Image.open(requests.get(image_url, stream=True).raw)


def main():
    with st.sidebar, st.form("sd"):
        provider = ...
        prompt = st.text_area("Prompt")
        negative_prompt = st.text_area("Negative Prompt")
        width = st.number_input("Width", step=1, min_value=256, max_value=2048, value=1920)
        height = st.number_input("Height", step=1, min_value=256, max_value=2048, value=1080)
        model = st.selectbox("Model", ["sd3.5-large", "sd3.5-large-turbo", "flux-pro-1.1", "flux-pro", "fal"][::-1])
        img = st.file_uploader("Image File", type=["png", "jpg", "jpeg", "webp"])
        if img is not None:
            img = Image.open(img)
        strength = st.slider('Strength', min_value=0., max_value=1., step=0.01)
        submit = st.form_submit_button("Generate")

    # cache_key = (provider, model, prompt, width, height)
    if submit:
        st.session_state.im = None
        with st.spinner("Generating image..."):
            if model.startswith("sd"):
                st.session_state.im = generate_image_stable_diffusion(
                    prompt, model, width, height, negative_prompt, img, strength
                )
            elif model.startswith("fal"):
                st.session_state.im = generate_image_fal(prompt, model, width, height, negative_prompt)
            else:
                st.session_state.im = generate_image_bfl(prompt, model, width, height, negative_prompt)

    if st.session_state.im:
        st.image(st.session_state.im, use_column_width=True)
        st.download_button("Download", img_to_bytes(st.session_state.im), mime="image/png")


if __name__ == "__main__":
    main()
