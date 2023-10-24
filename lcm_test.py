import time

import streamlit as st

from lcm_gen import LCMGenerator


@st.cache_resource
def get_generator():
    return LCMGenerator(
        progress_context=st.spinner,
    )


with st.form("generator"):
    prompt = st.text_input("Prompt", "hyperdetailed cityscape")
    wc, hc = st.columns(2)
    with wc:
        width = st.number_input("Width", min_value=128, max_value=1024, value=768)
    with hc:
        height = st.number_input("Height", min_value=128, max_value=1024, value=512)
    steps = st.number_input("Steps", min_value=1, max_value=10, value=5)
    cfg = st.number_input(
        "CFG",
        min_value=0.5,
        max_value=30.0,
        value=9.0,
        help="Creativity, more or less",
    )
    batch_size = st.number_input("Batch size", min_value=1, max_value=10, value=1)
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=2**32,
        value=0,
        help="0 for random",
    )
    if st.form_submit_button("Generate"):
        t0 = time.time()
        result = get_generator().generate(
            prompt=prompt,
            width=width,
            height=height,
            batch_size=batch_size,
            cfg=cfg,
            steps=steps,
            seed=seed,
        )
        gen_time = time.time() - t0
        st.write("Seed:", result.seed)
        for image in result.images:
            st.image(image)
        st.write(
            f"Time: {gen_time :.2f}s "
            f"({gen_time / batch_size :.2f}s per image, "
            f"{gen_time / batch_size / steps :.2f} seconds per step)",
        )
