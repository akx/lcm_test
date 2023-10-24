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
        result = get_generator().generate(
            prompt=prompt,
            width=width,
            height=height,
            batch_size=batch_size,
            cfg=cfg,
            steps=steps,
            seed=seed,
        )
        st.write("Seed:", result.seed)
        for image in result.images:
            st.image(image)
