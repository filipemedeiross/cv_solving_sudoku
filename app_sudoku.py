import cv2
import numpy as np
import streamlit as st
from keras.models import load_model

from cv import preprocess,      \
               biggest_contour, \
               get_perspective, \
               split_cells,     \
               get_grid,        \
               SHAPE
from sudoku import solver_backtracking


# Declaring cached functions

@st.cache_resource
def load_digit_model(model_path):
    return load_model(model_path)

# Configurations

st.set_page_config(layout='wide')

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 32px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="title">SOLVING A SUDOKU GAME</h1>', unsafe_allow_html=True)

col1, _, col3, _ = st.columns([5, 1, 3, 1])

# Streamlit app

model_path = 'models\\digits.keras'
model = load_digit_model(model_path)

with col1:
    bytes = st.camera_input('')

if bytes is not None:
    with col3:
        image = cv2.imdecode(np.frombuffer(bytes.getvalue(), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        process = preprocess(image)
        contour = biggest_contour(process)

        if contour is None:
            st.error('No valid contour were found. Try taking a clearer picture.')
        else:
            try:
                image_grid   = get_perspective(cv2.resize(image  , SHAPE), SHAPE, contour)
                image_sudoku = get_perspective(cv2.resize(process, SHAPE), SHAPE, contour)

                st.image(image_grid, use_column_width=True)
            except Exception as e:
                st.error('Error transforming perspective.')

nes1, nes2 = st.columns([1, 1])

with nes1:
    cells = split_cells(image_sudoku)
    grid  = get_grid(cells, model)
    grid  = st.data_editor(grid,
                            hide_index=True,
                            column_config=None)

with nes2:
    if st.button('Solving sudoku'):
        answer = solver_backtracking(grid)

        if answer is not None:
            st.dataframe(answer)
        else:
            st.write('No solution.')
