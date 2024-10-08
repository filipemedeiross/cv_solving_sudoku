import cv2
import numpy as np
import streamlit as st
from keras.models import load_model

from cv import preprocess,          \
               biggest_contour,     \
               reframe,             \
               get_perspective,     \
               get_perspective_inv, \
               split_cells,         \
               get_grid,            \
               SHAPE
from sudoku import solver_backtracking


# Declaring cached functions

@st.cache_resource
def load_digit_model(model_path):
    return load_model(model_path)


# Configurations

st.set_page_config(page_title='Sudoku solver', layout='wide')

path  = 'models\\digits.keras'
model = load_digit_model(path)

font = cv2.FONT_HERSHEY_SIMPLEX

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 64px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="title">SOLVING A SUDOKU GAME</h1>', unsafe_allow_html=True)


# Streamlit app

_, col, _ = st.columns([1, 9, 1])

with col:
    st.markdown('<h2>Take a picture</h2>', unsafe_allow_html=True)

    enable = st.checkbox('Enable camera')
    bytes  = st.camera_input('',
                             disabled=not enable,
                             label_visibility='collapsed')

if bytes is not None:
    image = cv2.imdecode(np.frombuffer(bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    process = preprocess(gray)
    contour = biggest_contour(process)

    if contour is None:
        with col:
            st.error('No valid contour were found.')
    else:
        image_grid   = get_perspective(cv2.resize(image  , SHAPE), SHAPE, contour)
        image_sudoku = get_perspective(cv2.resize(process, SHAPE), SHAPE, contour)

        with col:
            with st.expander('See sudoku grid.'):
                _, colx1, _ = st.columns([3, 3, 3])
                colx1.image(image_grid)

            st.markdown('<h2>Solving the sudoku</h2>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([4, 1, 4])

            with col1:
                cells = split_cells(image_sudoku)
                grid  = get_grid(cells, model)
                grid  = st.data_editor(grid,
                                       hide_index=True,
                                       column_config=None)

            if col2.button('Solve'):
                with col3:
                    answer = solver_backtracking(grid)

                    if answer is not None:
                        st.dataframe(answer)
                    else:
                        st.write('No solution.')

                with st.expander('See the answer in augmented reality.', expanded=True):
                    for grd, ans, cell in zip(grid.flatten()  ,
                                              answer.flatten(),
                                              split_cells(image_grid)):
                        if not grd:
                            cv2.putText(cell, str(ans), (10, 40), font,
                                        1, (0, 0, 255), 2, cv2.LINE_AA)

                    resiz = cv2.resize(image, SHAPE)
                    warp  = get_perspective_inv(image_grid, SHAPE, contour)
                    aug_image = cv2.addWeighted(resiz, 0.7, warp, 0.3, 0)

                    _, colx2, _ = st.columns([3, 3, 3])
                    with colx2:
                         st.image(aug_image)
