import streamlit as st
from streamlit.logger import get_logger
import pandas as pd


# ...

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to MandelaAir Guard! 👋")

    st.markdown(
        """
        This web page is created to monitor the air quality of locations in Nigeria. The application provides the overall Air Quality Index derived from different pollutants, namely - CO, NO, NO3, O3, SO2, NH3, PM2.5, PM10, present in the air.

        👈 Enter the longitude and latitude of a state in the sidebar to see what the air quality feels like!
    """
    )

    # Read data from CSV file
    file_path = 'nigerian_states.csv'
    state_coordinates_df = pd.read_csv(file_path)

    # Display Nigerian states and coordinates
    st.write("## Nigerian States and Coordinates")
    markdown_table = state_coordinates_df.to_markdown(index=False)
    st.markdown(markdown_table, unsafe_allow_html=True)



if __name__ == "__main__":
    run()
